"""Capture causal, fixed-depth Ouro trajectories without future-state inputs."""

import argparse
import json
import time
from pathlib import Path

import torch


def read_jsonl(path):
    with open(path, encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prompt_ids(tokenizer, record, max_prompt_tokens):
    text = record["prompt"]
    # Use the checkpoint's native template if one exists; otherwise use its
    # ordinary tokenizer. Record actual ids, so replay never reapplies a template.
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": text}],
            tokenize=True, add_generation_prompt=True, return_dict=False,
        )
    else:
        ids = tokenizer.encode(text, add_special_tokens=True)
    if len(ids) > max_prompt_tokens:
        return None
    if not ids:
        raise ValueError("empty tokenized prompt")
    return torch.tensor([ids], dtype=torch.long)


@torch.inference_mode()
def capture_one(target, input_ids, max_new_tokens, eos_ids):
    output = target.forward(input_ids)
    parts = [output.round_hidden[0].detach().cpu()]
    generated = []
    for step in range(max_new_tokens):
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        if token in eos_ids or step + 1 == max_new_tokens:
            break
        output = target.forward(
            torch.tensor([[token]], device=input_ids.device), cache=output.cache,
        )
        parts.append(output.round_hidden[0].detach().cpu())
    tokens = torch.cat((input_ids[0].cpu(), torch.tensor(generated, dtype=torch.long)))
    hidden = torch.cat(parts).to(torch.bfloat16).contiguous()
    assert hidden.shape[0] == tokens.numel() - 1
    return tokens, hidden


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=128)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--limit", type=int, default=0)
    parser.add_argument("--per-split-limit", type=int, default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--target-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--memory-fraction", type=float, default=0.18)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    if args.max_new_tokens < 2 or args.max_prompt_tokens < 1 or min(args.limit, args.per_split_limit) < 0:
        raise ValueError("need at least two output tokens, a positive prompt budget, and nonnegative limit")
    torch.set_num_threads(args.num_threads)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
    from .target import OuroTarget
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    (root / "trajectories").mkdir(exist_ok=True)
    config = vars(args) | {"target_rounds": 4, "generation": "greedy", "feature_positions": "0..T-2"}
    config_path = root / "capture_config.json"
    if config_path.exists():
        old = json.loads(config_path.read_text())
        for key in ("model", "prompts", "max_new_tokens", "max_prompt_tokens", "attention_backend", "target_dtype", "target_rounds"):
            if old[key] != config[key]:
                raise ValueError(f"resume configuration mismatch: {key}")
    config_path.write_text(json.dumps(config, indent=2) + "\n")
    records = [r for path in args.prompts for r in read_jsonl(path)]
    if len({r["id"] for r in records}) != len(records):
        raise ValueError("prompt ids must be unique")
    selected = []
    counts = {}
    for index, record in enumerate(records):
        split = record["split"]
        counts[split] = counts.get(split, 0) + 1
        if not args.per_split_limit or counts[split] <= args.per_split_limit:
            selected.append((index, record))
    if args.limit:
        selected = selected[:args.limit]
    manifest = root / "manifest.jsonl"
    existing = read_jsonl(manifest) if manifest.exists() else []
    completed = {r["id"] for r in existing}
    target = OuroTarget.from_pretrained(
        args.model, device=args.device, dtype=getattr(torch, args.target_dtype),
        attention_backend=args.attention_backend,
    )
    io_path = root / "frozen_io.pt"
    if not io_path.exists():
        embed = target.model.get_input_embeddings().weight.detach().cpu()
        head = target.model.get_output_embeddings().weight.detach().cpu()
        torch.save({"embedding": embed, "lm_head": head, "model_path": args.model}, io_path)
    eos = target.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    started = time.monotonic()
    captured_tokens = 0
    skipped = 0
    for index, record in selected:
        if record["id"] in completed:
            continue
        ids = prompt_ids(target.tokenizer, record, args.max_prompt_tokens)
        if ids is None:
            skipped += 1
            print(json.dumps({"skipped_long_prompt": record["id"]}), flush=True)
            continue
        one_start = time.monotonic()
        tokens, hidden = capture_one(target, ids.to(args.device), args.max_new_tokens, eos_ids)
        if args.device.startswith("cuda"):
            torch.cuda.synchronize()
        filename = f"{index:06d}.pt"
        path = root / "trajectories" / filename
        row = {
            "id": record["id"], "source": record["source"], "split": record["split"],
            "prompt_length": ids.numel(), "generated_length": tokens.numel() - ids.numel(),
            "path": str(path.resolve()),
        }
        torch.save(row | {"token_ids": tokens, "round_hidden": hidden}, path)
        with manifest.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        completed.add(record["id"])
        captured_tokens += row["generated_length"]
        print(json.dumps({"captured": len(completed), "total_selected": len(selected), "id": row["id"],
                          "tokens": row["generated_length"], "seconds": time.monotonic() - one_start,
                          "elapsed_seconds": time.monotonic() - started}), flush=True)
    summary = {"records": len(completed), "skipped_long_prompts": skipped,
               "new_output_tokens": captured_tokens, "elapsed_seconds": time.monotonic() - started}
    (root / "capture_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
