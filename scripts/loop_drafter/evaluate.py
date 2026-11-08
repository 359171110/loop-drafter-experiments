"""Greedy short-chain speculative decoding against a fixed four-round target."""

import argparse
import json
import time
from pathlib import Path

import torch


def sync(device):
    if str(device).startswith("cuda"):
        torch.cuda.synchronize(device)


@torch.inference_mode()
def decode_ar(target, input_ids, max_new_tokens, eos_ids):
    sync(input_ids.device)
    started = time.perf_counter()
    output = target.forward(input_ids)
    sync(input_ids.device)
    prefill_seconds = time.perf_counter() - started
    decode_started = time.perf_counter()
    generated = []
    for step in range(max_new_tokens):
        token = int(output.logits[0, -1].argmax())
        generated.append(token)
        if token in eos_ids or step + 1 == max_new_tokens:
            break
        output = target.forward(torch.tensor([[token]], device=input_ids.device), cache=output.cache)
    sync(input_ids.device)
    return {"tokens": generated, "prefill_seconds": prefill_seconds,
            "decode_seconds": time.perf_counter() - decode_started,
            "target_calls": len(generated)}


@torch.inference_mode()
def decode_speculative(target, draft, input_ids, max_new_tokens, eos_ids, children=None):
    """Keep cache/taps only through accepted tokens; a bonus is not processed yet."""
    sync(input_ids.device)
    started = time.perf_counter()
    output = target.forward(input_ids)
    sync(input_ids.device)
    prefill_seconds = time.perf_counter() - started
    decode_started = time.perf_counter()
    cache = output.cache
    # Match the BF16 feature snapshots used for training, independently of the
    # reference target's arithmetic dtype. Quantize before taking loop deltas.
    taps = output.round_hidden[:, -1].to(torch.bfloat16)
    bonus = output.logits[:, -1].argmax(-1)
    generated = []
    rounds = []
    proposal_seconds = verify_seconds = 0.0
    depth = draft.config.depth if children is None else children
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        # A known EOS or final budget slot needs no additional model computation.
        if int(bonus[0]) in eos_ids or remaining == 1:
            generated.append(int(bonus[0]))
            break
        count = min(depth, remaining - 1)
        sync(input_ids.device)
        section = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=input_ids.device.type == "cuda"):
            proposals = draft.generate_draft(taps, bonus, max_children=count)
        sync(input_ids.device)
        proposal_seconds += time.perf_counter() - section
        block = torch.cat((bonus[:, None], proposals), dim=1)
        prefix_length = cache.get_seq_length()
        section = time.perf_counter()
        verified = target.forward(block, cache=cache)
        sync(input_ids.device)
        verify_seconds += time.perf_counter() - section
        predictions = verified.logits.argmax(-1)
        accepted = 0
        for index in range(count):
            if int(proposals[0, index]) != int(predictions[0, index]):
                break
            accepted += 1
        committed = block[0, :1 + accepted].tolist()
        for index, token in enumerate(committed):
            if token in eos_ids:
                committed = committed[:index + 1]
                break
        generated.extend(committed)
        kept = len(committed)
        cropped = target.crop_cache(verified.cache, prefix_length + kept)
        cache = verified.cache if cropped is None else cropped
        taps = verified.round_hidden[:, kept - 1].to(torch.bfloat16)
        bonus = predictions[:, kept - 1]
        rounds.append({"proposed_children": count, "accepted_children": kept - 1,
                       "committed_tokens": kept})
        if generated[-1] in eos_ids:
            break
    sync(input_ids.device)
    return {"tokens": generated, "prefill_seconds": prefill_seconds,
            "decode_seconds": time.perf_counter() - decode_started,
            "proposal_seconds": proposal_seconds, "verify_seconds": verify_seconds,
            "target_calls": 1 + len(rounds), "rounds": rounds}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
    parser.add_argument("--frozen-io", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--attention-backend", default="sdpa")
    parser.add_argument("--target-dtype", choices=("bfloat16", "float32"), default="float32")
    parser.add_argument("--max-new-tokens", type=int, default=96)
    parser.add_argument("--limit", type=int, default=32)
    parser.add_argument("--memory-fraction", type=float, default=0.18)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    if args.max_new_tokens < 1 or args.limit < 1:
        raise ValueError("generation budget and evaluation limit must be positive")
    torch.set_num_threads(args.num_threads)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
    from .capture import read_jsonl
    from .draft import load_checkpoint
    from .target import OuroTarget
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    entries = [r for r in read_jsonl(args.manifest) if r["split"] == "eval"][:args.limit]
    if not entries:
        raise ValueError("manifest has no held-out eval entries")
    target = OuroTarget.from_pretrained(args.model, device=args.device, dtype=getattr(torch, args.target_dtype),
                                        attention_backend=args.attention_backend)
    eos = target.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    prompts = []
    for entry in entries:
        record = torch.load(entry["path"], weights_only=False, map_location="cpu")
        ids = record["token_ids"][:record["prompt_length"]][None].to(args.device)
        prompts.append((entry, ids))
    # Warm up the same target before measuring. Shared-GPU measurements remain
    # diagnostic even with warmup; record that explicitly in the report.
    decode_ar(target, prompts[0][1], min(8, args.max_new_tokens), eos_ids)
    references = {}
    records = []
    for entry, ids in prompts:
        result = decode_ar(target, ids, args.max_new_tokens, eos_ids)
        references[entry["id"]] = result
        records.append({"method": "ar", "id": entry["id"], "source": entry["source"], **result})
        print(json.dumps({"method": "ar", "id": entry["id"], "tokens": len(result["tokens"])}), flush=True)
    summaries = []
    for checkpoint in args.checkpoints:
        draft, payload = load_checkpoint(checkpoint, frozen_io_path=args.frozen_io, device=args.device)
        draft.eval()
        decode_speculative(target, draft, prompts[0][1], min(8, args.max_new_tokens), eos_ids)
        run_records = []
        for entry, ids in prompts:
            result = decode_speculative(target, draft, ids, args.max_new_tokens, eos_ids)
            reference = references[entry["id"]]["tokens"]
            equal = reference == result["tokens"]
            mismatch = next((i for i, (a, b) in enumerate(zip(reference, result["tokens"])) if a != b), None)
            if mismatch is None and len(reference) != len(result["tokens"]):
                mismatch = min(len(reference), len(result["tokens"]))
            row = {"method": draft.config.feature_mode, "checkpoint": str(checkpoint),
                   "checkpoint_step": payload["cursor"]["step"],
                   "id": entry["id"], "source": entry["source"], "exact_match": equal,
                   "first_mismatch": mismatch, **result}
            records.append(row)
            run_records.append(row)
            print(json.dumps({"method": row["method"], "id": row["id"], "exact_match": equal,
                              "rounds": len(result["rounds"])}), flush=True)
        cycles = sum(len(r["rounds"]) for r in run_records)
        commits = sum(sum(c["committed_tokens"] for c in r["rounds"]) for r in run_records)
        accepted = sum(sum(c["accepted_children"] for c in r["rounds"]) for r in run_records)
        proposed = sum(sum(c["proposed_children"] for c in r["rounds"]) for r in run_records)
        ref_seconds = sum(r["decode_seconds"] for r in references.values())
        spec_seconds = sum(r["decode_seconds"] for r in run_records)
        summaries.append({"method": draft.config.feature_mode, "checkpoint": str(checkpoint),
                          "checkpoint_step": payload["cursor"]["step"],
                          "requests": len(run_records), "exact_matches": sum(r["exact_match"] for r in run_records),
                          "committed_tokens_per_verify": commits / max(cycles, 1),
                          "accepted_children_per_verify": accepted / max(cycles, 1),
                          "accepted_fraction": accepted / max(proposed, 1),
                          "diagnostic_decode_speedup": ref_seconds / max(spec_seconds, 1e-9),
                          "proposal_seconds": sum(r["proposal_seconds"] for r in run_records),
                          "verify_seconds": sum(r["verify_seconds"] for r in run_records)})
        del draft, payload
        if args.device.startswith("cuda"):
            torch.cuda.empty_cache()
    with (root / "requests.jsonl").open("w") as stream:
        for row in records:
            stream.write(json.dumps(row) + "\n")
    summary = {"config": vars(args), "target_rounds": 4, "decoding": "greedy",
               "drafter_feature_dtype": "bfloat16",
               "timing_scope": "diagnostic_only_shared_gpu", "methods": summaries,
               "all_exact_matches": all(r["exact_matches"] == r["requests"] for r in summaries)}
    (root / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary), flush=True)
    if not summary["all_exact_matches"]:
        raise RuntimeError("greedy output mismatch: inspect saved summary and request records")


if __name__ == "__main__":
    main()
