"""Interleaved greedy comparison of SSD and accepted-history drafters."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random
import time

import torch

from .capture import prompt_ids, read_jsonl
from .capture_protocol import _journal as read_result_journal
from .draft import load_checkpoint
from .evaluate import decode_ar, decode_speculative, sync
from .evaluate_protocol import aggregate, atomic_json


METHODS = ("ar", "ssd", "history", "history_residual")


def load_history_checkpoint(*args, **kwargs):
    from .history_draft import load_history_checkpoint as loader
    return loader(*args, **kwargs)


@torch.inference_mode()
def decode_history_speculative(target, draft, input_ids, max_new_tokens, eos_ids, children=None):
    """Retain h4 only for processed, committed tokens; bonus is still unprocessed.

    Prefill contributes all prompt states, including the state predicting the
    first bonus. After verification, both KV and h4 stop at the same committed
    prefix. Rejected children and descendants never enter the history window.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("history evaluation requires one nonempty prompt")
    depth = draft.config.depth if children is None else children
    window = draft.config.window
    if max_new_tokens < 1 or window < 1 or not 0 <= depth <= draft.config.depth:
        raise ValueError("invalid generation budget, history window, or child count")
    sync(input_ids.device)
    started = time.perf_counter()
    output = target.forward(input_ids)
    sync(input_ids.device)
    prefill_seconds = time.perf_counter() - started
    decode_started = time.perf_counter()
    cache = output.cache
    # Quantize each retained state exactly as in the BF16 training snapshots.
    history = output.round_hidden[:, -window:, -1].to(torch.bfloat16)
    bonus = output.logits[:, -1].argmax(-1)
    generated, rounds = [], []
    proposal_seconds = verify_seconds = 0.0
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        if int(bonus[0]) in eos_ids or remaining == 1:
            generated.append(int(bonus[0]))
            break
        count = min(depth, remaining - 1)
        history_mask = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
        sync(input_ids.device)
        section = time.perf_counter()
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16,
                            enabled=input_ids.device.type == "cuda"):
            proposals = draft.generate_draft(history, history_mask, bonus, max_children=count)
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
        accepted_hidden = verified.round_hidden[:, :kept, -1].to(torch.bfloat16)
        history = torch.cat((history, accepted_hidden), dim=1)[:, -window:]
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
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--ssd-checkpoint", required=True)
    parser.add_argument("--history-checkpoint", required=True)
    parser.add_argument("--residual-checkpoint", required=True)
    parser.add_argument("--frozen-io", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-per-source", type=int, default=300)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, choices=[0], default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=.18)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args()
    if min(args.expected_per_source, args.max_new_tokens, args.max_prompt_tokens, args.num_threads) < 1:
        raise ValueError("evaluation counts must be positive")
    if not 0 < args.memory_fraction <= 1:
        raise ValueError("memory fraction must be in (0, 1]")
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    if args.device.startswith("cuda"):
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
    root = Path(args.out)
    root.mkdir(parents=True, exist_ok=True)
    prompts = read_jsonl(args.prompts)
    counts = Counter(p["source"] for p in prompts)
    if not prompts or any(n != args.expected_per_source for n in counts.values()):
        raise ValueError(f"unexpected evaluation counts: {dict(counts)}")
    if len({p["id"] for p in prompts}) != len(prompts) or any(p["split"] != "eval" for p in prompts):
        raise ValueError("evaluation prompts must have unique ids and split=eval")
    snapshot = root / "prompts.jsonl"
    if snapshot.exists() and read_jsonl(snapshot) != prompts:
        raise ValueError("evaluation prompt contents changed on resume")
    if not snapshot.exists():
        snapshot.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in prompts))
    config = vars(args) | {"target_dtype": "float32", "target_rounds": 4,
                           "feature_dtype": "bfloat16", "decoding": "greedy",
                           "batch_size": 1, "methods": list(METHODS),
                           "method_order": "cyclic rotation per shuffled prompt",
                           "checkpoint_selection": "fixed final training epoch; no evaluation-based selection"}
    config_path = root / "config.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("evaluation configuration changed on resume")
    atomic_json(config_path, config)
    results_path = root / "requests.jsonl"
    records = read_result_journal(results_path)
    completed = {(r["id"], r["method"]) for r in records}
    if len(completed) != len(records):
        raise ValueError("duplicate evaluation records")
    prompt_sources = {p["id"]: p["source"] for p in prompts}
    if any(prompt_sources.get(r["id"]) != r["source"] or r["method"] not in METHODS for r in records):
        raise ValueError("existing evaluation rows do not belong to this panel")

    hardware_path = root / "hardware.json"
    previous_hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else None
    visible_devices = os.environ.get("CUDA_VISIBLE_DEVICES")
    if records and previous_hardware and previous_hardware["cuda_visible_devices"] != visible_devices:
        raise ValueError("resume must retain the same GPU assignment for paired methods")
    from .target import OuroTarget
    target = OuroTarget.from_pretrained(args.model, device=args.device,
                                       dtype=torch.float32, attention_backend="sdpa")
    eos = target.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    paths = {"ssd": args.ssd_checkpoint, "history": args.history_checkpoint,
             "history_residual": args.residual_checkpoint}
    drafts, checkpoints = {}, {}
    for mode, path in paths.items():
        loader = load_checkpoint if mode == "ssd" else load_history_checkpoint
        draft, payload = loader(path, frozen_io_path=args.frozen_io, device=args.device)
        expected_mode = "final" if mode == "ssd" else mode
        if draft.config.feature_mode != expected_mode:
            raise ValueError(f"{mode} checkpoint has feature_mode={draft.config.feature_mode!r}")
        drafts[mode] = draft.eval()
        stat = Path(path).stat()
        checkpoints[mode] = {"path": str(Path(path).resolve()), "cursor": payload["cursor"],
                             "config": draft.config.dict(), "file_size": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns}
        del payload
    if previous_hardware and previous_hardware["checkpoints"] != checkpoints:
        raise ValueError("checkpoint identity changed since the previous evaluation records")
    order = list(prompts)
    random.Random(args.seed).shuffle(order)
    encoded = {}
    for entry in order:
        ids = prompt_ids(target.tokenizer, entry, args.max_prompt_tokens)
        if ids is None:
            raise ValueError(f"prepared evaluation prompt exceeds token budget: {entry['id']}")
        encoded[entry["id"]] = ids

    def run(method, ids, budget):
        if method == "ar":
            return decode_ar(target, ids, budget, eos_ids)
        decoder = decode_speculative if method == "ssd" else decode_history_speculative
        return decoder(target, drafts[method], ids, budget, eos_ids)

    warmup = encoded[order[0]["id"]].to(args.device)
    for method in METHODS:
        run(method, warmup, min(8, args.max_new_tokens))
    session = {"started_at": datetime.now(timezone.utc).isoformat(),
               "cuda_visible_devices": visible_devices, "device": args.device}
    hardware = {"cuda_visible_devices": visible_devices,
                "torch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                "timing_scope": "shared_gpu_interleaved_diagnostic", "checkpoints": checkpoints,
                "sessions": (previous_hardware.get("sessions", []) if previous_hardware else []) + [session]}
    atomic_json(hardware_path, hardware)
    with results_path.open("a") as stream:
        for index, entry in enumerate(order):
            ids = encoded[entry["id"]].to(args.device)
            offset = index % len(METHODS)
            for method in METHODS[offset:] + METHODS[:offset]:
                if (entry["id"], method) in completed:
                    continue
                result = run(method, ids, args.max_new_tokens)
                row = {"id": entry["id"], "source": entry["source"], "domain": entry.get("domain"),
                       "method": method, "prompt_tokens": ids.numel(),
                       "device": args.device, "cuda_visible_devices": visible_devices,
                       "finished_at": datetime.now(timezone.utc).isoformat(), **result}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(row)
                completed.add((entry["id"], method))
                progress = {"completed_method_requests": len(completed),
                            "total_method_requests": len(prompts) * len(METHODS),
                            "last_id": entry["id"], "last_method": method, "updated_at": row["finished_at"]}
                atomic_json(root / "progress.json", progress)
                print(json.dumps(progress), flush=True)
    summary = {"config": config, "hardware": hardware, "source_counts": dict(counts),
               "results": aggregate(records),
               "metric_notes": {"decode_tokens": "generated tokens minus first token supplied by prefill",
                                "speedup": "paired AR summed latency divided by drafter summed latency",
                                "output_comparison": "token identity diagnostic only; no answer correctness scoring",
                                "prefill": "separate from decode, included in end-to-end metrics"}}
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)


if __name__ == "__main__":
    main()
