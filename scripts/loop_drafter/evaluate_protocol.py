"""Interleaved, resumable speed evaluation on fixed per-dataset prompt panels."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import random

import torch

from .capture import prompt_ids, read_jsonl
from .capture_protocol import _journal as read_result_journal
from .draft import load_checkpoint
from .evaluate import decode_ar, decode_speculative


def atomic_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n")
    temporary.replace(path)


def aggregate(records):
    """Aggregate rates using elapsed time, with per-source paired AR baselines."""
    references = {(r["source"], r["id"]): r for r in records if r["method"] == "ar"}
    sources = sorted({r["source"] for r in records})
    result = []
    for source in [*sources, "__all__"]:
        selected = [r for r in records if source == "__all__" or r["source"] == source]
        for method in sorted({r["method"] for r in selected}):
            rows = [r for r in selected if r["method"] == method]
            tokens = sum(len(r["tokens"]) for r in rows)
            decode_tokens = sum(max(len(r["tokens"]) - 1, 0) for r in rows)
            prefill = sum(r["prefill_seconds"] for r in rows)
            decode = sum(r["decode_seconds"] for r in rows)
            cycles = [c for r in rows for c in r.get("rounds", [])]
            paired = [r for r in rows if (r["source"], r["id"]) in references]
            paired_ar_decode = sum(references[r["source"], r["id"]]["decode_seconds"] for r in paired)
            paired_ar_total = sum(references[r["source"], r["id"]]["decode_seconds"] + references[r["source"], r["id"]]["prefill_seconds"] for r in paired)
            paired_decode = sum(r["decode_seconds"] for r in paired)
            paired_total = sum(r["decode_seconds"] + r["prefill_seconds"] for r in paired)
            row = dict(source=source, method=method, requests=len(rows), generated_tokens=tokens,
                       decode_tokens=decode_tokens, prefill_seconds=prefill, decode_seconds=decode,
                       decode_tokens_per_second=decode_tokens / decode if decode else None,
                       end_to_end_tokens_per_second=tokens / (prefill + decode) if prefill + decode else None,
                       paired_requests=len(paired),
                       decode_latency_speedup=paired_ar_decode / paired_decode if paired_decode else None,
                       end_to_end_latency_speedup=paired_ar_total / paired_total if paired_total else None,
                       exact_matches=sum(r["tokens"] == references[r["source"], r["id"]]["tokens"] for r in paired),
                       verification_cycles=len(cycles),
                       mean_committed_tokens_per_verify=sum(c["committed_tokens"] for c in cycles) / len(cycles) if cycles else None,
                       mean_accepted_children_per_verify=sum(c["accepted_children"] for c in cycles) / len(cycles) if cycles else None)
            eligible = [c for c in cycles if c.get('proposed_children', 0) >= 1]
            row['first_child_acceptance_rate'] = (sum(c['accepted_children'] >= 1 for c in eligible) / len(eligible)
                                                   if eligible else None)
            result.append(row)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--checkpoints", nargs="+", required=True)
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
                           "batch_size": 1, "method_order": "cyclic rotation per shuffled prompt",
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

    from .target import OuroTarget
    target = OuroTarget.from_pretrained(args.model, device=args.device,
                                       dtype=torch.float32, attention_backend="sdpa")
    eos = target.tokenizer.eos_token_id
    eos_ids = set(eos if isinstance(eos, list) else ([] if eos is None else [eos]))
    drafts, checkpoints = {}, {}
    for path in args.checkpoints:
        draft, payload = load_checkpoint(path, frozen_io_path=args.frozen_io, device=args.device)
        mode = draft.config.feature_mode
        if mode in drafts:
            raise ValueError("provide one checkpoint per feature mode")
        drafts[mode] = draft.eval()
        stat = Path(path).stat()
        checkpoints[mode] = {"path": str(Path(path).resolve()), "cursor": payload["cursor"],
                             "config": draft.config.dict(), "file_size": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns}
        del payload
    if set(drafts) != {"final", "multiloop", "trajectory"}:
        raise ValueError("speed comparison requires all three drafter modes")
    methods = ["ar", "final", "multiloop", "trajectory"]
    prompt_sources = {p["id"]: p["source"] for p in prompts}
    if any(prompt_sources.get(r["id"]) != r["source"] or r["method"] not in methods for r in records):
        raise ValueError("existing evaluation rows do not belong to this panel")
    hardware_path = root / "hardware.json"
    previous_hardware = json.loads(hardware_path.read_text()) if hardware_path.exists() else None
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
    warmup = encoded[order[0]["id"]].to(args.device)
    decode_ar(target, warmup, min(8, args.max_new_tokens), eos_ids)
    for draft in drafts.values():
        decode_speculative(target, draft, warmup, min(8, args.max_new_tokens), eos_ids)
    session = {"started_at": datetime.now(timezone.utc).isoformat(),
               "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES")}
    hardware = {"cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "torch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                "timing_scope": "shared_gpu_interleaved_diagnostic", "checkpoints": checkpoints,
                "sessions": (previous_hardware.get("sessions", []) if previous_hardware else []) + [session]}
    atomic_json(hardware_path, hardware)
    with results_path.open("a") as stream:
        for index, entry in enumerate(order):
            ids = encoded[entry["id"]].to(args.device)
            offset = index % len(methods)
            for method in methods[offset:] + methods[:offset]:
                if (entry["id"], method) in completed:
                    continue
                if method == "ar":
                    result = decode_ar(target, ids, args.max_new_tokens, eos_ids)
                else:
                    result = decode_speculative(target, drafts[method], ids, args.max_new_tokens, eos_ids)
                row = {"id": entry["id"], "source": entry["source"], "domain": entry.get("domain"),
                       "method": method, "prompt_tokens": ids.numel(),
                       "finished_at": datetime.now(timezone.utc).isoformat(), **result}
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
                stream.flush()
                records.append(row)
                completed.add((entry["id"], method))
                progress = {"completed_method_requests": len(completed), "total_method_requests": len(prompts) * len(methods),
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
