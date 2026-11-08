"""Prepare fixed independent capture shards, then merge without moving trajectories."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import json
import os
from pathlib import Path
from types import SimpleNamespace

import torch

from .capture_protocol import _atomic_json, _atomic_torch, _configuration, _inputs, _load_state, _lock


def _jsonl(path):
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def _write_jsonl(path, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _ensure_frozen_io(capture, model_path):
    """Read only embedding/head tensors on CPU, never instantiate the target."""
    destination = capture / "frozen_io.pt"
    if destination.is_file():
        return destination
    from safetensors import safe_open
    model_path = Path(model_path)
    names = {"model.embed_tokens.weight": "embedding", "lm_head.weight": "lm_head"}
    tensors = {}
    for path in sorted(model_path.glob("*.safetensors")):
        with safe_open(str(path), framework="pt", device="cpu") as checkpoint:
            for name, field in names.items():
                if field not in tensors and name in checkpoint.keys():
                    tensors[field] = checkpoint.get_tensor(name).float().clone()
        if len(tensors) == 2:
            break
    if "embedding" not in tensors:
        raise ValueError(f"No Ouro embedding tensor found in {model_path}")
    if "lm_head" not in tensors:
        config = json.loads((model_path / "config.json").read_text())
        if not config.get("tie_word_embeddings", False):
            raise ValueError("Untied Ouro checkpoint is missing its lm_head")
        tensors["lm_head"] = tensors["embedding"]
    _atomic_torch(destination, tensors | {"model_path": str(model_path)})
    return destination


def _records_from_snapshot(snapshot):
    primary = [json.loads(line) for line in snapshot["primary"]["contents"].split("\n") if line.strip()]
    reserve_snapshot = snapshot["reserve"]
    reserves = ([] if reserve_snapshot is None else
                [json.loads(line) for line in reserve_snapshot["contents"].split("\n") if line.strip()])
    return primary, reserves


def _check_plan(plan):
    """A resumed plan is immutable even as progress and worker journals advance."""
    if plan.get("schema") != "loop_drafter_parallel_capture_v1":
        raise ValueError("Unsupported parallel capture plan")
    for item in plan["input_snapshot"].values():
        if item is not None and Path(item["path"]).read_bytes().decode("utf-8") != item["contents"]:
            raise ValueError("Original prompt contents changed after shard preparation")
    primary, reserves = _records_from_snapshot(plan["input_snapshot"])
    records = {record["id"]: record for record in primary + reserves}
    for worker in plan["workers"]:
        if _jsonl(Path(worker["prompts"])) != [records[identifier] for identifier in worker["primary_ids"]]:
            raise ValueError(f"Worker {worker['index']} primary shard changed")
        if _jsonl(Path(worker["reserves"])) != [records[identifier] for identifier in worker["reserve_ids"]]:
            raise ValueError(f"Worker {worker['index']} reserve shard changed")
        if (Path(worker["out"]) / "frozen_io.pt").resolve() != Path(plan["frozen_io"]).resolve():
            raise ValueError(f"Worker {worker['index']} frozen I/O link changed")
    config_path = Path(plan["capture_root"]) / "capture_config.json"
    if json.loads(config_path.read_text()) != plan["capture_config"]:
        raise ValueError("Original capture configuration changed after preparation")
    return records


def prepare(root, workers=4, model=None, max_new_tokens=None, max_prompt_tokens=None):
    """Freeze the old results and assign all remaining source quotas once."""
    if workers < 1:
        raise ValueError("workers must be positive")
    root = Path(root).resolve()
    capture, parallel = root / "capture_fp32", root / "capture_parallel"
    capture.mkdir(parents=True, exist_ok=True)
    parallel.mkdir(parents=True, exist_ok=True)
    with _lock(parallel), _lock(capture):
        plan_path = parallel / "plan.json"
        if plan_path.exists():
            plan = json.loads(plan_path.read_text())
            _check_plan(plan)
            if plan["requested_workers"] != workers:
                raise ValueError("Cannot change worker count after the fixed plan exists")
            for requested, key in ((model, "model"), (max_new_tokens, "max_new_tokens"),
                                   (max_prompt_tokens, "max_prompt_tokens")):
                if requested is not None and (str(Path(requested).resolve()) if key == "model" else requested) != plan["capture_config"][key]:
                    raise ValueError(f"Cannot change {key} on resume")
            return plan
        config_path = capture / "capture_config.json"
        existing_config = json.loads(config_path.read_text()) if config_path.exists() else {}
        selected_model = str(Path(model).resolve()) if model is not None else existing_config.get("model")
        if selected_model is None:
            raise ValueError("--model is required when no prior capture configuration exists")
        args = SimpleNamespace(
            model=selected_model, batch_size=4,
            max_new_tokens=max_new_tokens if max_new_tokens is not None else existing_config.get("max_new_tokens", 256),
            max_prompt_tokens=max_prompt_tokens if max_prompt_tokens is not None else existing_config.get("max_prompt_tokens", 512),
            prompts=str(root / "prompts/train.jsonl"),
            reserves=str(root / "prompts/reserves.jsonl") if (root / "prompts/reserves.jsonl").exists() else None,
        )
        config = _configuration(args, capture)
        entries, quotas = _inputs(args, capture)
        old_rows, old_rejected, completed, failed, counts = _load_state(capture, entries, quotas)
        snapshot = json.loads((capture / "prompt_snapshot.json").read_text())
        records = {entry["record"]["id"]: entry["record"] for entry in entries}
        primary_by_source, reserve_by_source = defaultdict(list), defaultdict(list)
        for entry in entries:
            record = entry["record"]
            if record["id"] in completed | failed:
                continue
            (primary_by_source if entry["kind"] == "primary" else reserve_by_source)[record["source"]].append(record)
        assigned_primary = [[] for _ in range(workers)]
        assigned_reserves = [[] for _ in range(workers)]
        promoted = []
        for source, quota in quotas.items():
            deficit = quota - counts[source]
            candidates = primary_by_source[source]
            if len(candidates) < deficit:
                needed = deficit - len(candidates)
                if len(reserve_by_source[source]) < needed:
                    raise ValueError(f"Not enough unused {source} reserve prompts to replace historical rejections")
                replacements = reserve_by_source[source][:needed]
                promoted.extend(record["id"] for record in replacements)
                candidates = candidates + replacements
                reserve_by_source[source] = reserve_by_source[source][needed:]
            selected, extra_primary = candidates[:deficit], candidates[deficit:]
            for index, record in enumerate(selected):
                assigned_primary[index % workers].append(record)
            eligible = [index for index in range(workers)
                        if any(record["source"] == source for record in assigned_primary[index])]
            for index, record in enumerate(extra_primary + reserve_by_source[source]):
                if eligible:
                    assigned_reserves[eligible[index % len(eligible)]].append(record)
        frozen_io = _ensure_frozen_io(capture, selected_model)
        planned_workers = []
        for index in range(workers):
            if not assigned_primary[index]:
                continue
            worker_root = parallel / f"worker_{index}"
            out = worker_root / "capture"
            out.mkdir(parents=True, exist_ok=True)
            prompts, reserves = worker_root / "train.jsonl", worker_root / "reserves.jsonl"
            _write_jsonl(prompts, assigned_primary[index])
            _write_jsonl(reserves, assigned_reserves[index])
            link = out / "frozen_io.pt"
            if link.exists() or link.is_symlink():
                if link.resolve() != frozen_io.resolve():
                    raise ValueError(f"Unexpected pre-existing worker I/O file: {link}")
            else:
                link.symlink_to(frozen_io)
            planned_workers.append({
                "index": index, "prompts": str(prompts), "reserves": str(reserves), "out": str(out),
                "quotas": dict(Counter(record["source"] for record in assigned_primary[index])),
                "primary_ids": [record["id"] for record in assigned_primary[index]],
                "reserve_ids": [record["id"] for record in assigned_reserves[index]],
            })
        plan = {
            "schema": "loop_drafter_parallel_capture_v1", "root": str(root), "capture_root": str(capture),
            "requested_workers": workers, "workers": planned_workers, "capture_config": config,
            "quotas": dict(quotas), "existing_counts": {source: counts[source] for source in quotas},
            "existing_records": len(old_rows), "existing_manifest": old_rows, "existing_rejected": old_rejected,
            "input_snapshot": snapshot, "promoted_reserve_ids": promoted, "frozen_io": str(frozen_io),
            "canonical_order": [entry["record"]["id"] for entry in entries],
        }
        _atomic_json(plan_path, plan)
        return plan


def merge(root):
    """Require complete independent workers, then atomically publish canonical data."""
    root = Path(root).resolve()
    parallel, capture = root / "capture_parallel", root / "capture_fp32"
    with ExitStack() as stack:
        stack.enter_context(_lock(parallel))
        stack.enter_context(_lock(capture))
        plan = json.loads((parallel / "plan.json").read_text())
        records = _check_plan(plan)
        rows, rejected = list(plan["existing_manifest"]), list(plan["existing_rejected"])
        worker_elapsed = []
        for worker in plan["workers"]:
            out = Path(worker["out"])
            stack.enter_context(_lock(out))
            summary_path = out / "capture_summary.json"
            if not summary_path.exists():
                raise ValueError(f"Worker {worker['index']} has not completed capture")
            summary = json.loads(summary_path.read_text())
            if not summary.get("complete") or summary.get("source_counts") != worker["quotas"]:
                raise ValueError(f"Worker {worker['index']} did not fulfill its source quotas")
            if json.loads((out / "capture_config.json").read_text()) != plan["capture_config"]:
                raise ValueError(f"Worker {worker['index']} used a different target capture configuration")
            entries = [{"index": index, "kind": "primary" if index < len(worker["primary_ids"]) else "reserve",
                        "record": records[identifier]}
                       for index, identifier in enumerate(worker["primary_ids"] + worker["reserve_ids"])]
            worker_rows, worker_rejected, _, _, actual_counts = _load_state(out, entries, Counter(worker["quotas"]))
            if dict(actual_counts) != worker["quotas"]:
                raise ValueError(f"Worker {worker['index']} manifest is short of its source quotas")
            rows.extend(worker_rows)
            rejected.extend(worker_rejected)
            worker_elapsed.append(summary.get("elapsed_seconds", 0))
        by_id = {}
        for row in rows:
            identifier = row["id"]
            if identifier not in records or identifier in by_id:
                raise ValueError(f"Unknown or duplicate successful trajectory: {identifier}")
            original = records[identifier]
            if row["source"] != original["source"] or row["split"] != original["split"]:
                raise ValueError(f"Trajectory source or split changed: {identifier}")
            if not Path(row["path"]).is_file() or row["generated_length"] < 2:
                raise ValueError(f"Trajectory file missing or invalid: {identifier}")
            by_id[identifier] = row
        rejected_ids = set()
        for row in rejected:
            if row["id"] in by_id or row["id"] in rejected_ids:
                raise ValueError(f"Repeated or successful rejection id: {row['id']}")
            rejected_ids.add(row["id"])
        counts = Counter(row["source"] for row in rows)
        if dict(counts) != plan["quotas"]:
            raise ValueError(f"Merged source quotas mismatch: {dict(counts)} != {plan['quotas']}")
        original_rows = _jsonl(capture / "manifest.jsonl")
        current = {row["id"]: row for row in original_rows}
        if len(current) != len(original_rows):
            raise ValueError("Current original manifest contains duplicate IDs")
        for old in plan["existing_manifest"]:
            if current.get(old["id"]) != old:
                raise ValueError(f"An original trajectory changed or disappeared: {old['id']}")
        for identifier, row in current.items():
            if by_id.get(identifier) != row:
                raise ValueError("Original capture advanced or changed after shard preparation")
        order = {identifier: index for index, identifier in enumerate(plan["canonical_order"])}
        rows.sort(key=lambda row: order[row["id"]])
        rejected.sort(key=lambda row: order[row["id"]])
        prompt_tokens = sum(row["prompt_length"] for row in rows)
        generated_tokens = sum(row["generated_length"] for row in rows)
        rejected_tokens = sum(row.get("generated_length", 0) for row in rejected)
        rejected_prompts = sum(row.get("prompt_length", 0) for row in rejected)
        summary = {
            "schema": "loop_drafter_capture_protocol_summary_v1", "complete": True,
            "records": len(rows), "requested_records": sum(plan["quotas"].values()), "quotas": plan["quotas"],
            "source_counts": dict(counts), "deficits": {}, "rejections": len(rejected),
            "rejections_by_reason": dict(Counter(row["reason"] for row in rejected)),
            "valid_prompt_tokens": prompt_tokens, "valid_generated_tokens": generated_tokens,
            "valid_total_tokens": prompt_tokens + generated_tokens,
            "rejected_generated_tokens": rejected_tokens, "rejected_prompt_tokens": rejected_prompts,
            "all_generated_tokens": generated_tokens + rejected_tokens,
            "all_processed_tokens": prompt_tokens + generated_tokens + rejected_prompts + rejected_tokens,
            "target_dtype": "float32", "feature_dtype": "bfloat16", "target_rounds": 4, "batch_size": 4,
            "manifest": str(capture / "manifest.jsonl"), "frozen_io": plan["frozen_io"],
            "parallel_capture": {"plan": str(parallel / "plan.json"), "workers": len(plan["workers"]),
                                 "retained_records": plan["existing_records"],
                                 "worker_elapsed_seconds": worker_elapsed},
            "canonical_order": "original primary indices, followed by original reserve indices",
        }
        _write_jsonl(capture / "manifest.jsonl", rows)
        _write_jsonl(capture / "rejected.jsonl", rejected)
        _atomic_json(capture / "capture_summary.json", summary)
        _atomic_json(parallel / "merge.json", {"complete": True, "records": len(rows), "source_counts": dict(counts)})
        return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--phase", choices=("prepare", "merge"), required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--model")
    parser.add_argument("--max-new-tokens", type=int)
    parser.add_argument("--max-prompt-tokens", type=int)
    args = parser.parse_args()
    if args.phase == "prepare":
        plan = prepare(args.root, args.workers, args.model, args.max_new_tokens, args.max_prompt_tokens)
        print(json.dumps({"plan": str(Path(args.root).resolve() / "capture_parallel/plan.json"),
                          "workers": plan["workers"], "existing_counts": plan["existing_counts"], "quotas": plan["quotas"]}))
    else:
        print(json.dumps(merge(args.root)))


if __name__ == "__main__":
    main()
