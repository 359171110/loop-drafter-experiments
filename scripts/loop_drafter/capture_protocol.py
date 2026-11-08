"""Capture FP32 Ouro training trajectories in batches of equal prompt length."""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from contextlib import contextmanager
import fcntl
import json
import os
from pathlib import Path
import time

import torch

from .capture import prompt_ids


@torch.inference_mode()
def capture_batch(target, input_ids, max_new_tokens, eos_ids):
    """Return independent CPU (tokens, BF16 states) pairs for unpadded rows.

    Ended rows feed their EOS while other rows continue. Saved hidden states
    cover positions 0..T-2, excluding the last generated token in each row.
    """
    if input_ids.ndim != 2 or min(input_ids.shape) < 1 or max_new_tokens < 1:
        raise ValueError("capture_batch needs a nonempty [B,L] prompt and positive generation budget")
    eos_ids = set(eos_ids)
    output = target.forward(input_ids)
    batch, prompt_length = input_ids.shape
    if tuple(output.round_hidden.shape[:3]) != (batch, prompt_length, 4):
        raise ValueError("Target must return four actual round states for every prompt token")
    parts = [output.round_hidden.detach().to(device="cpu", dtype=torch.bfloat16).clone()]
    generated = [[] for _ in range(batch)]
    active = [True] * batch
    for step in range(max_new_tokens):
        predicted = output.logits[:, -1].argmax(-1).tolist()
        for row in range(batch):
            if active[row]:
                generated[row].append(predicted[row])
                if predicted[row] in eos_ids:
                    active[row] = False
        if not any(active) or step + 1 == max_new_tokens:
            break
        next_ids = [predicted[row] if active[row] else generated[row][-1] for row in range(batch)]
        output = target.forward(torch.tensor(next_ids, dtype=torch.long, device=input_ids.device)[:, None],
                                cache=output.cache)
        if tuple(output.round_hidden.shape[:3]) != (batch, 1, 4):
            raise ValueError("Target decode must return one state per row and round")
        parts.append(output.round_hidden.detach().to(device="cpu", dtype=torch.bfloat16).clone())
    all_hidden = torch.cat(parts, dim=1)
    result = []
    for row in range(batch):
        tokens = torch.cat((input_ids[row].detach().cpu(), torch.tensor(generated[row], dtype=torch.long))).clone()
        # clone() is essential: a view would serialize the whole batch storage.
        hidden = all_hidden[row, :tokens.numel() - 1].contiguous().clone()
        if hidden.shape[0] != tokens.numel() - 1:
            raise RuntimeError("Captured hidden states are not aligned to positions 0..T-2")
        result.append((tokens, hidden))
    return result


def _atomic_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _atomic_torch(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as stream:
        torch.save(value, stream)
        stream.flush()
        os.fsync(stream.fileno())
    temporary.replace(path)


def _append(path, row):
    with path.open("a", encoding="utf-8") as stream:
        stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        stream.flush()
        os.fsync(stream.fileno())


def _journal(path):
    """Recover only an interrupted final append; other corruption is an error."""
    if not path.exists():
        return []
    raw = path.read_bytes()
    pieces = raw.split(b"\n")
    lines = [piece + b"\n" for piece in pieces[:-1]]
    if pieces[-1]:
        lines.append(pieces[-1])
    rows, offset = [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except (ValueError, UnicodeDecodeError):
            if index == len(lines) - 1 and not line.endswith(b"\n"):
                with path.open("r+b") as stream:
                    stream.truncate(offset)
                break
            raise ValueError(f"Invalid journal entry in {path} at line {index + 1}")
        if not isinstance(row, dict):
            raise ValueError(f"Journal entry is not an object: {path}:{index + 1}")
        rows.append(row)
        offset += len(line)
    if rows and raw and not raw.endswith(b"\n") and offset == len(raw):
        # A complete JSON object without its final newline is already committed.
        with path.open("ab") as stream:
            stream.write(b"\n")
            stream.flush()
            os.fsync(stream.fileno())
    return rows


@contextmanager
def _lock(root):
    with (root / "capture.lock").open("a+") as stream:
        try:
            fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise RuntimeError(f"Another capture owns {root / 'capture.lock'}") from error
        try:
            yield
        finally:
            fcntl.flock(stream, fcntl.LOCK_UN)


def _inputs(args, root):
    snapshot = {}
    groups = []
    for kind, path in (("primary", args.prompts), ("reserve", args.reserves)):
        if path is None:
            snapshot[kind] = None
            groups.append([])
            continue
        path = Path(path).resolve()
        text = path.read_bytes().decode("utf-8")
        snapshot[kind] = {"path": str(path), "contents": text}
        # U+2028/U+2029 are legal inside JSON strings, not JSONL delimiters.
        records = [json.loads(line) for line in text.split("\n") if line.strip()]
        groups.append(records)
    snapshot_path = root / "prompt_snapshot.json"
    if snapshot_path.exists():
        if json.loads(snapshot_path.read_text()) != snapshot:
            raise ValueError("Prompt input contents or paths changed since the saved snapshot")
    else:
        _atomic_json(snapshot_path, snapshot)
    primary, reserves = groups
    if not primary:
        raise ValueError("Primary prompts must not be empty")
    seen = set()
    entries = []
    for index, record in enumerate(primary + reserves):
        for key in ("id", "source", "prompt", "split"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"Every prompt requires a nonempty string {key}")
        if record["split"] != "train":
            raise ValueError("capture_protocol accepts training prompts only; evaluation prompts are scored online")
        if record["id"] in seen:
            raise ValueError(f"Duplicate primary/reserve id: {record['id']}")
        seen.add(record["id"])
        entries.append({"index": index, "kind": "primary" if index < len(primary) else "reserve", "record": record})
    quotas = Counter(record["source"] for record in primary)
    if any(record["source"] not in quotas for record in reserves):
        raise ValueError("Reserve source does not exist among primary prompts")
    return entries, quotas


def _configuration(args, root):
    config = {
        "schema": "loop_drafter_capture_protocol_v1", "model": str(Path(args.model).resolve()),
        "batch_size": args.batch_size, "max_new_tokens": args.max_new_tokens,
        "max_prompt_tokens": args.max_prompt_tokens, "target_dtype": "float32",
        "feature_dtype": "bfloat16", "target_rounds": 4, "attention_backend": "sdpa",
        "generation": "greedy", "feature_positions": "0..T-2", "batching": "equal_prompt_length_no_padding",
    }
    path = root / "capture_config.json"
    if path.exists() and json.loads(path.read_text()) != config:
        raise ValueError("Capture configuration changed on resume")
    if not path.exists():
        _atomic_json(path, config)
    return config


def _metadata(entry, root, prompt_length=None, generated_length=None):
    record = entry["record"]
    row = {"id": record["id"], "source": record["source"], "domain": record.get("domain", record["source"]),
           "split": record["split"], "input_kind": entry["kind"], "input_index": entry["index"],
           "path": str((root / "trajectories" / f"{entry['index']:06d}.pt").resolve())}
    if prompt_length is not None:
        row["prompt_length"] = int(prompt_length)
    if generated_length is not None:
        row["generated_length"] = int(generated_length)
    return row


def _load_state(root, entries, quotas):
    manifest = _journal(root / "manifest.jsonl")
    rejected = _journal(root / "rejected.jsonl")
    by_id = {entry["record"]["id"]: entry for entry in entries}
    completed, failed = set(), set()
    counts = Counter()
    for rows, seen, success in ((manifest, completed, True), (rejected, failed, False)):
        for row in rows:
            identifier = row["id"]
            if identifier not in by_id or identifier in completed or identifier in failed:
                raise ValueError(f"Unknown or repeated id in capture journals: {identifier}")
            expected = _metadata(by_id[identifier], root)
            if any(row.get(key) != value for key, value in expected.items()):
                raise ValueError(f"Capture metadata changed for {identifier}")
            seen.add(identifier)
            if success:
                if not Path(row["path"]).is_file() or row.get("generated_length", 0) < 2:
                    raise ValueError(f"Completed trajectory is missing or invalid: {identifier}")
                counts[row["source"]] += 1
    # A crash after .pt rename but before journal append must not recapture.
    for entry in entries:
        identifier = entry["record"]["id"]
        if identifier in completed or identifier in failed:
            continue
        expected = _metadata(entry, root)
        path = Path(expected["path"])
        if not path.exists():
            continue
        record = torch.load(path, map_location="cpu", weights_only=True)
        if any(record.get(key) != value for key, value in expected.items()):
            raise ValueError(f"Orphan trajectory metadata mismatch: {path}")
        tokens, hidden = record["token_ids"], record["round_hidden"]
        prompt_length, generated_length = record["prompt_length"], record["generated_length"]
        if (generated_length < 2 or len(tokens) != prompt_length + generated_length
                or hidden.shape[:2] != (len(tokens) - 1, 4) or hidden.dtype != torch.bfloat16):
            raise ValueError(f"Orphan trajectory shape mismatch: {path}")
        row = expected | {"prompt_length": prompt_length, "generated_length": generated_length}
        _append(root / "manifest.jsonl", row)
        manifest.append(row)
        completed.add(identifier)
        counts[row["source"]] += 1
    if any(counts[source] > quota for source, quota in quotas.items()):
        raise ValueError("Completed trajectories exceed a source quota")
    return manifest, rejected, completed, failed, counts


def _default_target(args):
    from .target import OuroTarget
    return OuroTarget.from_pretrained(args.model, device=args.device, dtype=torch.float32, attention_backend="sdpa")


def run_capture(args, target_factory=None):
    """Run or resume capture; return a summary whose ``complete`` controls CLI exit."""
    if not 1 <= args.batch_size <= 4 or args.max_new_tokens < 2 or args.max_prompt_tokens < 1:
        raise ValueError("batch-size must be 1..4, output budget >=2, prompt budget positive")
    if not 0 < args.memory_fraction <= 1 or args.num_threads < 1:
        raise ValueError("memory fraction must be in (0,1] and num-threads positive")
    root = Path(args.out).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with _lock(root):
        (root / "trajectories").mkdir(exist_ok=True)
        entries, quotas = _inputs(args, root)
        _configuration(args, root)
        manifest, rejected, completed, failed, counts = _load_state(root, entries, quotas)
        started = time.monotonic()
        prior_progress = root / "progress.json"
        prior_elapsed = json.loads(prior_progress.read_text()).get("elapsed_seconds", 0) if prior_progress.exists() else 0
        new_tokens, batches = 0, 0

        def progress():
            row = {"done": len(manifest), "total": sum(quotas.values()),
                   "source_counts": {source: counts[source] for source in quotas},
                   "quotas": dict(quotas), "rejected": len(rejected), "new_tokens": new_tokens,
                   "batches_this_run": batches, "elapsed_seconds": prior_elapsed + time.monotonic() - started,
                   "session_elapsed_seconds": time.monotonic() - started}
            _atomic_json(root / "progress.json", row)
            print(json.dumps(row, ensure_ascii=False), flush=True)
            return row

        def reject(entry, reason, prompt_length=None, generated_length=0):
            row = _metadata(entry, root, prompt_length, generated_length) | {"reason": reason}
            _append(root / "rejected.jsonl", row)
            rejected.append(row)
            failed.add(row["id"])

        pending_candidates = [entry for entry in entries if entry["record"]["id"] not in completed | failed
                              and counts[entry["record"]["source"]] < quotas[entry["record"]["source"]]]
        if len(manifest) < sum(quotas.values()) and pending_candidates:
            torch.set_num_threads(args.num_threads)
            if args.device.startswith("cuda"):
                torch.cuda.set_per_process_memory_fraction(args.memory_fraction, args.device)
            target = (target_factory or _default_target)(args)
            first_parameter = next(target.model.parameters())
            if first_parameter.dtype != torch.float32:
                raise ValueError("Capture target must use FP32 parameters")
            io_path = root / "frozen_io.pt"
            if not io_path.exists():
                _atomic_torch(io_path, {
                    "embedding": target.model.get_input_embeddings().weight.detach().cpu().clone(),
                    "lm_head": target.model.get_output_embeddings().weight.detach().cpu().clone(),
                    "model_path": args.model,
                })
            eos = target.tokenizer.eos_token_id
            eos_ids = set(eos if isinstance(eos, list) else [] if eos is None else [eos])

            def execute(cohort):
                nonlocal new_tokens, batches
                buckets = defaultdict(list)
                for entry in cohort:
                    ids = prompt_ids(target.tokenizer, entry["record"], args.max_prompt_tokens)
                    if ids is None:
                        reject(entry, "prompt_too_long")
                    else:
                        buckets[ids.shape[1]].append((entry, ids))
                for length in sorted(buckets):
                    bucket = buckets[length]
                    for start in range(0, len(bucket), args.batch_size):
                        selected = bucket[start:start + args.batch_size]
                        ids = torch.cat([tokens for _, tokens in selected], dim=0).to(args.device)
                        captured = capture_batch(target, ids, args.max_new_tokens, eos_ids)
                        if len(captured) != len(selected):
                            raise RuntimeError("Target returned a different batch size")
                        for (entry, _), (tokens, hidden) in zip(selected, captured):
                            generated_length = tokens.numel() - length
                            new_tokens += generated_length
                            if generated_length < 2:
                                reject(entry, "fewer_than_two_generated_tokens", length, generated_length)
                                continue
                            row = _metadata(entry, root, length, generated_length)
                            if counts[row["source"]] >= quotas[row["source"]]:
                                raise RuntimeError("Scheduling attempted to exceed a source quota")
                            _atomic_torch(Path(row["path"]), row | {"token_ids": tokens, "round_hidden": hidden})
                            _append(root / "manifest.jsonl", row)
                            manifest.append(row)
                            completed.add(row["id"])
                            counts[row["source"]] += 1
                        batches += 1
                        progress()
                if not buckets:
                    progress()

            pending_primary = [entry for entry in entries if entry["kind"] == "primary"
                               and entry["record"]["id"] not in completed | failed]
            execute(pending_primary)
            while any(counts[source] < quota for source, quota in quotas.items()):
                selected_counts = Counter()
                cohort = []
                for entry in entries:
                    source, identifier = entry["record"]["source"], entry["record"]["id"]
                    if (entry["kind"] == "reserve" and identifier not in completed | failed
                            and selected_counts[source] < quotas[source] - counts[source]):
                        cohort.append(entry)
                        selected_counts[source] += 1
                if not cohort:
                    break
                execute(cohort)
        last_progress = progress()
        valid_prompt_tokens = sum(row["prompt_length"] for row in manifest)
        valid_generated_tokens = sum(row["generated_length"] for row in manifest)
        rejected_generated_tokens = sum(row.get("generated_length", 0) for row in rejected)
        rejected_prompt_tokens = sum(row.get("prompt_length", 0) for row in rejected)
        deficits = {source: quota - counts[source] for source, quota in quotas.items() if counts[source] < quota}
        summary = {
            "schema": "loop_drafter_capture_protocol_summary_v1", "complete": not deficits,
            "records": len(manifest), "requested_records": sum(quotas.values()), "quotas": dict(quotas),
            "source_counts": {source: counts[source] for source in quotas},
            "deficits": deficits, "rejections": len(rejected),
            "rejections_by_reason": dict(Counter(row["reason"] for row in rejected)),
            "valid_prompt_tokens": valid_prompt_tokens, "valid_generated_tokens": valid_generated_tokens,
            "valid_total_tokens": valid_prompt_tokens + valid_generated_tokens,
            "rejected_generated_tokens": rejected_generated_tokens,
            "rejected_prompt_tokens": rejected_prompt_tokens,
            "all_generated_tokens": valid_generated_tokens + rejected_generated_tokens,
            "all_processed_tokens": valid_prompt_tokens + valid_generated_tokens + rejected_prompt_tokens + rejected_generated_tokens,
            "new_output_tokens_this_run": new_tokens, "elapsed_seconds": last_progress["elapsed_seconds"],
            "target_dtype": "float32", "feature_dtype": "bfloat16", "target_rounds": 4,
            "batch_size": args.batch_size, "manifest": str(root / "manifest.jsonl"),
            "frozen_io": str(root / "frozen_io.pt"),
            "precision_note": "Equal-length batched FP32 execution; no bitwise equivalence to B1 is asserted.",
        }
        _atomic_json(root / "capture_summary.json", summary)
        return summary


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    result.add_argument("--model", required=True)
    result.add_argument("--prompts", required=True)
    result.add_argument("--reserves")
    result.add_argument("--out", required=True)
    result.add_argument("--batch-size", type=int, default=4)
    result.add_argument("--max-new-tokens", type=int, default=256)
    result.add_argument("--max-prompt-tokens", type=int, default=512)
    result.add_argument("--device", default="cuda:0")
    result.add_argument("--memory-fraction", type=float, default=.18)
    result.add_argument("--num-threads", type=int, default=8)
    return result


def main():
    summary = run_capture(parser().parse_args())
    print(json.dumps(summary, ensure_ascii=False), flush=True)
    if not summary["complete"]:
        raise SystemExit("Capture incomplete: insufficient same-source reserve prompts")


if __name__ == "__main__":
    main()
