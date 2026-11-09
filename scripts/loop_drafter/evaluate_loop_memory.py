"""Paired FP32 Ouro evaluation of the four frozen loop-memory ablations."""

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import random
import time

import torch

from .capture import prompt_ids, read_jsonl
from .capture_protocol import _journal as read_result_journal
from .evaluate import decode_ar, decode_speculative, sync
from .train import amp_context
from .evaluate_protocol import aggregate, atomic_json


METHODS = ("ar", "none", "final", "final_capacity", "trajectory")
LEGACY_PROTOCOL = "loop_memory_warmstart_ce_v1"
SCRATCH_PROTOCOL = "loop_memory_from_scratch_ce_v2"


def load_loop_memory_checkpoint(*args, **kwargs):
    from .loop_memory_draft import load_loop_memory_checkpoint as loader
    return loader(*args, **kwargs)


@torch.inference_mode()
def decode_loop_memory_speculative(target, draft, input_ids, max_new_tokens, eos_ids, children=None):
    """Append only committed round states to the incremental projected memory.

    Target KV may contain rejected descendants and is cropped. Memory never
    receives these descendants, so no stale suffix can survive verification.
    """
    if input_ids.ndim != 2 or input_ids.shape[0] != 1 or input_ids.shape[1] < 1:
        raise ValueError("loop memory evaluation requires one nonempty prompt")
    depth = draft.config.depth if children is None else children
    window = draft.config.window
    if max_new_tokens < 1 or window < 1 or not 0 <= depth <= draft.config.depth:
        raise ValueError("invalid generation budget, history window, or child count")
    if getattr(draft.config, "memory_mode", None) == "none":
        result = decode_speculative(target, draft, input_ids, max_new_tokens, eos_ids, children=depth)
        return result | {"memory_update_seconds": 0.0}
    sync(input_ids.device)
    started = time.perf_counter()
    output = target.forward(input_ids)
    history = output.round_hidden[:, -window:].to(torch.bfloat16)
    history_mask = torch.ones(history.shape[:2], dtype=torch.bool, device=history.device)
    with amp_context(input_ids.device):
        memory = draft.prepare_memory(history, history_mask)
    sync(input_ids.device)
    prefill_seconds = time.perf_counter() - started
    decode_started = time.perf_counter()
    cache = output.cache
    # Quantize each retained state exactly as in the BF16 training snapshots.
    taps = output.round_hidden[:, -1].to(torch.bfloat16)
    bonus = output.logits[:, -1].argmax(-1)
    generated, rounds = [], []
    proposal_seconds = verify_seconds = memory_update_seconds = 0.0
    while len(generated) < max_new_tokens:
        remaining = max_new_tokens - len(generated)
        if int(bonus[0]) in eos_ids or remaining == 1:
            generated.append(int(bonus[0]))
            break
        count = min(depth, remaining - 1)
        sync(input_ids.device)
        section = time.perf_counter()
        with amp_context(input_ids.device):
            proposals = draft.generate_draft(taps, bonus, memory=memory, max_children=count)
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
        sync(input_ids.device)
        section = time.perf_counter()
        accepted_hidden = verified.round_hidden[:, :kept].to(torch.bfloat16)
        with amp_context(input_ids.device):
            memory = draft.append_memory(memory, accepted_hidden)
        sync(input_ids.device)
        memory_update_seconds += time.perf_counter() - section
        taps = accepted_hidden[:, -1]
        bonus = predictions[:, kept - 1]
        rounds.append({"proposed_children": count, "accepted_children": kept - 1,
                       "committed_tokens": kept})
        if generated[-1] in eos_ids:
            break
    sync(input_ids.device)
    return {"tokens": generated, "prefill_seconds": prefill_seconds,
            "decode_seconds": time.perf_counter() - decode_started,
            "proposal_seconds": proposal_seconds, "verify_seconds": verify_seconds,
            "memory_update_seconds": memory_update_seconds,
            "target_calls": 1 + len(rounds), "rounds": rounds}


def encode_prompt(tokenizer, entry, maximum):
    """Honor completion whitespace and optionally consume frozen native IDs."""
    if "prompt_sha256" in entry:
        actual = hashlib.sha256(entry["prompt"].encode("utf-8")).hexdigest()
        if actual != entry["prompt_sha256"]:
            raise ValueError("frozen prompt hash changed")
    if "input_ids" in entry:
        identity = entry.get("tokenizer_path")
        if identity is None or Path(identity).resolve() != Path(tokenizer.name_or_path).resolve():
            raise ValueError("frozen input_ids require matching tokenizer_path")
        ids = entry["input_ids"]
        vocab_size = len(tokenizer)
        if not isinstance(ids, list) or not ids or any(type(i) is not int or not 0 <= i < vocab_size for i in ids):
            raise ValueError("frozen input_ids must be nonempty valid vocabulary IDs")
        if "input_ids_sha256" in entry:
            actual = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            if actual != entry["input_ids_sha256"]:
                raise ValueError("frozen input_ids hash changed")
        result = torch.tensor([ids], dtype=torch.long)
    elif entry.get("prompt_format", "native_chat") == "raw_completion":
        # HumanEval ends with meaningful newlines/indentation. Never strip it.
        result = torch.tensor([tokenizer.encode(entry["prompt"], add_special_tokens=False)], dtype=torch.long)
    elif entry.get("prompt_format", "native_chat") == "native_chat":
        result = prompt_ids(tokenizer, entry, maximum)
    else:
        raise ValueError("unknown prompt_format")
    if result is None or result.numel() < 1 or result.shape[1] > maximum:
        raise ValueError(f"evaluation prompt exceeds token budget or is empty: {entry['id']}")
    return result


def expected_counts(value):
    if value is None:
        return None
    text = Path(value).read_text() if not value.lstrip().startswith("{") else value
    counts = json.loads(text)
    if not isinstance(counts, dict) or not counts or any(type(v) is not int or v < 1 for v in counts.values()):
        raise ValueError("expected counts must map sources to positive integer counts")
    return counts


def verify_tokenizer_manifest(path, model):
    if path is None:
        return
    metadata = json.loads(Path(path).read_text())
    files = metadata["tokenizer"]["files"]
    if not files:
        raise ValueError("tokenizer manifest has no identity files")
    for entry in files:
        file = Path(entry["path"]).resolve()
        if file.parent != Path(model).resolve():
            raise ValueError("tokenizer manifest belongs to a different model directory")
        if file.stat().st_size != entry["size_bytes"]:
            raise ValueError(f"tokenizer file size changed: {file.name}")
        if hashlib.sha256(file.read_bytes()).hexdigest() != entry["sha256"]:
            raise ValueError(f"tokenizer file hash changed: {file.name}")


def checkpoint_training_identity(payload, config, frozen_io):
    """Validate final training state and return the common four-arm contract."""
    protocol = payload.get("protocol")
    if protocol not in (LEGACY_PROTOCOL, SCRATCH_PROTOCOL):
        raise ValueError("unsupported loop memory training protocol")
    scratch = protocol == SCRATCH_PROTOCOL
    if payload.get("from_scratch", False) is not scratch:
        raise ValueError("checkpoint protocol and from_scratch flag disagree")
    epochs, offset = (4, 0) if scratch else (2, 4)
    cursor, settings = payload.get("cursor", {}), payload.get("settings", {})
    if (settings.get("epochs") != epochs or cursor.get("epoch") != epochs
            or cursor.get("next_request") != 0 or payload.get("sampling_epoch_offset") != offset):
        raise ValueError(f"evaluation requires fixed final {epochs}-epoch checkpoints with sampling offset {offset}")
    if settings.get("role") != ("baseline" if scratch else "branch"):
        raise ValueError("checkpoint training role differs from its initialization protocol")
    parent = payload.get("parent")
    if scratch and parent is not None:
        raise ValueError("from-scratch checkpoint must not have an SSD parent")
    if not scratch and parent is None:
        raise ValueError("checkpoint is missing its SSD parent identity")
    entries = payload.get("data_index")
    if not isinstance(entries, list) or not entries or any(not isinstance(r, dict) or "id" not in r for r in entries):
        raise ValueError("checkpoint is missing its ordered training data identity")
    if len({r["id"] for r in entries}) != len(entries):
        raise ValueError("checkpoint training data contains duplicate request IDs")
    batch = settings.get("requests_per_update")
    if type(batch) is not int or batch < 1:
        raise ValueError("checkpoint is missing its global request batch budget")
    expected_steps = epochs * ((len(entries) + batch - 1) // batch)
    if cursor.get("step") != expected_steps or cursor.get("total_requests") != epochs * len(entries):
        raise ValueError("checkpoint final cursor does not complete its training data/update budget")
    saved_io = payload.get("frozen_io")
    if not isinstance(saved_io, str) or Path(saved_io).resolve() != Path(frozen_io).resolve():
        raise ValueError("evaluation frozen target IO differs from the training checkpoint")
    if payload.get("config") != config:
        raise ValueError("loaded model config differs from its training checkpoint")
    initialization = payload.get("initialization") if scratch else None
    if scratch:
        if not isinstance(initialization, dict) or initialization.get("kind") != "from_scratch":
            raise ValueError("scratch checkpoint is missing its shared initialization identity")
        fingerprint = initialization.get("initial_state_sha256", "")
        if (initialization.get("seed") != settings.get("seed") or type(settings.get("seed")) is not int
                or not isinstance(fingerprint, str) or len(fingerprint) != 64
                or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError("scratch initialization seed or state fingerprint is invalid")
    return {"protocol": protocol, "from_scratch": scratch, "parent": parent, "initialization": initialization,
            "settings": settings, "execution_plan": payload.get("execution_plan"), "sampling_epoch_offset": offset, "cursor": cursor,
            "config_except_memory_mode": {k: v for k, v in config.items() if k != "memory_mode"},
            "training_requests": len(entries),
            "data_index_sha256": hashlib.sha256(json.dumps(entries, sort_keys=True, separators=(",", ":")).encode()).hexdigest(),
            "frozen_io": str(Path(saved_io).resolve())}


def require_matching_training_identity(reference, actual):
    """Arm-specific model mode and run directory are the only expected differences."""
    if reference["protocol"] != actual["protocol"]:
        raise ValueError("cannot mix scratch and warmstart checkpoints in one evaluation")
    requirements = {
        "parent": "same SSD baseline parent",
        "initialization": "same scratch initialization identity",
        "settings": "same training settings and budget",
        "execution_plan": "same microbatch execution schedule",
        "config_except_memory_mode": "same drafter configuration except memory_mode",
        "training_requests": "same ordered training data",
        "data_index_sha256": "same ordered training data",
        "frozen_io": "same frozen target IO",
        "cursor": "same completed training budget",
    }
    for field, description in requirements.items():
        if reference[field] != actual[field]:
            raise ValueError("all loop memory arms must share the " + description)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--prompts", required=True)
    parser.add_argument("--checkpoints", nargs="+")
    for mode in METHODS[1:]:
        parser.add_argument("--" + mode.replace("_", "-") + "-checkpoint")
    parser.add_argument("--tokenizer-manifest", help="frozen sources.json with tokenizer.files identities")
    parser.add_argument("--frozen-io", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--expected-counts", help="source-count JSON object or path; otherwise use the frozen panel")
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--seed", type=int, choices=[0], default=0)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument("--memory-fraction", type=float, default=.18)
    parser.add_argument("--num-threads", type=int, default=8)
    args = parser.parse_args(argv)
    named = [getattr(args, mode + "_checkpoint") for mode in METHODS[1:]]
    if args.checkpoints and any(named):
        raise ValueError("use either --checkpoints or all four named checkpoint flags")
    paths = args.checkpoints or named
    if len(paths) != 4 or not all(paths):
        raise ValueError("provide exactly four loop memory checkpoints")
    verify_tokenizer_manifest(args.tokenizer_manifest, args.model)
    if min(args.max_new_tokens, args.max_prompt_tokens, args.num_threads) < 1:
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
    required_counts = expected_counts(args.expected_counts)
    if not prompts or (required_counts is not None and dict(counts) != required_counts):
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
    drafts, checkpoints = {}, {}
    training_identity = None
    for path in paths:
        draft, payload = load_loop_memory_checkpoint(path, frozen_io_path=args.frozen_io, device=args.device)
        mode = draft.config.memory_mode
        if mode not in METHODS[1:] or mode in drafts:
            raise ValueError("provide one checkpoint per loop memory mode")
        identity = checkpoint_training_identity(payload, draft.config.dict(), args.frozen_io)
        if training_identity is None:
            training_identity = identity
        else:
            require_matching_training_identity(training_identity, identity)
        drafts[mode] = draft.eval()
        stat = Path(path).stat()
        checkpoints[mode] = {"path": str(Path(path).resolve()), "cursor": payload["cursor"],
                             "config": draft.config.dict(), "file_size": stat.st_size,
                             "mtime_ns": stat.st_mtime_ns}
        del payload
    if set(drafts) != set(METHODS[1:]):
        raise ValueError("evaluation requires none, final, final_capacity and trajectory checkpoints")
    if previous_hardware and previous_hardware["checkpoints"] != checkpoints:
        raise ValueError("checkpoint identity changed since the previous evaluation records")
    if previous_hardware and previous_hardware.get("training_identity", training_identity) != training_identity:
        raise ValueError("training protocol identity changed since the previous evaluation records")
    order = list(prompts)
    random.Random(args.seed).shuffle(order)
    encoded = {}
    for entry in order:
        ids = encode_prompt(target.tokenizer, entry, args.max_prompt_tokens)
        encoded[entry["id"]] = ids

    def run(method, ids, budget):
        if method == "ar":
            return decode_ar(target, ids, budget, eos_ids)
        return decode_loop_memory_speculative(target, drafts[method], ids, budget, eos_ids)

    warmup = encoded[order[0]["id"]].to(args.device)
    for method in METHODS:
        run(method, warmup, min(8, args.max_new_tokens))
    session = {"started_at": datetime.now(timezone.utc).isoformat(),
               "cuda_visible_devices": visible_devices, "device": args.device}
    hardware = {"cuda_visible_devices": visible_devices,
                "torch_version": torch.__version__,
                "gpu_name": torch.cuda.get_device_name(args.device) if args.device.startswith("cuda") else None,
                "timing_scope": "shared_gpu_interleaved_diagnostic", "checkpoints": checkpoints,
                "training_identity": training_identity,
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
                                "memory_cost": "initial memory projection included in prefill; incremental accepted-token projection recorded in memory_update_seconds and decode; none uses original SSD decoder",
                                "speedup": "paired AR summed latency divided by drafter summed latency",
                                "output_comparison": "token identity diagnostic only; no answer correctness scoring",
                                "prefill": "separate from decode, included in end-to-end metrics"}}
    atomic_json(root / "summary.json", summary)
    print(json.dumps(summary), flush=True)
    return summary


if __name__ == "__main__":
    main()
