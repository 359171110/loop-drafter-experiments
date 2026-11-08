"""Train history drafters with the original four-epoch request/anchor budget.

The sampling epoch is zero-based and equals the checkpoint's next epoch plus
``sampling_epoch_offset``. At offset zero, request order and sampled anchors
are exactly those of ``train_protocol``. Resuming never resets this counter.
"""
import argparse
from dataclasses import asdict
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .draft import load_frozen_io
from .history_draft import HistoryConfig, HistoryDraft, load_history_checkpoint, make_history_batch
from .train import amp_context, validate_record
from .train_protocol import (Settings, distributed_context, epoch_order, index_training_data,
                             learning_rate, merge_statistics, prepare_requests,
                             request_label_counts, shard_indices, write_json)


PROTOCOL = "loop_history_complete_epoch_ce_v1"
SCHEMA = "loop_history_drafter_checkpoint_v1"


def load_training_index(source, config, expected_requests=0, index_file=None):
    """Reuse an already validated index after checking its manifest and provenance.

    A cached index must have its original sibling config.json. All ordered
    training rows and trajectory file metadata are checked without loading
    hidden tensors. Each record is also validated when consumed by training.
    """
    if index_file is None:
        return index_training_data(source, config, expected_requests)
    source, index_file = Path(source).resolve(), Path(index_file).resolve()
    if not source.is_file() or source.suffix == ".pt":
        raise ValueError("--index-file requires the original JSONL manifest as --data")
    provenance_path = index_file.parent / "config.json"
    if not provenance_path.is_file():
        raise ValueError("cached index requires its sibling config.json for shape provenance")
    provenance = json.loads(provenance_path.read_text())["config"]
    for key in ("target_hidden", "rounds", "vocab_size"):
        if provenance.get(key) != getattr(config, key):
            raise ValueError(f"cached index model configuration differs: {key}")
    entries = json.loads(index_file.read_text())
    if not isinstance(entries, list) or not entries:
        raise ValueError("cached data index must be a nonempty list")
    rows = []
    with source.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            if not isinstance(row, dict):
                raise ValueError("cached index requires manifest rows with explicit split and path")
            split = row.get("split")
            if split in ("eval", "dev", "validation", "test"):
                continue
            if split != "train":
                raise ValueError("cached index requires an explicit training split")
            rows.append(row)
    if len(rows) != len(entries) or (expected_requests and len(entries) != expected_requests):
        raise ValueError("cached index request count differs from manifest or expected count")
    ids, paths = set(), set()
    for row, entry in zip(rows, entries):
        path = Path(row["path"])
        path = (path if path.is_absolute() else source.parent / path).resolve()
        if str(path) != entry["path"]:
            raise ValueError("cached index path or order differs from manifest")
        if entry["id"] in ids or str(path) in paths:
            raise ValueError("duplicate request ID or path in cached index")
        ids.add(entry["id"])
        paths.add(str(path))
        for key in ("id", "source", "prompt_length"):
            if key in row and row[key] != entry[key]:
                raise ValueError(f"cached index manifest metadata differs: {key}")
        prompt, tokens = entry["prompt_length"], entry["token_count"]
        if (not isinstance(prompt, int) or not isinstance(tokens, int) or prompt < 1
                or tokens - prompt < 2 or entry["available_anchors"] != tokens - prompt - 1):
            raise ValueError("cached index contains inconsistent token/anchor counts")
        if "generated_length" in row and row["generated_length"] != tokens - prompt:
            raise ValueError("cached index generated length differs from manifest")
        stat = path.stat()
        if stat.st_size != entry["file_size"] or stat.st_mtime_ns != entry["mtime_ns"]:
            raise ValueError(f"cached trajectory changed since indexing: {path}")
    return entries


def accumulate_gradients(model, requests, microbatch, device, denominator=None):
    """Accumulate CE sums divided by one global count of valid child labels."""
    depth = model.config.depth
    per_request_labels = request_label_counts(requests, depth)
    local_labels = sum(per_request_labels)
    denominator = local_labels if denominator is None else denominator
    if denominator <= 0 or microbatch < 1:
        raise ValueError("an update needs valid labels and a positive microbatch")
    loss_sum = 0.
    labels_by_depth = torch.zeros(depth, dtype=torch.long)
    correct_by_depth = torch.zeros_like(labels_by_depth)
    for _, record, roots in requests:
        validate_record(record, model.config)
        for start in range(0, len(roots), microbatch):
            batch = make_history_batch(record, roots[start:start + microbatch], depth,
                                       model.config.window, device)
            with amp_context(device):
                logits = model(batch["history"], batch["history_mask"], batch["bonus"])
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch["labels"].flatten(),
                                   ignore_index=-100, reduction="sum")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite CE; optimizer update was not committed")
            (loss / denominator).backward()
            loss_sum += loss.detach().item()
            labels_by_depth += batch["valid"].sum(0).cpu()
            correct_by_depth += ((logits.detach().argmax(-1) == batch["labels"])
                                 & batch["valid"]).sum(0).cpu()
    return dict(loss=loss_sum / max(local_labels, 1), nll_sum=loss_sum, valid_labels=local_labels,
                anchors=sum(len(roots) for _, _, roots in requests),
                request_ids=[entry["id"] for entry, _, _ in requests],
                request_anchors=[len(roots) for _, _, roots in requests],
                request_labels=per_request_labels, labels_by_depth=labels_by_depth.tolist(),
                correct_by_depth=correct_by_depth.tolist(),
                top1_by_depth=(correct_by_depth / labels_by_depth.clamp_min(1)).tolist())


def synchronized_gradients(model, requests, microbatch, device):
    """SUM locally normalized gradients once, including empty tail ranks."""
    _, world_size = distributed_context()
    if world_size == 1:
        return accumulate_gradients(model, requests, microbatch, device)
    denominator = torch.tensor(sum(request_label_counts(requests, model.config.depth)),
                               dtype=torch.long, device=device)
    dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    local = accumulate_gradients(model, requests, microbatch, device, int(denominator))
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    parts = [None] * world_size
    dist.all_gather_object(parts, local)
    return merge_statistics(parts)


def save_training_checkpoint(path, model, optimizer, cursor, settings, entries, frozen_io,
                             sampling_epoch_offset):
    rank, world_size = distributed_context()
    device = next(model.parameters()).device
    rng = dict(torch_rng=torch.get_rng_state(), python_rng=random.getstate())
    if device.type == "cuda":
        rng["cuda_rng"] = torch.cuda.get_rng_state(device)
    rank_rng = [rng]
    if world_size > 1:
        rank_rng = [None] * world_size
        dist.all_gather_object(rank_rng, rng)
    if rank == 0:
        payload = dict(schema=SCHEMA, protocol=PROTOCOL, config=model.config.dict(),
                       model=model.state_dict(), optimizer=optimizer.state_dict(), cursor=dict(cursor),
                       settings=asdict(settings), data_index=entries,
                       sampling_epoch_offset=sampling_epoch_offset,
                       frozen_io=str(Path(frozen_io).resolve()),
                       split=dict(train=[entry["path"] for entry in entries], validation=[]),
                       distributed=dict(world_size=world_size, gradient_reduction="sum_global_label_normalized"),
                       rank_rng=rank_rng, **rng)
        path = Path(path)
        temporary = path.with_suffix(path.suffix + ".tmp")
        torch.save(payload, temporary)
        temporary.replace(path)
    if world_size > 1:
        dist.barrier()


def restore_training_state(payload, optimizer, settings, entries, device, sampling_epoch_offset):
    if (payload.get("schema") != SCHEMA or payload.get("protocol") != PROTOCOL
            or payload.get("settings") != asdict(settings)):
        raise ValueError("resume protocol/settings differ from saved training")
    if payload.get("sampling_epoch_offset") != sampling_epoch_offset:
        raise ValueError("resume sampling epoch offset differs from saved training")
    if payload.get("data_index") != entries:
        raise ValueError("training data identity, order, or file metadata changed since checkpoint")
    cursor = dict(payload["cursor"])
    count = len(entries)
    per_epoch = math.ceil(count / settings.requests_per_update)
    epoch, offset, step = cursor["epoch"], cursor["next_request"], cursor["step"]
    if not 0 <= epoch <= settings.epochs or not 0 <= offset < count:
        raise ValueError("checkpoint cursor is outside the epoch budget")
    if offset % settings.requests_per_update or (epoch == settings.epochs and offset):
        raise ValueError("checkpoint cursor does not end at an optimizer-update boundary")
    if step != epoch * per_epoch + offset // settings.requests_per_update:
        raise ValueError("checkpoint global step and epoch cursor disagree")
    if cursor["total_requests"] != epoch * count + offset:
        raise ValueError("checkpoint total requests and epoch cursor disagree")
    optimizer.load_state_dict(payload["optimizer"])
    rank, _ = distributed_context()
    saved_rng = payload.get("rank_rng", [payload])
    rng = saved_rng[rank] if rank < len(saved_rng) else saved_rng[0]
    torch.set_rng_state(rng["torch_rng"])
    random.setstate(rng["python_rng"])
    if device.type == "cuda" and "cuda_rng" in rng:
        torch.cuda.set_rng_state(rng["cuda_rng"], device)
    return cursor


def reconcile_metrics(path, committed_step):
    """Recover complete checkpointed JSONL rows, preserving Unicode separators."""
    path = Path(path)
    if not path.exists():
        return 0
    with path.open(encoding="utf-8") as handle:
        lines = list(handle)
    kept, discarded = [], 0
    for index, line in enumerate(lines):
        try:
            row = json.loads(line)
        except json.JSONDecodeError:
            if index != len(lines) - 1:
                raise ValueError("metrics contain an invalid line before the final line")
            discarded += 1
            continue
        if row.get("event") == "update" and row["step"] <= committed_step:
            kept.append(row)
        else:
            discarded += 1
    if [row["step"] for row in kept] != list(range(1, committed_step + 1)):
        raise ValueError("checkpointed training metrics have missing or duplicate updates")
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    temporary.replace(path)
    return discarded


def train_epochs(model, entries, settings, out, frozen_io, *, resume_payload=None,
                 sampling_epoch_offset=0, save_every=50, stop_after_steps=0):
    """Train a fresh history model or resume its next committed update."""
    if not entries or save_every < 1 or stop_after_steps < 0 or sampling_epoch_offset < 0:
        raise ValueError("invalid data, save interval, stop step or sampling offset")
    if settings.role != "baseline" or model.config.feature_mode not in ("history", "history_residual"):
        raise ValueError("history models train from scratch with baseline settings")
    rank, world_size = distributed_context()
    primary = rank == 0
    if world_size not in (1, 2, 4) or world_size > settings.requests_per_update:
        raise ValueError("world size must be 1, 2 or 4 and <= global requests per update")
    out = Path(out)
    if primary:
        out.mkdir(parents=True, exist_ok=True)
    device = next(model.parameters()).device
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.peak_lr,
                                  betas=(settings.beta1, settings.beta2), weight_decay=settings.weight_decay)
    total_steps = settings.epochs * math.ceil(len(entries) / settings.requests_per_update)
    cursor = dict(step=0, epoch=0, next_request=0, total_requests=0, total_anchors=0, total_labels=0)
    metrics = out / "metrics.jsonl"
    if resume_payload is not None:
        if resume_payload["config"] != model.config.dict():
            raise ValueError("resume model config changed")
        if Path(resume_payload["frozen_io"]).resolve() != Path(frozen_io).resolve():
            raise ValueError("resume frozen IO path differs from checkpoint")
        model.load_state_dict(resume_payload["model"], strict=True)
        cursor = restore_training_state(resume_payload, optimizer, settings, entries, device,
                                        sampling_epoch_offset)
        if primary:
            dropped = reconcile_metrics(metrics, cursor["step"])
            print(json.dumps(dict(event="resume", **cursor, discarded_uncommitted_metric_rows=dropped)), flush=True)
    else:
        if primary and ((out / "latest.pt").exists() or metrics.exists()):
            raise FileExistsError("output already contains training state; use --resume or a new directory")
        if primary:
            write_json(out / "config.json", dict(
                protocol=PROTOCOL, config=model.config.dict(), settings=asdict(settings),
                sampling_epoch_offset=sampling_epoch_offset, total_steps=total_steps, requests=len(entries),
                distributed=dict(world_size=world_size, gradient_reduction="sum_global_label_normalized"),
                loss="uniform_ce_over_all_valid_child_labels_in_update",
                frozen_io=str(Path(frozen_io).resolve()),
                trainable_parameters=sum(p.numel() for p in model.parameters() if p.requires_grad)))
            write_json(out / "data_index.json", entries)
    if world_size > 1:
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)

    def save(name):
        save_training_checkpoint(out / name, model, optimizer, cursor, settings, entries,
                                 frozen_io, sampling_epoch_offset)

    if resume_payload is None:
        save("initial.pt")
        save("latest.pt")
    model.train()
    start_time = time.monotonic()
    if primary:
        print(json.dumps(dict(event="training_budget", feature_mode=model.config.feature_mode,
                              requests=len(entries), epochs=settings.epochs, total_steps=total_steps,
                              world_size=world_size, sampling_epoch_offset=sampling_epoch_offset,
                              remaining_steps=total_steps - cursor["step"])), flush=True)
    while cursor["epoch"] < settings.epochs:
        if stop_after_steps and cursor["step"] >= stop_after_steps:
            break
        epoch = cursor["epoch"]
        sampling_epoch = epoch + sampling_epoch_offset
        order = epoch_order(len(entries), settings.seed, sampling_epoch)
        start = cursor["next_request"]
        indices = order[start:start + settings.requests_per_update]
        requests = prepare_requests(entries, shard_indices(indices, rank, world_size), settings, sampling_epoch)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(cursor["step"] + 1, total_steps, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        tick = time.monotonic()
        stats = synchronized_gradients(model, requests, settings.microbatch, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        end = start + len(indices)
        finished_epoch = end == len(entries)
        cursor = dict(step=cursor["step"] + 1, epoch=epoch + int(finished_epoch),
                      next_request=0 if finished_epoch else end,
                      total_requests=cursor["total_requests"] + len(indices),
                      total_anchors=cursor["total_anchors"] + stats["anchors"],
                      total_labels=cursor["total_labels"] + stats["valid_labels"])
        row = dict(event="update", step=cursor["step"], epoch=epoch + 1, sampling_epoch=sampling_epoch,
                   requests=len(indices), request_offset=start, lr=lr, grad_norm=float(grad_norm),
                   world_size=world_size, requests_by_rank=[len(shard_indices(indices, i, world_size))
                                                          for i in range(world_size)],
                   update_seconds=time.monotonic() - tick, **stats)
        if primary:
            with metrics.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if finished_epoch:
            save(f"epoch{epoch + 1}.pt")
        if finished_epoch or cursor["step"] % save_every == 0:
            save("latest.pt")
        del requests
    save("latest.pt")
    complete = cursor["epoch"] == settings.epochs
    if complete:
        save("model.pt")
    result = dict(event="complete" if complete else "paused", **cursor, total_steps=total_steps,
                  sampling_epoch_offset=sampling_epoch_offset, world_size=world_size,
                  invocation_seconds=time.monotonic() - start_time,
                  checkpoint=str(out / ("model.pt" if complete else "latest.pt")))
    if primary:
        write_json(out / "status.json", result)
        print(json.dumps(result), flush=True)
    if world_size > 1:
        dist.barrier()
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ("data", "frozen-io", "out"):
        p.add_argument("--" + name, required=True)
    p.add_argument("--feature-mode", choices=("history", "history_residual"), required=True)
    p.add_argument("--index-file", help="validated baseline data_index.json with sibling config.json")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--sampling-epoch-offset", type=int, default=0)
    p.add_argument("--requests-per-update", type=int, default=8)
    p.add_argument("--max-anchors-per-request", type=int, default=64)
    p.add_argument("--batch-size", "--microbatch", dest="microbatch", type=int, default=16)
    p.add_argument("--expected-requests", type=int, default=8192)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--stop-after-steps", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--memory-fraction", type=float, default=.18)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--width", type=int, default=512)
    p.add_argument("--mlp-width", type=int, default=1536)
    p.add_argument("--layers", type=int, default=2)
    p.add_argument("--heads", type=int, default=8)
    p.add_argument("--window", type=int, default=64)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.memory_fraction <= 1 or args.num_threads < 1 or args.expected_requests < 0:
        raise ValueError("invalid memory fraction, thread count or expected request count")
    if args.save_every < 1 or args.stop_after_steps < 0 or args.sampling_epoch_offset < 0:
        raise ValueError("invalid save interval, stop step or sampling offset")
    settings = Settings(epochs=args.epochs, requests_per_update=args.requests_per_update,
                        max_anchors_per_request=args.max_anchors_per_request,
                        microbatch=args.microbatch, seed=args.seed)
    torch.set_num_threads(args.num_threads)
    torch.manual_seed(args.seed)
    random.seed(args.seed)
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size not in (1, 2, 4) or world_size > settings.requests_per_update:
        raise ValueError("torchrun world size must be 1, 2 or 4 and <= global requests per update")
    device = torch.device(args.device)
    if world_size > 1 and device.type == "cuda":
        local_rank = int(os.environ.get("LOCAL_RANK", "0")) if args.local_rank is None else args.local_rank
        device = torch.device("cuda", local_rank)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
        torch.cuda.manual_seed_all(args.seed)
    created_group = world_size > 1 and not dist.is_initialized()
    if created_group:
        dist.init_process_group(backend="nccl" if device.type == "cuda" else "gloo")
    try:
        return _run(args, settings, device)
    finally:
        if created_group:
            dist.destroy_process_group()


def _run(args, settings, device):
    embedding, head = load_frozen_io(args.frozen_io)
    config = HistoryConfig(target_hidden=embedding.shape[1], vocab_size=embedding.shape[0],
                           feature_mode=args.feature_mode, width=args.width, mlp_width=args.mlp_width,
                           layers=args.layers, heads=args.heads, window=args.window, depth=args.depth)
    if args.resume:
        del embedding, head
        model, payload = load_history_checkpoint(Path(args.out) / "latest.pt", args.frozen_io, device)
        if model.config.dict() != config.dict():
            raise ValueError("resume requested model configuration differs from checkpoint")
    else:
        payload = None
        model = HistoryDraft(config, embedding, head).to(device)
        del embedding, head
    rank, world_size = distributed_context()
    entries = None
    if rank == 0:
        entries = load_training_index(args.data, config, args.expected_requests, args.index_file)
    if world_size > 1:
        message = [entries]
        dist.broadcast_object_list(message, src=0)
        entries = message[0]
    return train_epochs(model, entries, settings, args.out, args.frozen_io, resume_payload=payload,
                        sampling_epoch_offset=args.sampling_epoch_offset,
                        save_every=args.save_every, stop_after_steps=args.stop_after_steps)


if __name__ == "__main__":
    main()
