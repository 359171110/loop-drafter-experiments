"""Complete-epoch CE training for the canonical Ouro loop-feature comparison.

Each update uses distinct requests, with at most 64 unique anchors per request.
All microbatch CE sums share one denominator: the update's total valid child
labels. There is no held-out checkpoint selection; epoch checkpoints and the
final checkpoint retain the same model schema as ``draft.load_checkpoint``.
"""
import argparse
from dataclasses import asdict, dataclass
import json
import math
import os
from pathlib import Path
import random
import time

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .draft import Config, LoopDraft, load_checkpoint, load_frozen_io
from .train import (amp_context, anchor_positions, load_record, make_batch,
                    manifest_splits, record_paths, validate_record)


PROTOCOL = "loop_drafter_complete_epoch_ce_v1"


def distributed_context():
    return (dist.get_rank(), dist.get_world_size()) if dist.is_initialized() else (0, 1)


def shard_indices(indices, rank, world_size):
    """Balanced contiguous partitions preserve global request order on gather."""
    quotient, remainder = divmod(len(indices), world_size)
    start = rank * quotient + min(rank, remainder)
    return indices[start:start + quotient + int(rank < remainder)]


def request_label_counts(requests, depth):
    return [sum(min(depth, len(record["token_ids"]) - s - 2) for s in roots)
            for _, record, roots in requests]


def merge_statistics(parts):
    """Combine exact label/correct counts and summed NLL, never per-rank means."""
    total_labels = sum(part["valid_labels"] for part in parts)
    nll = sum(part["nll_sum"] for part in parts)
    depth = len(parts[0]["labels_by_depth"])
    labels = [sum(part["labels_by_depth"][i] for part in parts) for i in range(depth)]
    correct = [sum(part["correct_by_depth"][i] for part in parts) for i in range(depth)]
    return dict(loss=nll / total_labels, nll_sum=nll, valid_labels=total_labels,
                anchors=sum(part["anchors"] for part in parts),
                request_ids=[item for part in parts for item in part["request_ids"]],
                request_anchors=[item for part in parts for item in part["request_anchors"]],
                request_labels=[item for part in parts for item in part["request_labels"]],
                labels_by_depth=labels, correct_by_depth=correct,
                top1_by_depth=[n / max(d, 1) for n, d in zip(correct, labels)])


def synchronized_gradients(model, requests, microbatch, device):
    """Equivalent to DDP, with one gradient SUM after all local microbatches.

    Automatic DDP backward collectives would deadlock when ranks execute
    different numbers of microbatches. Every local CE sum instead uses the
    GLOBAL label denominator; an explicit gradient SUM then gives the global
    mean gradient. There is intentionally no further division by world size.
    Empty ranks in an epoch's final update contribute zeros to every collective.
    """
    _, world_size = distributed_context()
    if world_size == 1:
        return accumulate_gradients(model, requests, microbatch, device)
    denominator = torch.tensor(sum(request_label_counts(requests, model.config.depth)),
                               dtype=torch.long, device=device)
    dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    local = accumulate_gradients(model, requests, microbatch, device, denominator=int(denominator))
    for parameter in model.parameters():
        if parameter.requires_grad:
            if parameter.grad is None:
                parameter.grad = torch.zeros_like(parameter)
            dist.all_reduce(parameter.grad, op=dist.ReduceOp.SUM)
    parts = [None] * world_size
    dist.all_gather_object(parts, local)
    return merge_statistics(parts)


@dataclass(frozen=True)
class Settings:
    role: str = "baseline"
    epochs: int = 4
    requests_per_update: int = 8
    max_anchors_per_request: int = 64
    microbatch: int = 16
    seed: int = 0
    peak_lr: float = 1e-4
    floor_lr: float = 1e-5
    warmup_fraction: float = .05
    weight_decay: float = .01
    beta1: float = .9
    beta2: float = .95
    grad_clip: float = 1.

    def __post_init__(self):
        if self.role not in ("baseline", "branch"):
            raise ValueError("role must be baseline or branch")
        if min(self.epochs, self.requests_per_update, self.max_anchors_per_request,
               self.microbatch) < 1:
            raise ValueError("epoch, request, anchor and microbatch counts must be positive")
        if not 0 <= self.warmup_fraction < 1 or not 0 < self.floor_lr <= self.peak_lr:
            raise ValueError("invalid warmup fraction or learning-rate bounds")
        if not 0 <= self.beta1 < 1 or not 0 <= self.beta2 < 1:
            raise ValueError("AdamW betas must lie in [0, 1)")
        if self.weight_decay < 0 or self.grad_clip <= 0:
            raise ValueError("weight decay must be nonnegative and clipping positive")


def learning_rate(update, total_updates, settings):
    """One-based update schedule; the final update uses exactly floor_lr."""
    if not 1 <= update <= total_updates:
        raise ValueError("update must lie in the planned training budget")
    warmup = min(total_updates - 1, math.ceil(total_updates * settings.warmup_fraction))
    if warmup and update <= warmup:
        return settings.peak_lr * update / warmup
    progress = (update - warmup) / (total_updates - warmup)
    return settings.floor_lr + .5 * (settings.peak_lr - settings.floor_lr) * (1 + math.cos(math.pi * progress))


def epoch_order(count, seed, epoch):
    order = list(range(count))
    random.Random(seed + epoch).shuffle(order)
    return order


def index_training_data(source, config, expected_requests=0):
    """Validate effective training requests; explicit nontraining splits are excluded."""
    declared = manifest_splits(source)
    entries, ids = [], set()
    for path in record_paths(source):
        record = load_record(path)
        manifest_split, record_split = declared.get(str(path)), record.get("split")
        if manifest_split is not None and record_split is not None and manifest_split != record_split:
            raise ValueError(f"manifest and trajectory split disagree: {path}")
        split = manifest_split if manifest_split is not None else record_split
        if split in ("eval", "dev", "validation", "test"):
            continue
        if split not in (None, "train"):
            raise ValueError(f"unknown split {split!r}: {path}")
        validate_record(record, config)
        count = len(anchor_positions(record))
        if not count:
            raise ValueError(f"training request has no valid child anchors: {path}")
        request_id = str(record.get("id", path.stem))
        if request_id in ids:
            raise ValueError(f"duplicate training request ID {request_id!r}")
        ids.add(request_id)
        stat = path.stat()
        entries.append(dict(path=str(path), id=request_id, source=str(record.get("source", "unknown")),
                            prompt_length=int(record["prompt_length"]),
                            token_count=len(record["token_ids"]), available_anchors=count,
                            file_size=stat.st_size, mtime_ns=stat.st_mtime_ns))
    if not entries:
        raise ValueError("no effective training requests")
    if expected_requests and len(entries) != expected_requests:
        raise ValueError(f"expected {expected_requests} effective requests, found {len(entries)}")
    return entries


def prepare_requests(entries, indices, settings, epoch):
    prepared = []
    for index in indices:
        entry = entries[index]
        record = load_record(entry["path"])
        roots = anchor_positions(record, settings.max_anchors_per_request,
                                 settings.seed + epoch * len(entries) + index)
        prepared.append((entry, record, roots))
    return prepared


def accumulate_gradients(model, requests, microbatch, device, denominator=None):
    """Backpropagate total CE / total labels, leaving clipping/stepping to caller."""
    depth = model.config.depth
    per_request_labels = request_label_counts(requests, depth)
    local_labels = sum(per_request_labels)
    denominator = local_labels if denominator is None else denominator
    if denominator <= 0:
        raise ValueError("an optimizer update must contain valid child labels")
    loss_sum = 0.
    labels_by_depth = torch.zeros(depth, dtype=torch.long)
    correct_by_depth = torch.zeros_like(labels_by_depth)
    for _, record, roots in requests:
        for start in range(0, len(roots), microbatch):
            batch = make_batch(record, roots[start:start + microbatch], depth, device)
            with amp_context(device):
                logits = model(batch["taps"], batch["bonus"], batch["previous"])
            loss = F.cross_entropy(logits.float().flatten(0, 1), batch["labels"].flatten(),
                                   ignore_index=-100, reduction="sum")
            if not bool(torch.isfinite(loss)):
                raise FloatingPointError("nonfinite CE; optimizer update was not committed")
            (loss / denominator).backward()
            loss_sum += loss.detach().item()
            labels_by_depth += batch["valid"].sum(0).cpu()
            correct_by_depth += ((logits.detach().argmax(-1) == batch["labels"]) & batch["valid"]).sum(0).cpu()
    return dict(loss=loss_sum / max(local_labels, 1), nll_sum=loss_sum, valid_labels=local_labels,
                anchors=sum(len(roots) for _, _, roots in requests),
                request_ids=[entry["id"] for entry, _, _ in requests],
                request_anchors=[len(roots) for _, _, roots in requests],
                request_labels=per_request_labels,
                labels_by_depth=labels_by_depth.tolist(),
                correct_by_depth=correct_by_depth.tolist(),
                top1_by_depth=(correct_by_depth / labels_by_depth.clamp_min(1)).tolist())


def write_json(path, value):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + "\n")
    temporary.replace(path)


def save_training_checkpoint(path, model, optimizer, cursor, settings, entries, frozen_io, parent):
    rank, world_size = distributed_context()
    device = next(model.parameters()).device
    rng = dict(torch_rng=torch.get_rng_state(), python_rng=random.getstate())
    if device.type == "cuda":
        rng["cuda_rng"] = torch.cuda.get_rng_state(device)
    rank_rng = [rng]
    if world_size > 1:
        rank_rng = [None] * world_size
        dist.all_gather_object(rank_rng, rng)
    if rank != 0:
        dist.barrier()
        return
    payload = dict(schema="loop_drafter_checkpoint_v1", protocol=PROTOCOL,
                   config=model.config.dict(), model=model.state_dict(),
                   optimizer=optimizer.state_dict(), cursor=dict(cursor),
                   settings=asdict(settings), data_index=entries,
                   frozen_io=str(Path(frozen_io).resolve()), parent=parent,
                   split=dict(train=[entry["path"] for entry in entries], validation=[]),
                   distributed=dict(world_size=world_size, gradient_reduction="sum_global_label_normalized"),
                   rank_rng=rank_rng, **rng)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)
    if world_size > 1:
        dist.barrier()


def restore_training_state(payload, optimizer, settings, entries, device):
    if payload.get("protocol") != PROTOCOL or payload.get("settings") != asdict(settings):
        raise ValueError("resume protocol/settings differ from the saved training run")
    if payload.get("data_index") != entries:
        raise ValueError("training data identity, order, or file metadata changed since checkpoint")
    optimizer.load_state_dict(payload["optimizer"])
    rank, _ = distributed_context()
    saved_rng = payload.get("rank_rng", [payload])
    # Older single-rank checkpoints and a newly expanded world use rank 0's
    # state. Sampling itself is a pure function of global epoch/request index.
    rng = saved_rng[rank] if rank < len(saved_rng) else saved_rng[0]
    torch.set_rng_state(rng["torch_rng"])
    random.setstate(rng["python_rng"])
    if device.type == "cuda" and "cuda_rng" in rng:
        torch.cuda.set_rng_state(rng["cuda_rng"], device)
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
    return cursor


def reconcile_metrics(path, committed_step):
    """Keep only checkpointed updates after a crash; ignore an incomplete last line."""
    path = Path(path)
    if not path.exists():
        return 0
    kept, discarded = [], 0
    lines = path.read_text().splitlines()
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
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in kept))
    temporary.replace(path)
    return discarded


def train_epochs(model, entries, settings, out, frozen_io, *, resume_payload=None,
                 parent=None, save_every=50, stop_after_steps=0):
    """Train the complete planned budget, optionally pausing at an absolute step.

    ``stop_after_steps`` only interrupts execution for integration checks; it never
    changes the planned schedule. Resume restores the authoritative saved cursor.
    """
    if not entries or save_every < 1 or stop_after_steps < 0:
        raise ValueError("training requires requests and a positive save interval")
    if model.config.source_layout != "canonical":
        raise ValueError("this protocol requires canonical source layout")
    if settings.role == "baseline" and model.config.feature_mode != "final":
        raise ValueError("baseline role requires final-state features")
    if settings.role == "branch" and parent is None and resume_payload is None:
        raise ValueError("branch role requires a shared baseline parent")
    rank, world_size = distributed_context()
    primary = rank == 0
    if world_size not in (1, 2, 4) or world_size > settings.requests_per_update:
        raise ValueError("world size must be 1, 2 or 4 and cannot exceed global requests per update")
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
        model.load_state_dict(resume_payload["model"], strict=True)
        cursor = restore_training_state(resume_payload, optimizer, settings, entries, device)
        parent = resume_payload.get("parent")
        if primary:
            dropped = reconcile_metrics(metrics, cursor["step"])
            print(json.dumps(dict(event="resume", **cursor, discarded_uncommitted_metric_rows=dropped)), flush=True)
    else:
        if primary and ((out / "latest.pt").exists() or metrics.exists()):
            raise FileExistsError("output already contains training state; use --resume or a new directory")
        metadata = dict(protocol=PROTOCOL, config=model.config.dict(), settings=asdict(settings),
                        total_steps=total_steps, requests=len(entries), parent=parent,
                        distributed=dict(world_size=world_size, gradient_reduction="sum_global_label_normalized"),
                        loss="uniform_ce_over_all_valid_child_labels_in_update",
                        frozen_io=str(Path(frozen_io).resolve()),
                        trainable_parameters=sum(p.numel() for p in model.parameters()),
                        inactive_final_projection_parameters=(model.config.init_rank * model.config.target_hidden *
                            (model.config.rounds - 1) if model.config.feature_mode == "final" else 0))
        if primary:
            write_json(out / "config.json", metadata)
            write_json(out / "data_index.json", entries)
    if world_size > 1:
        # Initial parameter synchronization, also covering callers that created
        # the models with different local RNG states before entering this API.
        for parameter in model.parameters():
            dist.broadcast(parameter.data, src=0)
    if resume_payload is None:
        save_training_checkpoint(out / "initial.pt", model, optimizer, cursor, settings, entries, frozen_io, parent)
        save_training_checkpoint(out / "latest.pt", model, optimizer, cursor, settings, entries, frozen_io, parent)
    model.train()
    start_time = time.monotonic()
    if primary:
        print(json.dumps(dict(event="training_budget", role=settings.role, feature_mode=model.config.feature_mode,
                              requests=len(entries), epochs=settings.epochs, total_steps=total_steps,
                              world_size=world_size, remaining_steps=total_steps - cursor["step"])), flush=True)
    while cursor["epoch"] < settings.epochs:
        if stop_after_steps and cursor["step"] >= stop_after_steps:
            break
        epoch = cursor["epoch"]
        order = epoch_order(len(entries), settings.seed, epoch)
        start = cursor["next_request"]
        indices = order[start:start + settings.requests_per_update]
        local_indices = shard_indices(indices, rank, world_size)
        requests = prepare_requests(entries, local_indices, settings, epoch)
        optimizer.zero_grad(set_to_none=True)
        lr = learning_rate(cursor["step"] + 1, total_steps, settings)
        for group in optimizer.param_groups:
            group["lr"] = lr
        tick = time.monotonic()
        stats = synchronized_gradients(model, requests, settings.microbatch, device)
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), settings.grad_clip, error_if_nonfinite=True)
        optimizer.step()
        # The cursor always describes the NEXT update, including at epoch ends.
        end = start + len(indices)
        finished_epoch = end == len(entries)
        cursor = dict(step=cursor["step"] + 1, epoch=epoch + int(finished_epoch),
                      next_request=0 if finished_epoch else end,
                      total_requests=cursor["total_requests"] + len(indices),
                      total_anchors=cursor["total_anchors"] + stats["anchors"],
                      total_labels=cursor["total_labels"] + stats["valid_labels"])
        row = dict(event="update", step=cursor["step"], epoch=epoch + 1,
                   requests=len(indices), request_offset=start, lr=lr, grad_norm=float(grad_norm),
                   world_size=world_size, requests_by_rank=[len(shard_indices(indices, i, world_size)) for i in range(world_size)],
                   update_seconds=time.monotonic() - tick, **stats)
        if primary:
            with metrics.open("a") as handle:
                handle.write(json.dumps(row, ensure_ascii=False) + "\n")
            print(json.dumps(row, ensure_ascii=False), flush=True)
        if finished_epoch:
            save_training_checkpoint(out / f"epoch{epoch + 1}.pt", model, optimizer, cursor,
                                     settings, entries, frozen_io, parent)
        if finished_epoch or cursor["step"] % save_every == 0:
            save_training_checkpoint(out / "latest.pt", model, optimizer, cursor, settings, entries, frozen_io, parent)
        del requests
    save_training_checkpoint(out / "latest.pt", model, optimizer, cursor, settings, entries, frozen_io, parent)
    complete = cursor["epoch"] == settings.epochs
    if complete:
        save_training_checkpoint(out / "model.pt", model, optimizer, cursor, settings, entries, frozen_io, parent)
    result = dict(event="complete" if complete else "paused", **cursor, total_steps=total_steps,
                  world_size=world_size,
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
    p.add_argument("--data", required=True)
    p.add_argument("--frozen-io", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--role", choices=("baseline", "branch"), required=True)
    p.add_argument("--feature-mode", choices=("final", "multiloop", "trajectory"), default="final")
    p.add_argument("--init-checkpoint", help="canonical final checkpoint; required for a new branch")
    p.add_argument("--resume", action="store_true", help="resume out/latest.pt, including optimizer and RNG")
    p.add_argument("--epochs", type=int, help="defaults to baseline=4, branch=2")
    p.add_argument("--requests-per-update", type=int, default=8)
    p.add_argument("--max-anchors-per-request", type=int, default=64)
    p.add_argument("--batch-size", "--microbatch", dest="microbatch", type=int, default=16)
    p.add_argument("--expected-requests", type=int, default=8192, help="zero disables the exact count check")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--stop-after-steps", type=int, default=0, help="integration pause without changing the LR budget")
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--memory-fraction", type=float, default=.18)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--mlp-width", type=int, default=2304)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--init-rank", type=int, default=192)
    p.add_argument("--prefix-width", type=int, default=128)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.memory_fraction <= 1 or args.num_threads < 1 or args.expected_requests < 0:
        raise ValueError("invalid memory fraction, thread count or expected request count")
    if args.save_every < 1 or args.stop_after_steps < 0:
        raise ValueError("save interval must be positive and stop step nonnegative")
    settings = Settings(role=args.role, epochs=args.epochs if args.epochs is not None else (4 if args.role == "baseline" else 2),
                        requests_per_update=args.requests_per_update,
                        max_anchors_per_request=args.max_anchors_per_request, microbatch=args.microbatch,
                        seed=args.seed, peak_lr=1e-4 if args.role == "baseline" else 5e-5,
                        floor_lr=1e-5 if args.role == "baseline" else 5e-6)
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
    parent, payload, baseline = None, None, None
    if args.resume:
        model, payload = load_checkpoint(Path(args.out) / "latest.pt", args.frozen_io, device)
        if model.config.feature_mode != args.feature_mode:
            raise ValueError("resume feature mode disagrees with checkpoint")
    elif args.role == "branch":
        if not args.init_checkpoint:
            raise ValueError("a new branch requires --init-checkpoint")
        from .draft import fork_from_final
        model, baseline = fork_from_final(args.init_checkpoint, args.feature_mode, args.frozen_io, device)
        baseline_settings = baseline.get("settings", {})
        baseline_cursor = baseline.get("cursor", {})
        if (baseline.get("protocol") != PROTOCOL or baseline_settings.get("role") != "baseline"
                or baseline_cursor.get("epoch") != baseline_settings.get("epochs")
                or baseline_cursor.get("next_request") != 0):
            raise ValueError("branches must start from this protocol's completed baseline budget")
        parent = dict(checkpoint=str(Path(args.init_checkpoint).resolve()),
                      step=baseline_cursor["step"], epoch=baseline_cursor["epoch"])
    else:
        if args.feature_mode != "final" or args.init_checkpoint:
            raise ValueError("new baseline requires final mode and no init checkpoint")
        embedding, head = load_frozen_io(args.frozen_io)
        config = Config(target_hidden=embedding.shape[1], vocab_size=embedding.shape[0],
                        feature_mode="final", source_layout="canonical", depth=args.depth,
                        width=args.width, mlp_width=args.mlp_width, layers=args.layers,
                        init_rank=args.init_rank, prefix_width=args.prefix_width)
        model = LoopDraft(config, embedding, head).to(device)
    rank, world_size = distributed_context()
    entries = index_training_data(args.data, model.config, args.expected_requests) if rank == 0 else None
    if world_size > 1:
        message = [entries]
        dist.broadcast_object_list(message, src=0)
        entries = message[0]
    if baseline is not None and baseline.get("data_index") != entries:
        raise ValueError("branches must use the same ordered requests as their baseline")
    return train_epochs(model, entries, settings, args.out, args.frozen_io, resume_payload=payload,
                        parent=parent, save_every=args.save_every, stop_after_steps=args.stop_after_steps)


if __name__ == "__main__":
    main()
