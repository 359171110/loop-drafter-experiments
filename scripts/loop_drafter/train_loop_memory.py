"""Train loop-memory ablations from random weights or an explicit legacy SSD.

--from-scratch uses four epochs on the new 16K data, sampling epochs 0..3,
and no drafter parent. The explicit legacy checkpoint path retains the former
two-epoch continuation protocol. Resume preserves committed updates in either.
"""
import argparse
from dataclasses import asdict
import hashlib
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
from .loop_memory_draft import (LoopMemoryConfig, LoopMemoryDraft, fork_from_ssd,
                                load_loop_memory_checkpoint, make_loop_memory_batch)
from .train_history import load_training_index, reconcile_metrics
from .train import amp_context, validate_record
from .train_protocol import (Settings, distributed_context, epoch_order, index_training_data,
                             PROTOCOL as BASELINE_PROTOCOL, learning_rate, merge_statistics, prepare_requests,
                             request_label_counts, shard_indices, write_json)


PROTOCOL = "loop_memory_warmstart_ce_v1"
SCRATCH_PROTOCOL = "loop_memory_from_scratch_ce_v2"
SCHEMA = "loop_memory_drafter_checkpoint_v1"
BACKBONE_KEYS = ("width", "mlp_width", "layers", "init_rank", "prefix_width", "depth",
                 "state_dim", "state_head_dim", "state_expand")


def initialize_from_scratch(config, frozen_io, *, seed=0, device="cpu"):
    """Load only frozen target I/O; construct every drafter parameter afresh."""
    embedding, head = load_frozen_io(frozen_io)
    if isinstance(config, dict):
        config = LoopMemoryConfig(target_hidden=embedding.shape[1], vocab_size=embedding.shape[0], **config)
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LoopMemoryDraft(config, embedding, head)
    return model.to(device)


def initial_state_hash(state):
    """Stable identity of drafter state only; frozen target I/O is nonpersistent."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        description = [name, str(tensor.dtype), list(tensor.shape)]
        digest.update(json.dumps(description, separators=(",", ":")).encode() + b"\n")
        digest.update(tensor.reshape(-1).view(torch.uint8).numpy().tobytes())
    return digest.hexdigest()


def validate_initialization(payload, from_scratch):
    expected_protocol = SCRATCH_PROTOCOL if from_scratch else PROTOCOL
    if (payload.get("protocol") != expected_protocol
            or payload.get("from_scratch", False) != from_scratch):
        raise ValueError("resume initialization mode differs; scratch runs require --from-scratch")
    if from_scratch and payload.get("parent") is not None:
        raise ValueError("from-scratch checkpoint must not have a drafter parent")
    if from_scratch:
        identity = payload.get("initialization") or {}
        fingerprint = identity.get("initial_state_sha256", "")
        if (identity.get("kind") != "from_scratch" or identity.get("seed") != payload.get("settings", {}).get("seed")
                or len(fingerprint) != 64 or any(c not in "0123456789abcdef" for c in fingerprint)):
            raise ValueError("scratch resume checkpoint is missing its original initialization identity")
    if not from_scratch and payload.get("parent") is None:
        raise ValueError("legacy resume checkpoint is missing its SSD parent identity")


def execution_microbatch(settings, plan, completed_steps):
    if plan is None:
        return settings.microbatch
    if (set(plan) != {'schema', 'microbatch', 'from_step'}
            or plan['schema'] != 'loop_training_execution_v1'
            or type(plan['microbatch']) is not int or plan['microbatch'] < 1
            or type(plan['from_step']) is not int or plan['from_step'] < 0):
        raise ValueError('invalid microbatch execution schedule')
    return plan['microbatch'] if completed_steps >= plan['from_step'] else settings.microbatch


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
            batch = make_loop_memory_batch(record, roots[start:start + microbatch], depth,
                                       model.config.window, device)
            with amp_context(device):
                logits = model.forward_batch(batch)
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
                             sampling_epoch_offset, parent, from_scratch=False, initialization=None, execution_plan=None):
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
        payload = dict(schema=SCHEMA, protocol=SCRATCH_PROTOCOL if from_scratch else PROTOCOL,
                       from_scratch=from_scratch, run_directory=str(Path(path).parent.resolve()),
                       initialization=initialization, execution_plan=execution_plan,
                       config=model.config.dict(),
                       model=model.state_dict(), optimizer=optimizer.state_dict(), cursor=dict(cursor),
                       settings=asdict(settings), data_index=entries,
                       sampling_epoch_offset=sampling_epoch_offset, parent=parent,
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


def restore_training_state(payload, optimizer, settings, entries, device, sampling_epoch_offset,
                           from_scratch=False):
    validate_initialization(payload, from_scratch)
    if (payload.get("schema") != SCHEMA
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


def train_epochs(model, entries, settings, out, frozen_io, *, resume_payload=None,
                 sampling_epoch_offset=None, parent=None, save_every=50, stop_after_steps=0,
                 from_scratch=False, execution_plan=None):
    """Train a randomly initialized or legacy arm from its next committed update."""
    if sampling_epoch_offset is None:
        sampling_epoch_offset = 0 if from_scratch else 4
    if not entries or save_every < 1 or stop_after_steps < 0 or sampling_epoch_offset < 0:
        raise ValueError("invalid data, save interval, stop step or sampling offset")
    expected_role = "baseline" if from_scratch else "branch"
    if settings.role != expected_role or model.config.memory_mode not in ("none", "final", "final_capacity", "trajectory"):
        raise ValueError(f"loop memory initialization requires {expected_role} settings and a valid memory mode")
    if from_scratch and parent is not None:
        raise ValueError("from-scratch training must not have a drafter parent")
    if not from_scratch and parent is None and resume_payload is None:
        raise ValueError("loop memory training requires a shared SSD baseline parent")
    if resume_payload is not None:
        prior_plan = resume_payload.get('execution_plan')
        if execution_plan is None:
            execution_plan = prior_plan
        elif prior_plan is not None and execution_plan != prior_plan:
            raise ValueError('resume microbatch execution schedule changed')
        elif prior_plan is None and resume_payload['cursor']['step'] > execution_plan['from_step']:
            raise ValueError('cannot introduce an execution schedule retroactively')
    execution_microbatch(settings, execution_plan, 0)
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
    initialization = None
    if resume_payload is not None:
        validate_initialization(resume_payload, from_scratch)
        if from_scratch and resume_payload.get("run_directory") != str(out.resolve()):
            raise ValueError("scratch resume must restore the same run directory")
        if resume_payload["config"] != model.config.dict():
            raise ValueError("resume model config changed")
        if Path(resume_payload["frozen_io"]).resolve() != Path(frozen_io).resolve():
            raise ValueError("resume frozen IO path differs from checkpoint")
        model.load_state_dict(resume_payload["model"], strict=True)
        cursor = restore_training_state(resume_payload, optimizer, settings, entries, device,
                                        sampling_epoch_offset, from_scratch)
        parent = resume_payload.get("parent")
        initialization = resume_payload.get("initialization")
        if primary:
            dropped = reconcile_metrics(metrics, cursor["step"])
            print(json.dumps(dict(event="resume", **cursor, discarded_uncommitted_metric_rows=dropped)), flush=True)
    else:
        if primary and ((out / "latest.pt").exists() or metrics.exists()):
            raise FileExistsError("output already contains training state; use --resume or a new directory")
        if from_scratch:
            if primary:
                initialization = dict(kind="from_scratch", seed=settings.seed,
                                      initial_state_sha256=initial_state_hash(model.state_dict()))
            if world_size > 1:
                identity = [initialization]
                dist.broadcast_object_list(identity, src=0)
                initialization = identity[0]
        if primary:
            write_json(out / "config.json", dict(
                protocol=SCRATCH_PROTOCOL if from_scratch else PROTOCOL, from_scratch=from_scratch,
                initialization=initialization,
                run_directory=str(out.resolve()), config=model.config.dict(), settings=asdict(settings),
                sampling_epoch_offset=sampling_epoch_offset, parent=parent, optimizer_initialization="fresh_AdamW",
                total_steps=total_steps, requests=len(entries),
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
                                 frozen_io, sampling_epoch_offset, parent, from_scratch, initialization, execution_plan)

    if resume_payload is None:
        save("initial.pt")
        save("latest.pt")
    model.train()
    start_time = time.monotonic()
    if primary:
        print(json.dumps(dict(event="training_budget", memory_mode=model.config.memory_mode,
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
        physical_microbatch = execution_microbatch(settings, execution_plan, cursor["step"])
        stats = synchronized_gradients(model, requests, physical_microbatch, device)
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
                   execution_microbatch=physical_microbatch,
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
    p.add_argument("--memory-mode", choices=("none", "final", "final_capacity", "trajectory"), required=True)
    initialization = p.add_mutually_exclusive_group()
    initialization.add_argument("--init-checkpoint", help="explicit legacy completed four-epoch canonical final SSD baseline")
    initialization.add_argument("--from-scratch", action="store_true", help="random drafter weights; reuse only frozen target I/O")
    p.add_argument("--index-file", help="validated baseline data_index.json with sibling config.json")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--epochs", type=int, default=None, help="scratch default 4; legacy default 2")
    p.add_argument("--sampling-epoch-offset", type=int, default=None, help="scratch default 0; legacy default 4")
    p.add_argument("--requests-per-update", type=int, default=8)
    p.add_argument("--max-anchors-per-request", type=int, default=64)
    p.add_argument("--batch-size", "--microbatch", dest="microbatch", type=int, default=16)
    p.add_argument("--expected-requests", type=int, default=None, help="scratch default 16384; legacy default 8192")
    p.add_argument("--seed", type=int, choices=[0], default=0)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--stop-after-steps", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--local-rank", "--local_rank", type=int, default=None, help=argparse.SUPPRESS)
    p.add_argument("--memory-fraction", type=float, default=.18)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--memory-width", type=int, default=384)
    p.add_argument("--memory-heads", type=int, default=6)
    p.add_argument("--window", type=int, default=64)
    defaults = LoopMemoryConfig()
    for key in BACKBONE_KEYS:
        p.add_argument("--" + key.replace("_", "-"), type=int, default=getattr(defaults, key),
                       help="random initialization architecture (legacy checkpoints retain their own dimensions)")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not args.from_scratch and not args.init_checkpoint and not args.resume:
        raise ValueError("new training requires explicit --from-scratch or --init-checkpoint")
    if args.epochs is None:
        args.epochs = 4 if args.from_scratch else 2
    if args.sampling_epoch_offset is None:
        args.sampling_epoch_offset = 0 if args.from_scratch else 4
    if args.expected_requests is None:
        args.expected_requests = 16384 if args.from_scratch else 8192
    if not 0 < args.memory_fraction <= 1 or args.num_threads < 1 or args.expected_requests < 0:
        raise ValueError("invalid memory fraction, thread count or expected request count")
    if args.save_every < 1 or args.stop_after_steps < 0 or args.sampling_epoch_offset < 0:
        raise ValueError("invalid save interval, stop step or sampling offset")
    settings = Settings(role="baseline" if args.from_scratch else "branch", epochs=args.epochs, requests_per_update=args.requests_per_update,
                        max_anchors_per_request=args.max_anchors_per_request,
                        microbatch=args.microbatch, seed=args.seed,
                        peak_lr=1e-4 if args.from_scratch else 5e-5,
                        floor_lr=1e-5 if args.from_scratch else 5e-6)
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
    baseline, parent, payload = None, None, None
    if args.resume:
        model, payload = load_loop_memory_checkpoint(Path(args.out) / "latest.pt", args.frozen_io, device)
        validate_initialization(payload, args.from_scratch)
        if model.config.memory_mode != args.memory_mode:
            raise ValueError("resume memory mode differs from checkpoint")
        config_keys = ("memory_width", "memory_heads", "window") + (BACKBONE_KEYS if args.from_scratch else ())
        if any(getattr(model.config, key) != getattr(args, key) for key in config_keys):
            raise ValueError("resume memory configuration differs from checkpoint")
        if args.init_checkpoint and str(Path(args.init_checkpoint).resolve()) != payload.get("parent", {}).get("checkpoint"):
            raise ValueError("resume SSD parent differs from checkpoint")
    elif args.from_scratch:
        config = dict(memory_mode=args.memory_mode, memory_width=args.memory_width,
                      memory_heads=args.memory_heads, window=args.window,
                      **{key: getattr(args, key) for key in BACKBONE_KEYS})
        model = initialize_from_scratch(config, args.frozen_io, seed=args.seed, device=device)
    else:
        if not args.init_checkpoint:
            raise ValueError("new training requires --init-checkpoint")
        model, baseline = fork_from_ssd(args.init_checkpoint, args.memory_mode, args.frozen_io, device, seed=args.seed,
                                        memory_width=args.memory_width, memory_heads=args.memory_heads,
                                        window=args.window)
        saved_settings, saved_cursor = baseline.get("settings", {}), baseline.get("cursor", {})
        if (baseline.get("protocol") != BASELINE_PROTOCOL or saved_settings.get("role") != "baseline"
                or saved_settings.get("epochs") != 4 or saved_cursor.get("epoch") != 4
                or saved_cursor.get("next_request") != 0):
            raise ValueError("warm-start requires a completed four-epoch baseline")
        stat = Path(args.init_checkpoint).stat()
        parent = dict(checkpoint=str(Path(args.init_checkpoint).resolve()), step=saved_cursor["step"],
                      epoch=saved_cursor["epoch"], file_size=stat.st_size, mtime_ns=stat.st_mtime_ns)
    rank, world_size = distributed_context()
    entries = None
    if rank == 0:
        entries = load_training_index(args.data, model.config, args.expected_requests, args.index_file)
    if world_size > 1:
        message = [entries]
        dist.broadcast_object_list(message, src=0)
        entries = message[0]
    if baseline is not None and baseline.get("data_index") != entries:
        raise ValueError("all arms must use the same ordered data as their SSD baseline")
    execution_path = Path(args.out).parent / 'training_execution.json'
    execution_plan = json.loads(execution_path.read_text()) if execution_path.is_file() else None
    return train_epochs(model, entries, settings, args.out, args.frozen_io, resume_payload=payload,
                        sampling_epoch_offset=args.sampling_epoch_offset, parent=parent,
                        save_every=args.save_every, stop_after_steps=args.stop_after_steps,
                        from_scratch=args.from_scratch, execution_plan=execution_plan)


if __name__ == "__main__":
    main()
