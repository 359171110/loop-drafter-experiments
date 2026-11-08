"""CE-only pilot on frozen Ouro trajectory records, split by complete prompt.

Run from the repository root with ``python -m scripts.loop_drafter.train``.
One optimizer update covers one request's sampled anchors, normalized by its
valid child labels. Validation reports teacher-forced CE/top1; these are not
speculative acceptance or speedup measurements.
"""
import argparse
from contextlib import nullcontext
import hashlib
import json
import math
from pathlib import Path
import random
import time

import torch
from torch.nn import functional as F

from .draft import Config, LoopDraft, load_checkpoint, load_frozen_io


def load_record(path):
    return torch.load(path, map_location="cpu", weights_only=True, mmap=True)


def record_paths(source):
    source = Path(source).resolve()
    if source.is_dir():
        paths = sorted(source.glob("*.pt"))
    elif source.suffix == ".pt":
        paths = [source]
    else:
        rows = [json.loads(line) for line in source.read_text().splitlines() if line.strip()]
        paths = []
        for row in rows:
            path = Path(row if isinstance(row, str) else row["path"])
            paths.append(path if path.is_absolute() else source.parent / path)
    paths = [p.resolve() for p in paths]
    if not paths or len(set(paths)) != len(paths):
        raise ValueError("data must contain a nonempty list of unique trajectory paths")
    return paths


def validate_record(record, config):
    ids, hidden = record["token_ids"], record["round_hidden"]
    p = int(record["prompt_length"])
    if ids.ndim != 1 or ids.dtype != torch.int64 or not 1 <= p <= len(ids):
        raise ValueError("token_ids must be int64[T] with a valid prompt_length")
    if tuple(hidden.shape) != (len(ids) - 1, config.rounds, config.target_hidden):
        raise ValueError("round_hidden must be [T-1, rounds, target_hidden]")
    if not hidden.is_floating_point():
        raise ValueError("round_hidden must contain floating point states")
    if ids.numel() and (int(ids.min()) < 0 or int(ids.max()) >= config.vocab_size):
        raise ValueError("token IDs are outside the target vocabulary")


def prompt_group(record):
    # Stable prompt identity prevents different continuations of one prompt
    # crossing the split, including independently named copies of a record.
    ids = record["token_ids"][:int(record["prompt_length"])].tolist()
    return hashlib.blake2b(json.dumps(ids).encode(), digest_size=16).hexdigest()


def manifest_splits(source):
    source = Path(source).resolve()
    if source.is_dir() or source.suffix == ".pt":
        return {}
    declared = {}
    for line in source.read_text().splitlines():
        if not line.strip():
            continue
        row = json.loads(line)
        if isinstance(row, dict) and "split" in row:
            path = Path(row["path"])
            path = path if path.is_absolute() else source.parent / path
            declared[str(path.resolve())] = row["split"]
    return declared


def split_paths(paths, eval_fraction, seed, declared_splits=None):
    groups = {}
    assignments = {}
    for path in paths:
        record = load_record(path)
        group = prompt_group(record)
        groups.setdefault(group, []).append(path)
        manifest_split = (declared_splits or {}).get(str(path.resolve()))
        record_split = record.get("split")
        if manifest_split is not None and record_split is not None and manifest_split != record_split:
            raise ValueError("manifest and trajectory split disagree")
        assignments[path] = manifest_split if manifest_split is not None else record_split
    if any(value is not None for value in assignments.values()):
        if any(value not in ("train", "eval") for value in assignments.values()):
            raise ValueError("declared splits must assign every trajectory to train or eval")
        for values in groups.values():
            if len({assignments[p] for p in values}) != 1:
                raise ValueError("the same prompt occurs in both declared splits")
        train = sorted(p for p, split in assignments.items() if split == "train")
        valid = sorted(p for p, split in assignments.items() if split == "eval")
        if not train or not valid:
            raise ValueError("declared splits must contain train and eval trajectories")
        return train, valid
    keys = sorted(groups)
    if len(keys) < 2:
        raise ValueError("at least two distinct prompts are required for a held-out split")
    random.Random(seed).shuffle(keys)
    n_eval = min(len(keys) - 1, max(1, round(len(keys) * eval_fraction)))
    eval_keys = set(keys[:n_eval])
    train = sorted(p for k, values in groups.items() if k not in eval_keys for p in values)
    valid = sorted(p for k, values in groups.items() if k in eval_keys for p in values)
    return train, valid


def anchor_positions(record, max_anchors=0, seed=0):
    # At the initial root, s=P-1, bonus=first output token. Child labels start
    # at P+1. The last eligible root predicts the final output token/EOS.
    roots = list(range(int(record["prompt_length"]) - 1, len(record["token_ids"]) - 2))
    if max_anchors and len(roots) > max_anchors:
        roots = sorted(random.Random(seed).sample(roots, max_anchors))
    return roots


def make_batch(record, roots, depth, device="cpu"):
    ids = record["token_ids"]
    roots = torch.as_tensor(roots, dtype=torch.long)
    if not len(roots):
        raise ValueError("an anchor batch cannot be empty")
    if bool((roots < int(record["prompt_length"]) - 1).any()) or bool((roots + 2 >= len(ids)).any()):
        raise ValueError("anchors must have an output child and a processed prompt prefix")
    columns = roots[:, None] + 2 + torch.arange(depth)[None]
    valid = (columns < len(ids)) & (columns >= int(record["prompt_length"]))
    labels = ids[columns.clamp_max(len(ids) - 1)].masked_fill(~valid, -100)
    bonus = ids[roots + 1]
    previous = torch.cat((bonus[:, None], labels[:, :-1].clamp_min(0)), dim=1)
    return dict(taps=record["round_hidden"][roots].to(device), bonus=bonus.to(device),
                previous=previous.to(device), labels=labels.to(device), valid=valid.to(device))


def amp_context(device):
    return torch.autocast("cuda", dtype=torch.bfloat16) if device.type == "cuda" else nullcontext()


@torch.inference_mode()
def evaluate(model, paths, batch_size, max_anchors, seed, device):
    model.eval()
    nll, count = 0., 0
    correct = torch.zeros(model.config.depth, dtype=torch.long)
    totals = torch.zeros_like(correct)
    for i, path in enumerate(paths):
        record = load_record(path)
        roots = anchor_positions(record, max_anchors, seed + i)
        for start in range(0, len(roots), batch_size):
            batch = make_batch(record, roots[start:start + batch_size], model.config.depth, device)
            with amp_context(device):
                logits = model(batch["taps"], batch["bonus"], batch["previous"])
            nll += F.cross_entropy(logits.float().flatten(0, 1), batch["labels"].flatten(),
                                   ignore_index=-100, reduction="sum").item()
            valid = batch["valid"]
            count += int(valid.sum())
            correct += ((logits.argmax(-1) == batch["labels"]) & valid).sum(0).cpu()
            totals += valid.sum(0).cpu()
    model.train()
    return dict(metric_scope="teacher_forced_held_out", ce=nll / count if count else None,
                valid_labels=count, top1_by_depth=(correct / totals.clamp_min(1)).tolist(),
                labels_by_depth=totals.tolist())


def save_checkpoint(path, model, optimizer, cursor, args, split):
    payload = dict(schema="loop_drafter_checkpoint_v1", config=model.config.dict(),
                   model=model.state_dict(), optimizer=optimizer.state_dict(), cursor=cursor,
                   frozen_io=str(Path(args.frozen_io).resolve()), split=split,
                   train_args=vars(args), torch_rng=torch.get_rng_state(),
                   python_rng=random.getstate())
    if next(model.parameters()).device.type == "cuda":
        payload["cuda_rng"] = torch.cuda.get_rng_state(next(model.parameters()).device)
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", required=True, help="trajectory directory or JSONL path manifest")
    p.add_argument("--frozen-io", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--feature-mode", choices=("final", "multiloop", "trajectory"), default="trajectory")
    p.add_argument("--rounds", type=int, default=4)
    p.add_argument("--depth", type=int, default=5)
    p.add_argument("--width", type=int, default=768)
    p.add_argument("--mlp-width", type=int, default=2304)
    p.add_argument("--layers", type=int, default=4)
    p.add_argument("--init-rank", type=int, default=192)
    p.add_argument("--prefix-width", type=int, default=128)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--max-steps", type=int, default=300)
    p.add_argument("--batch-size", type=int, default=16, help="anchor microbatch; one request per optimizer update")
    p.add_argument("--max-anchors", "--max-anchors-per-request", dest="max_anchors", type=int, default=128)
    p.add_argument("--eval-max-anchors", type=int, default=64)
    p.add_argument("--eval-fraction", type=float, default=.1)
    p.add_argument("--eval-every", type=int, default=50)
    p.add_argument("--save-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cpu")
    p.add_argument("--memory-fraction", type=float, default=.18)
    p.add_argument("--num-threads", type=int, default=8)
    p.add_argument("--resume", action="store_true")
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    if not 0 < args.eval_fraction < 1 or not 0 < args.memory_fraction <= 1:
        raise ValueError("invalid evaluation or GPU memory fraction")
    if min(args.batch_size, args.epochs, args.max_steps, args.eval_every, args.save_every, args.num_threads) < 1:
        raise ValueError("batch/epoch/step/interval/thread counts must be positive")
    if args.max_anchors < 0 or args.eval_max_anchors < 0:
        raise ValueError("anchor limits must be nonnegative; zero means all")
    torch.set_num_threads(args.num_threads)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = torch.device(args.device)
    if device.type == "cuda":
        torch.cuda.set_device(device)
        torch.cuda.set_per_process_memory_fraction(args.memory_fraction, device)
        torch.cuda.manual_seed(args.seed)
    out = Path(args.out).resolve()
    if out.exists() and any(out.iterdir()) and not args.resume:
        raise ValueError("output directory is nonempty; use a new run or --resume")
    out.mkdir(parents=True, exist_ok=True)
    embedding, lm_head = load_frozen_io(args.frozen_io)
    config = Config(target_hidden=embedding.shape[1], vocab_size=embedding.shape[0],
                    rounds=args.rounds, feature_mode=args.feature_mode, depth=args.depth,
                    width=args.width, mlp_width=args.mlp_width, layers=args.layers,
                    init_rank=args.init_rank, prefix_width=args.prefix_width)
    paths = record_paths(args.data)
    eligible = []
    for path in paths:
        record = load_record(path)
        validate_record(record, config)
        if anchor_positions(record):
            eligible.append(path)
    train_paths, eval_paths = split_paths(eligible, args.eval_fraction, args.seed,
                                         declared_splits=manifest_splits(args.data))
    split = dict(train=[str(p) for p in train_paths], validation=[str(p) for p in eval_paths],
                 seed=args.seed, unit="unique_prompt_tokens", skipped_no_child=len(paths) - len(eligible))
    model = LoopDraft(config, embedding, lm_head).to(device)
    del embedding, lm_head
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(.9, .95), weight_decay=.01)
    cursor = dict(step=0, epoch=0, next_request=0)
    if args.resume:
        payload = torch.load(out / "latest.pt", map_location="cpu", weights_only=True)
        if payload["config"] != config.dict() or payload["split"] != split:
            raise ValueError("resume model or prompt split differs")
        for key in ("seed", "batch_size", "max_anchors", "lr"):
            if payload["train_args"][key] != vars(args)[key]:
                raise ValueError(f"resume training setting differs: {key}")
        if Path(payload["frozen_io"]).resolve() != Path(args.frozen_io).resolve():
            raise ValueError("resume frozen target IO differs")
        model.load_state_dict(payload["model"], strict=True)
        optimizer.load_state_dict(payload["optimizer"])
        cursor = payload["cursor"]
        torch.set_rng_state(payload["torch_rng"])
        random.setstate(payload["python_rng"])
        if device.type == "cuda" and "cuda_rng" in payload:
            torch.cuda.set_rng_state(payload["cuda_rng"], device)
    metadata = dict(config=config.dict(), split=split, args=vars(args),
                    trainable_parameters=sum(p.numel() for p in model.parameters()),
                    loss="whole-request valid-label normalized CE", target_frozen=True,
                    source="states through s; bonus token s+1; child labels s+2 onward")
    (out / "config.json").write_text(json.dumps(metadata, indent=2) + "\n")
    (out / "split.json").write_text(json.dumps(split, indent=2) + "\n")

    def log(row):
        with (out / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(row) + "\n")
        print(json.dumps(row), flush=True)

    model.train()
    if cursor["step"] == 0:
        save_checkpoint(out / "initial.pt", model, optimizer, cursor, args, split)
        save_checkpoint(out / "latest.pt", model, optimizer, cursor, args, split)
        log(dict(event="initial_validation", step=0,
                 **evaluate(model, eval_paths, args.batch_size, args.eval_max_anchors, args.seed, device)))
    start_epoch = cursor["epoch"]
    for epoch in range(start_epoch, args.epochs):
        order = list(range(len(train_paths)))
        random.Random(args.seed + epoch).shuffle(order)
        begin = cursor["next_request"] if epoch == start_epoch else 0
        for offset in range(begin, len(order)):
            if cursor["step"] >= args.max_steps:
                break
            index = order[offset]
            record = load_record(train_paths[index])
            roots = anchor_positions(record, args.max_anchors, args.seed + epoch * len(train_paths) + index)
            denominator = sum(min(config.depth, len(record["token_ids"]) - s - 2) for s in roots)
            optimizer.zero_grad(set_to_none=True)
            ce_sum = 0.
            started = time.perf_counter()
            for start in range(0, len(roots), args.batch_size):
                batch = make_batch(record, roots[start:start + args.batch_size], config.depth, device)
                with amp_context(device):
                    logits = model(batch["taps"], batch["bonus"], batch["previous"])
                ce = F.cross_entropy(logits.float().flatten(0, 1), batch["labels"].flatten(),
                                     ignore_index=-100, reduction="sum")
                (ce / denominator).backward()
                ce_sum += float(ce.detach())
            grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), 1.)
            if not math.isfinite(float(grad_norm)) or not math.isfinite(ce_sum):
                raise FloatingPointError("nonfinite training loss or gradient")
            optimizer.step()
            cursor = dict(step=cursor["step"] + 1, epoch=epoch, next_request=offset + 1)
            if cursor["next_request"] == len(order):
                cursor.update(epoch=epoch + 1, next_request=0)
            log(dict(event="train", **cursor, ce=ce_sum / denominator, anchors=len(roots),
                     valid_labels=denominator, gradient_norm=float(grad_norm), seconds=time.perf_counter() - started))
            if cursor["step"] % args.eval_every == 0:
                log(dict(event="validation", step=cursor["step"],
                         **evaluate(model, eval_paths, args.batch_size, args.eval_max_anchors, args.seed, device)))
            if cursor["step"] % args.save_every == 0:
                save_checkpoint(out / "latest.pt", model, optimizer, cursor, args, split)
        if cursor["step"] >= args.max_steps:
            break
    log(dict(event="final_validation", step=cursor["step"],
             **evaluate(model, eval_paths, args.batch_size, args.eval_max_anchors, args.seed, device)))
    save_checkpoint(out / "latest.pt", model, optimizer, cursor, args, split)


if __name__ == "__main__":
    main()
