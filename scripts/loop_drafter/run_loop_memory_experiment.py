"""Prepare the scratch v2 recipe on CPU; --execute requires its complete data.

The core and reasoning panels have separate shards and summaries. This entry
never loads a previous drafter. Legacy warmstart runs retain their archived
source snapshots and the trainer's explicit --init-checkpoint interface.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import time

from .capture import read_jsonl
from .evaluate_protocol import aggregate, atomic_json


MODES = ("none", "final", "final_capacity", "trajectory")
METHODS = ("ar", *MODES)



def now():
    return datetime.now(timezone.utc).isoformat()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(4 << 20), b""):
            result.update(block)
    return result.hexdigest()


def object_digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def freeze_json(path, value):
    path = Path(path)
    if path.exists():
        if json.loads(path.read_text()) != value:
            raise ValueError(f"frozen input changed: {path}")
    else:
        atomic_json(path, value)


def panels(prompts, workers, seed=0):
    """Balanced shards support unequal dataset sizes, including HumanEval164."""
    if workers < 1 or len({p["id"] for p in prompts}) != len(prompts):
        raise ValueError("positive worker count and unique prompt IDs required")
    result = [[] for _ in range(workers)]
    rng = random.Random(seed)
    for source in sorted({p["source"] for p in prompts}):
        rows = [p for p in prompts if p["source"] == source]
        rng.shuffle(rows)
        for i, row in enumerate(rows):
            result[i % workers].append(row)
    return result


def training_settings(recipe):
    """Translate the registered recipe, rather than inheriting a parent run."""
    t = recipe["training"]
    if t["initialization"] != "from_scratch" or t["parent_checkpoint"] is not None:
        raise ValueError("this runner only supports from_scratch with parent_checkpoint=null")
    if t["variants"] != list(MODES) or recipe["evaluation"]["methods"] != list(METHODS):
        raise ValueError("the registered four modes and AR reference are required")
    if recipe["seed"] != 0 or t["sampling_epoch_offset"] != 0:
        raise ValueError("scratch uses seed 0 and sampling epoch offset 0")
    if t["sampling_epochs"] != list(range(t["epochs"])):
        raise ValueError("sampling epochs must cover the complete scratch run")
    count, batch = recipe["training_data"]["effective_requests"], t["requests_per_update"]
    if (t["updates_per_variant"] != ((count + batch - 1) // batch) * t["epochs"]
            or t["request_exposures_per_variant"] != count * t["epochs"]):
        raise ValueError("registered update/request budget does not match complete epochs")
    # These are currently fixed by train_loop_memory's scratch implementation.
    fixed = {"peak_lr": 1e-4, "floor_lr": 1e-5, "warmup_fraction": .05,
             "weight_decay": .01, "grad_clip": 1., "adam_betas": [.9, .95]}
    if any(t[key] != value for key, value in fixed.items()):
        raise ValueError("recipe optimizer differs from the scratch trainer")
    return dict(role="baseline", epochs=t["epochs"], requests_per_update=batch,
                max_anchors_per_request=t["max_anchors_per_request"], microbatch=t["microbatch"],
                seed=recipe["seed"], peak_lr=t["peak_lr"], floor_lr=t["floor_lr"],
                warmup_fraction=t["warmup_fraction"], weight_decay=t["weight_decay"],
                beta1=t["adam_betas"][0], beta2=t["adam_betas"][1], grad_clip=t["grad_clip"],
                sampling_epoch_offset=t["sampling_epoch_offset"])


def sampling_summary(entries, training, depth):
    # Actual sampler, with token-count-only records: no hidden tensor loading.
    from .train import anchor_positions
    from .train_protocol import epoch_order
    result = []
    for epoch in range(training["sampling_epoch_offset"],
                       training["sampling_epoch_offset"] + training["epochs"]):
        fingerprint = hashlib.sha256()
        by_source = {}
        for index in epoch_order(len(entries), training["seed"], epoch):
            entry = entries[index]
            record = {"prompt_length": entry["prompt_length"],
                      "token_ids": range(entry["token_count"])}
            roots = anchor_positions(record, training["max_anchors_per_request"],
                                     training["seed"] + epoch * len(entries) + index)
            fingerprint.update(json.dumps([entry["id"], roots], separators=(",", ":")).encode())
            totals = by_source.setdefault(entry["source"], dict(requests=0, anchors=0, valid_labels=0))
            totals["requests"] += 1
            totals["anchors"] += len(roots)
            totals["valid_labels"] += sum(min(depth, entry["token_count"] - root - 2) for root in roots)
        result.append(dict(sampling_epoch=epoch, requests=len(entries),
                           updates=(len(entries) + training["requests_per_update"] - 1)
                                   // training["requests_per_update"],
                           anchors=sum(r["anchors"] for r in by_source.values()),
                           valid_labels=sum(r["valid_labels"] for r in by_source.values()),
                           by_source=by_source, request_anchor_sha256=fingerprint.hexdigest()))
    return result


def input_identity(path):
    path = Path(path).resolve()
    if not path.is_file():
        return dict(path=str(path), exists=False)
    return dict(path=str(path), exists=True, bytes=path.stat().st_size, sha256=digest(path))


def read_panel(manifest, requests, maximum):
    path = Path(manifest).resolve()
    rows = read_jsonl(path)
    metadata = json.loads(path.with_name("sources.json").read_text())
    if len(rows) != requests or len({r["id"] for r in rows}) != requests:
        raise ValueError(f"evaluation count or unique IDs differ: {path}")
    if any(r.get("split") != "eval" or not 1 <= len(r.get("input_ids", [])) <= maximum for r in rows):
        raise ValueError(f"evaluation requires frozen valid input_ids and split=eval: {path}")
    counts = dict(Counter(r["source"] for r in rows))
    if metadata.get("expected_counts") != counts:
        raise ValueError(f"evaluation source counts differ from sources.json: {path}")
    if any("mbpp" in source.lower() for source in counts):
        raise ValueError("MBPP is not part of the scratch evaluation protocol")
    identity = metadata.get("prompts_file")
    if identity and identity.get("sha256") != digest(path):
        raise ValueError(f"evaluation prompt identity changed: {path}")
    return rows


def prepare_protocol(recipe_path, run, repo):
    """Return a CPU-only plan; freeze executable inputs only after data is complete.

    Capture must publish manifest.jsonl, data_index.json, and config.json in
    the same directory. The latter has a `config` dict with target_hidden,
    rounds and vocab_size, as required by load_training_index.
    """
    from .loop_memory_draft import LoopMemoryConfig
    from .train_history import load_training_index
    recipe = json.loads(Path(recipe_path).read_text())
    training = training_settings(recipe)
    t, e, capture = recipe["training"], recipe["evaluation"], recipe["capture"]
    config = LoopMemoryConfig(memory_mode="trajectory", depth=t["depth"], window=t["history_window"])
    if recipe["architecture_version"] != config.architecture:
        raise ValueError("recipe architecture does not match LoopMemoryConfig")
    if (capture["target_rounds"] != config.rounds or capture["target_dtype"] != "float32"
            or e["target_rounds"] != config.rounds or e["target_dtype"] != "float32"
            or e["batch_size"] != 1 or e["drafter_autocast"] != "bfloat16"):
        raise ValueError("registered target/evaluation implementation differs")
    data = Path(recipe["training_data"]["trajectory_manifest"]).resolve()
    paths = {"recipe": Path(recipe_path), "data": data,
             "data_index": data.with_name("data_index.json"),
             "index_config": data.with_name("config.json"),
             "frozen_io": Path(t["frozen_io"]),
             "train_prompts": Path(recipe["training_data"]["prompt_manifest"])}
    suites = {}
    for name in ("core", "reasoning_extension"):
        section = e[name]
        paths[name + "_prompts"] = Path(section["manifest"])
        paths[name + "_sources"] = Path(section["manifest"]).with_name("sources.json")
        suites[name] = read_panel(section["manifest"], section["requests"], e["max_prompt_tokens"])
    if {p["id"] for p in suites["core"]} & {p["id"] for p in suites["reasoning_extension"]}:
        raise ValueError("core and reasoning panels overlap")
    identities = {name: input_identity(path) for name, path in paths.items()}
    missing = [dict(name=name, path=row["path"]) for name, row in identities.items() if not row["exists"]]
    expected = recipe["training_data"]["effective_requests"]
    observed = None
    if data.is_file():
        rows = read_jsonl(data)
        if any(row.get("split") != "train" for row in rows):
            raise ValueError("effective capture manifest must contain only explicit train rows")
        observed = len(rows)
        if len({r["id"] for r in rows}) != observed or observed > expected:
            raise ValueError("effective capture manifest has duplicate IDs or excess requests")
    incomplete = observed is not None and observed != expected
    entries = None
    if not missing and not incomplete:
        entries = load_training_index(data, config, expected_requests=expected, index_file=paths["data_index"])
        if any(entry["prompt_length"] > capture["max_prompt_tokens"]
               or entry["token_count"] - entry["prompt_length"] > capture["max_new_tokens"] for entry in entries):
            raise ValueError("trajectory token budget exceeds the registered capture limits")
    ready = entries is not None
    comparison = dict(inputs=identities, training=training, initialization="from_scratch",
                      parent_checkpoint=None, expected_requests=expected,
                      sampling=sampling_summary(entries, training, config.depth) if ready else None,
                      optimizer="fresh AdamW in every mode", loss=t["loss"],
                      checkpoint_selection=t["checkpoint_selection"], additional_seeds=[], development_panel=None,
                      target_model=capture["target"], target_rounds=capture["target_rounds"],
                      target_dtype=capture["target_dtype"], attention_backend=capture["attention_backend"],
                      feature_dtype=capture["feature_dtype"], drafter_autocast=e["drafter_autocast"],
                      max_prompt_tokens=e["max_prompt_tokens"], max_new_tokens=e["max_new_tokens"],
                      batch_size=e["batch_size"], decoding=e["decoding"],
                      train_max_prompt_tokens=capture["max_prompt_tokens"], train_max_new_tokens=capture["max_new_tokens"],
                      eval_source_counts={name: dict(Counter(p["source"] for p in rows)) for name, rows in suites.items()},
                      panels_reported_separately=True, python=recipe["runtime"]["python"],
                      architecture=config.dict(), metrics=e["metrics"], timing=e["timing"])
    protocol = dict(version=recipe["protocol_revision"], comparison_id=object_digest(comparison),
                    architecture=config.dict(), variants=list(MODES), initialization="from_scratch",
                    parent_checkpoint=None, seed=recipe["seed"],
                    trainable_initialization=t["trainable_initialization"],
                    new_parameter_limit=5_000_000,
                    memory_parameters_allocated=4_277_376, total_parameters_allocated=67_872_928,
                    memory_parameters_structurally_active={"none": 0, "final": 1_671_168,
                                                          "final_capacity": 3_490_944, "trajectory": 4_277_376},
                    final_capacity_control="same allocated parameter shapes; h4-only, delta input zero; not identical effective capacity")
    state = "prepared" if ready else "awaiting_trajectories"
    preparation = dict(state=state, missing_inputs=missing, observed_trajectory_requests=observed,
                       expected_trajectory_requests=expected, training_started=False,
                       index_contract="capture/data_index.json plus sibling config.json with config.target_hidden/rounds/vocab_size",
                       recipe=recipe, comparison=comparison)
    if ready:
        target = Path(capture["target"])
        comparison["target_files"] = {
            str(path.relative_to(target)): {"bytes": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in sorted(target.iterdir()) if path.is_file()}
        protocol["comparison_id"] = object_digest(comparison)
        freeze_json(run / "recipe.json", recipe)
        freeze_json(run / "comparison_protocol.json", comparison)
        freeze_json(run / "protocol.json", protocol)
        sources = [p for p in (repo / "scripts" / "loop_drafter").glob("*.py") if not p.name.startswith("test_")]
        sources += [repo / "scripts" / "fest_hybrid" / "model.py",
                    repo / "scripts" / "fest_hybrid_prefix_tree" / "model.py"]
        freeze_json(run / "source_identity.json", {str(p.relative_to(repo)): digest(p) for p in sorted(sources)})
        for source in sources:
            snapshot = run / "source_snapshot" / source.relative_to(repo)
            if not snapshot.exists():
                snapshot.parent.mkdir(parents=True, exist_ok=True)
                snapshot.write_bytes(source.read_bytes())
    elif (run / "comparison_protocol.json").exists():
        raise ValueError("previously frozen trajectory inputs are now missing or incomplete")
    atomic_json(run / "preparation.json", preparation)
    return comparison, protocol, suites, preparation


def job_specs(run, comparison, suites, gpus, memory_fraction, num_threads, *, allow_repartition=False):
    paths = {key: row["path"] for key, row in comparison["inputs"].items()}
    t, c = comparison["training"], comparison["architecture"]
    common = ["--device", "cuda:0", "--memory-fraction", str(memory_fraction),
              "--num-threads", str(num_threads)]
    jobs = []
    for i, mode in enumerate(MODES):
        group = gpus[i::4] if len(gpus) >= 4 else [gpus[i % len(gpus)]]
        command = [comparison["python"], "-m"]
        if len(group) > 1:
            command += ["torch.distributed.run", "--standalone", f"--nproc_per_node={len(group)}", "-m"]
        out = run / mode
        command += ["scripts.loop_drafter.train_loop_memory", "--from-scratch", "--data", paths["data"],
                    "--frozen-io", paths["frozen_io"], "--index-file", paths["data_index"], "--out", str(out),
                    "--memory-mode", mode, "--expected-requests", str(comparison["expected_requests"]),
                    "--epochs", str(t["epochs"]), "--sampling-epoch-offset", str(t["sampling_epoch_offset"]),
                    "--seed", str(t["seed"]), "--requests-per-update", str(t["requests_per_update"]),
                    "--max-anchors-per-request", str(t["max_anchors_per_request"]), "--microbatch", str(t["microbatch"]),
                    "--memory-width", str(c["memory_width"]), "--memory-heads", str(c["memory_heads"]),
                    "--window", str(c["window"]), "--depth", str(c["depth"]), *common]
        jobs.append(dict(name=mode, stage="training", gpus=group, command=command,
                         artifact=str(out / "model.pt")))
    for name, prompts in suites.items():
        for i, (panel, gpu) in enumerate(zip(panels(prompts, len(gpus), t["seed"]), gpus)):
            if not panel:
                raise ValueError("evaluation worker has no requests; use fewer GPUs")
            path = run / "evaluation" / name / "panels" / f"worker{i}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            serialized = "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in panel)
            if path.exists() and path.read_text() != serialized and not allow_repartition:
                raise ValueError("frozen evaluation shard changed")
            path.write_text(serialized)
            out = run / "evaluation" / name / "shards" / f"worker{i}"
            command = [comparison["python"], "-m", "scripts.loop_drafter.evaluate_loop_memory",
                       "--model", comparison["target_model"], "--prompts", str(path),
                       "--frozen-io", paths["frozen_io"], "--out", str(out),
                       "--max-prompt-tokens", str(comparison["max_prompt_tokens"]),
                       "--max-new-tokens", str(comparison["max_new_tokens"]), "--seed", str(t["seed"]),
                       "--tokenizer-manifest", paths[name + "_sources"],
                       "--expected-counts", json.dumps(dict(Counter(row["source"] for row in panel)), sort_keys=True)]
            for mode in MODES:
                command += [f"--{mode.replace('_', '-')}-checkpoint", str(run / mode / "model.pt")]
            command += common
            jobs.append(dict(name=f"evaluation_{name}_{i}", stage="evaluation", panel=name, gpus=[gpu],
                             command=command, artifact=str(out / "summary.json")))
    return jobs


def merge_evaluation(root, prompts, gpus):
    expected = {(p["id"], method): p["source"] for p in prompts for method in METHODS}
    rows, hardware, per_gpu, seen = [], [], [], set()
    for i, gpu in enumerate(gpus):
        panel = read_jsonl(root / "panels" / f"worker{i}.jsonl")
        shard = root / "shards" / f"worker{i}"
        records = read_jsonl(shard / "requests.jsonl")
        meta = json.loads((shard / "summary.json").read_text())["hardware"]
        keys = [(r["id"], r["method"]) for r in records]
        if len(set(keys)) != len(keys):
            raise ValueError("duplicate evaluation records")
        if set(keys) != {(p["id"], m) for p in panel for m in METHODS}:
            raise ValueError("incomplete evaluation shard")
        if meta["cuda_visible_devices"] != gpu:
            raise ValueError("evaluation GPU changed")
        for row in records:
            key = row["id"], row["method"]
            if key in seen or expected.get(key) != row["source"]:
                raise ValueError("duplicate or foreign evaluation record")
            seen.add(key)
            rows.append(row | {"worker_index": i, "physical_gpu": gpu})
        hardware.append(meta)
        per_gpu.append(dict(physical_gpu=gpu, results=aggregate(records)))
    if seen != set(expected):
        raise ValueError("incomplete evaluation coverage")
    (root / "requests.jsonl").write_text("".join(json.dumps(row) + "\n" for row in rows))
    costs = []
    for source in ("__all__", *sorted({r["source"] for r in rows})):
        for method in METHODS:
            selected = [r for r in rows if r["method"] == method and
                        (source == "__all__" or r["source"] == source)]
            cycles = sum(len(r["rounds"]) for r in selected)
            totals = {key: sum(r.get(key, 0.) for r in selected)
                      for key in ("proposal_seconds", "verify_seconds", "memory_update_seconds")}
            costs.append(dict(source=source, method=method, requests=len(selected), verify_cycles=cycles,
                              **totals, milliseconds_per_verify={
                                  key: 1000 * value / cycles if cycles else None
                                  for key, value in totals.items()}))
    result = dict(source_counts=dict(Counter(p["source"] for p in prompts)),
                  results=aggregate(rows), results_by_gpu=per_gpu, hardware=hardware,
                  component_costs=costs,
                  timing_scope="shared_gpu_interleaved_diagnostic",
                  output_comparison="token identity only; no task correctness score")
    atomic_json(root / "summary.json", result)
    return result


def execute(jobs, run, repo, status, lock, threads):
    def save():
        status["updated_at"] = now()
        atomic_json(run / "status.json", status)
    for stage in ("training", "evaluation"):
        status.update(state="running", stage=stage)
        save()
        pending = []
        for job in (j for j in jobs if j["stage"] == stage):
            if status["jobs"].get(job["name"], {}).get("state") == "complete":
                if not Path(job["artifact"]).is_file():
                    raise FileNotFoundError(job["artifact"])
            else:
                pending.append(job)
        active = []
        with ExitStack() as stack:
            try:
                while pending or active:
                    occupied = {g for job, child in active for g in job["gpus"]}
                    for job in list(pending):
                        if occupied.intersection(job["gpus"]):
                            continue
                        command = list(job["command"])
                        if stage == "training" and (Path(job["artifact"]).parent / "latest.pt").exists():
                            command.append("--resume")
                        stream = stack.enter_context((run / f"{job['name']}.log").open("a"))
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=",".join(job["gpus"]),
                                   OMP_NUM_THREADS=str(threads), PYTHONUNBUFFERED="1")
                        child = subprocess.Popen(command, cwd=repo, env=env, stdout=stream,
                                                 stderr=subprocess.STDOUT, start_new_session=True,
                                                 pass_fds=(lock.fileno(),))
                        active.append((job, child))
                        pending.remove(job)
                        occupied.update(job["gpus"])
                        status["jobs"][job["name"]] = dict(state="running", pid=child.pid, started_at=now())
                        save()
                    for job, child in list(active):
                        code = child.poll()
                        if code is None:
                            continue
                        if code or not Path(job["artifact"]).is_file():
                            raise RuntimeError(f"{job['name']} failed: exit={code}; see its log")
                        status["jobs"][job["name"]].update(state="complete", completed_at=now())
                        active.remove((job, child))
                        save()
                    if active:
                        time.sleep(1)
            except BaseException:
                for job, child in active:
                    if child.poll() is None:
                        os.killpg(child.pid, signal.SIGTERM)
                        try:
                            child.wait(timeout=15)
                        except subprocess.TimeoutExpired:
                            os.killpg(child.pid, signal.SIGKILL)
                            child.wait()
                    status["jobs"][job["name"]].update(state="interrupted", returncode=child.returncode)
                raise


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default="/experiments/loop_drafter")
    parser.add_argument("--protocol", default=str(Path(__file__).with_name("data_protocol_v2.json")))
    parser.add_argument("--run-name", help="default: the registered recipe's run_name")
    parser.add_argument("--gpus", default="0,1,2,3")
    parser.add_argument("--memory-fraction", type=float, default=.18)
    parser.add_argument("--num-threads", type=int, default=8)
    parser.add_argument("--execute", action="store_true", help="start only after all effective trajectories and index are complete")
    parser.add_argument("--train-only", action="store_true", help="finish training without automatically running evaluation")
    args = parser.parse_args(argv)
    gpus = args.gpus.split(",")
    if len(gpus) not in (1, 2, 4, 8) or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError("select 1, 2, 4 or 8 distinct GPU indices")
    if not 0 < args.memory_fraction <= 1 or args.num_threads < 1:
        raise ValueError("invalid runtime configuration")
    recipe = json.loads(Path(args.protocol).read_text())
    run = Path(args.root).resolve() / (args.run_name or recipe["run_name"])
    run.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    with (run / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        comparison, protocol, suites, preparation = prepare_protocol(args.protocol, run, repo)
        status_path = run / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        jobs = job_specs(run, comparison, suites, gpus, args.memory_fraction, args.num_threads,
                         allow_repartition=not previous.get("jobs"))
        execution = dict(gpus=gpus, memory_fraction=args.memory_fraction,
                         num_threads=args.num_threads, jobs=jobs)
        if previous.get("jobs"):
            freeze_json(run / "execution.json", execution)
        else:
            atomic_json(run / "execution.json", execution)
        state = preparation["state"]
        if not previous.get("jobs"):
            atomic_json(status_path, dict(state=state, training_started=False, prepared_at=now(),
                        comparison_id=protocol["comparison_id"] if state == "prepared" else None,
                        missing_inputs=preparation["missing_inputs"],
                        observed_trajectory_requests=preparation["observed_trajectory_requests"],
                        expected_trajectory_requests=preparation["expected_trajectory_requests"], jobs={}))
        if state != "prepared" or not args.execute:
            print(json.dumps(dict(state=previous.get("state", state) if previous.get("jobs") else state,
                                  run=str(run), comparison_id=protocol["comparison_id"] if state == "prepared" else None,
                                  source_counts=comparison["eval_source_counts"], missing_inputs=preparation["missing_inputs"],
                                  training_started=bool(previous.get("training_started", False)))))
            if args.execute and state != "prepared":
                raise RuntimeError("awaiting_trajectories: complete the effective manifest, data_index.json and sibling config.json before --execute")
            return
        status = previous | dict(state="running", pid=os.getpid(), jobs=previous.get("jobs", {}),
                                 comparison_id=protocol["comparison_id"],
                                 training_started=True, started_at=previous.get("started_at", now()))
        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"launcher received signal {signum}")
        signal.signal(signal.SIGTERM, interrupted)
        try:
            selected_jobs = [job for job in jobs if job["stage"] == "training"] if args.train_only else jobs
            execute(selected_jobs, run, repo, status, lock, args.num_threads)
            if args.train_only:
                status.update(state="training_complete", stage="training", training_completed_at=now())
            else:
                for name, prompts in suites.items():
                    merge_evaluation(run / "evaluation" / name, prompts, gpus)
                status.update(state="complete", completed_at=now(), evaluation_summaries={
                    name: str(run / "evaluation" / name / "summary.json") for name in suites})
        except BaseException as error:
            status.update(state="failed", error=repr(error), failed_at=now())
            raise
        finally:
            atomic_json(status_path, status)


if __name__ == "__main__":
    main()
