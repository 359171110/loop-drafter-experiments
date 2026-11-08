"""Train the history/residual pair, then evaluate against matched-budget SSD.

Two independent GPU jobs share the original four-epoch training schedule.
Evaluation shards keep every method for one prompt on the same GPU.
"""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .capture import read_jsonl
from .evaluate_protocol import aggregate, atomic_json
from .run_parallel_protocol import evaluation_panels


METHODS = ("ar", "ssd", "history", "history_residual")


def now():
    return datetime.now(timezone.utc).isoformat()


def merge_evaluation(root, prompts, gpus):
    root = Path(root)
    expected = {(p["id"], method): p["source"] for p in prompts for method in METHODS}
    rows, hardware, by_gpu, seen = [], [], [], set()
    for index, gpu in enumerate(gpus):
        shard = root / "shards" / f"worker{index}"
        panel = read_jsonl(root / "panels" / f"worker{index}.jsonl")
        records = read_jsonl(shard / "requests.jsonl")
        summary = json.loads((shard / "summary.json").read_text())
        shard_keys = {(p["id"], m) for p in panel for m in METHODS}
        if {(r["id"], r["method"]) for r in records} != shard_keys:
            raise ValueError(f"incomplete evaluation shard {index}")
        if summary["hardware"]["cuda_visible_devices"] != gpu:
            raise ValueError(f"evaluation shard {index} GPU changed")
        for record in records:
            key = record["id"], record["method"]
            if key in seen or expected.get(key) != record["source"]:
                raise ValueError(f"duplicate/foreign evaluation record {key}")
            seen.add(key)
            rows.append(record | {"worker_index": index, "physical_gpu": gpu})
        hardware.append(summary["hardware"])
        by_gpu.append({"physical_gpu": gpu, "results": aggregate(records)})
    if seen != set(expected):
        raise ValueError("evaluation does not cover the complete panel")
    temporary = root / "requests.jsonl.tmp"
    temporary.write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
    temporary.replace(root / "requests.jsonl")
    result = {"source_counts": dict(Counter(p["source"] for p in prompts)),
              "results": aggregate(rows), "results_by_gpu": by_gpu, "hardware": hardware,
              "metric_notes": {
                  "speedup": "paired summed AR latency / summed drafter latency; same prompt, same GPU",
                  "throughput": "sum tokens / sum request seconds; not multi-GPU serving throughput",
                  "checkpoint_selection": "fixed final epoch, no evaluation-based selection",
                  "comparison": "matched training exposures, not matched parameter counts",
                  "timing_scope": "shared_gpu_interleaved_diagnostic"}}
    atomic_json(root / "summary.json", result)
    return result


def parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", default="/experiments/loop_drafter")
    p.add_argument("--run-name", default="history_ablation_seed0")
    p.add_argument("--gpus", default="0,1,2,3")
    p.add_argument("--memory-fraction", type=float, default=.18)
    p.add_argument("--num-threads", type=int, default=8)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    gpus = args.gpus.split(",")
    if len(gpus) not in (2, 4) or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
        raise ValueError("provide two or four distinct GPU indices")
    training_groups = [gpus[0::2], gpus[1::2]]
    if not 0 < args.memory_fraction <= 1 or args.num_threads < 1:
        raise ValueError("invalid runtime settings")
    root = Path(args.root).resolve()
    run = root / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    capture = root / "capture_fp32"
    baseline = root / "run_seed0" / "baseline"
    data = capture / "manifest.jsonl"
    frozen = capture / "frozen_io.pt"
    for path in (data, frozen, baseline / "model.pt", baseline / "data_index.json",
                 baseline / "config.json", baseline / "status.json", root / "prompts" / "eval.jsonl"):
        if not path.is_file():
            raise FileNotFoundError(path)
    base_config = json.loads((baseline / "config.json").read_text())
    base_status = json.loads((baseline / "status.json").read_text())
    settings = base_config["settings"]
    required = {"epochs": 4, "seed": 0, "requests_per_update": 8,
                "max_anchors_per_request": 64, "microbatch": 16,
                "peak_lr": 1e-4, "floor_lr": 1e-5, "warmup_fraction": .05,
                "weight_decay": .01, "beta1": .9, "beta2": .95, "grad_clip": 1.}
    if any(settings.get(k) != v for k, v in required.items()):
        raise ValueError("existing SSD does not match the intended training budget")
    if base_status.get("event") != "complete" or base_status.get("epoch") != 4:
        raise ValueError("SSD four-epoch baseline is incomplete")
    with (run / "launcher.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        protocol = {
            "experiment": "h4_history_transformer_residual_ablation_v1",
            "data": str(data), "frozen_io": str(frozen), "expected_requests": 8192,
            "ssd_checkpoint": str(baseline / "model.pt"),
            "training": required | {"sampling_epoch_offset": 0, "sampling_epochs": [0, 1, 2, 3]},
            "history_model": {"width": 512, "mlp_width": 1536, "heads": 8,
                              "layers": 2, "window": 64, "depth": 5, "feature_round": 4},
            "variants": ["ssd", "history", "history_residual"],
            "history_pair_initialization": "same seed and identical trainable weights; residual flag only",
            "loss": "uniform CE over valid child labels; no future child tokens in Transformer inputs",
            "comparison": "matched training exposures and optimizer settings; different parameter counts",
            "ssd_scope": "existing 4-epoch baseline, not the 6-epoch continuation",
            "eval_prompts": str(root / "prompts" / "eval.jsonl"), "eval_per_source": 300,
            "eval_seed": 0, "checkpoint_selection": "fixed final epoch",
            "max_prompt_tokens": 512, "max_new_tokens": 256,
            "target_dtype": "float32", "target_rounds": 4, "feature_dtype": "bfloat16",
            "development_panel": None, "additional_seeds": [],
            "implementation": "LAD-inspired history decoder; not a full LoopAhead/DFlash reproduction"}
        config_path = run / "protocol.json"
        if config_path.exists() and json.loads(config_path.read_text()) != protocol:
            raise ValueError("experiment protocol changed on resume")
        atomic_json(config_path, protocol)
        execution = {"gpus": gpus, "training_gpus": dict(zip(("history", "history_residual"), training_groups)),
                     "memory_fraction": args.memory_fraction, "num_threads": args.num_threads,
                     "evaluation": "source-balanced shards; all methods for one prompt on one GPU"}
        execution_path = run / "execution.json"
        if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
            raise ValueError("GPU placement or runtime settings changed on resume")
        atomic_json(execution_path, execution)
        snapshot = run / "source_snapshot"
        if not snapshot.exists():
            snapshot.mkdir()
            for source in Path(__file__).parent.glob("*.py"):
                shutil.copy2(source, snapshot / source.name)
        status_path = run / "status.json"
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        jobs = dict(previous.get("jobs", {}))
        status = {"state": "running", "pid": os.getpid(), "gpus": gpus,
                  "started_at": previous.get("started_at", now()), "resumed_at": now(),
                  "jobs": jobs, "completed_stages": previous.get("completed_stages", [])}

        def save():
            status["updated_at"] = now()
            atomic_json(status_path, status)

        def spec(name, module, options, gpu, artifact):
            gpu_ids = gpu.split(",")
            command = [sys.executable, "-m"]
            if len(gpu_ids) > 1:
                command += ["torch.distributed.run", "--standalone", f"--nproc_per_node={len(gpu_ids)}", "-m"]
            command += [f"scripts.loop_drafter.{module}", *map(str, options)]
            return {"name": name, "command": command, "gpu": gpu, "artifact": str(artifact)}

        def pool(stage, specs):
            status["stage"] = stage
            save()
            active = []
            with ExitStack() as stack:
                try:
                    for item in specs:
                        name = item["name"]
                        if jobs.get(name, {}).get("state") == "complete":
                            if not Path(item["artifact"]).is_file():
                                raise FileNotFoundError(item["artifact"])
                            continue
                        log_path = run / f"{name}.log"
                        stream = stack.enter_context(log_path.open("a"))
                        stream.write("\n" + json.dumps({"started_at": now(), **item}) + "\n")
                        stream.flush()
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=item["gpu"],
                                   OMP_NUM_THREADS=str(args.num_threads), PYTHONUNBUFFERED="1")
                        child = subprocess.Popen(item["command"], cwd=repo, env=env,
                                                 stdout=stream, stderr=subprocess.STDOUT,
                                                 start_new_session=True, pass_fds=(lock.fileno(),))
                        jobs[name] = item | {"state": "running", "pid": child.pid,
                                             "log": str(log_path), "started_at": now()}
                        active.append((item, child))
                        save()
                    last_save = 0.
                    while active:
                        for item, child in list(active):
                            code = child.poll()
                            if code is None:
                                continue
                            if code or not Path(item["artifact"]).is_file():
                                jobs[item["name"]].update(state="failed", returncode=code or 1)
                                raise RuntimeError(f"{item['name']} failed; see {jobs[item['name']]['log']}")
                            jobs[item["name"]].update(state="complete", finished_at=now())
                            active.remove((item, child))
                            save()
                        if time.monotonic() - last_save > 15:
                            status["progress"] = {}
                            for item, _ in active:
                                directory = Path(item["artifact"]).parent
                                progress_path = directory / "progress.json"
                                if progress_path.exists():
                                    status["progress"][item["name"]] = json.loads(progress_path.read_text())
                                elif (directory / "metrics.jsonl").exists():
                                    with (directory / "metrics.jsonl").open("rb") as metric:
                                        metric.seek(0, 2)
                                        size = metric.tell()
                                        metric.seek(max(0, size - 32768))
                                        lines = metric.read().splitlines()
                                    for line in reversed(lines):
                                        try:
                                            row = json.loads(line)
                                        except (ValueError, UnicodeDecodeError):
                                            continue
                                        status["progress"][item["name"]] = {
                                            k: row[k] for k in ("step", "epoch", "loss", "update_seconds") if k in row}
                                        break
                            save()
                            last_save = time.monotonic()
                        if active:
                            time.sleep(1)
                except BaseException:
                    for item, child in active:
                        if child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            jobs[item["name"]].update(state="interrupted", returncode=child.returncode)
                    raise
            if stage not in status["completed_stages"]:
                status["completed_stages"].append(stage)
            save()

        def interrupted(signum, frame):
            raise KeyboardInterrupt(f"launcher received signal {signum}")

        signal.signal(signal.SIGTERM, interrupted)
        common = ["--device", "cuda:0", "--memory-fraction", args.memory_fraction,
                  "--num-threads", args.num_threads]
        try:
            train_jobs = []
            for mode, gpu_group in zip(("history", "history_residual"), training_groups):
                out = run / mode
                options = ["--data", data, "--index-file", baseline / "data_index.json",
                           "--frozen-io", frozen, "--out", out, "--feature-mode", mode,
                           "--epochs", 4, "--expected-requests", 8192, "--seed", 0,
                           "--requests-per-update", 8, "--max-anchors-per-request", 64,
                           "--microbatch", 16, "--width", 512, "--mlp-width", 1536,
                           "--heads", 8, "--layers", 2, "--window", 64, "--depth", 5,
                           "--sampling-epoch-offset", 0, *common]
                if (out / "latest.pt").exists():
                    options.append("--resume")
                train_jobs.append(spec(mode, "train_history", options, ",".join(gpu_group), out / "model.pt"))
            pool("training", train_jobs)
            evaluation = run / "evaluation"
            panels_dir = evaluation / "panels"
            panels_dir.mkdir(parents=True, exist_ok=True)
            prompts = read_jsonl(root / "prompts" / "eval.jsonl")
            counts = Counter(p["source"] for p in prompts)
            if len(prompts) != 900 or len(counts) != 3 or set(counts.values()) != {300}:
                raise ValueError("expected original 900-prompt evaluation panel")
            panels = evaluation_panels(prompts, len(gpus), per_source=300, seed=0)
            eval_jobs = []
            for index, (panel, gpu) in enumerate(zip(panels, gpus)):
                panel_path = panels_dir / f"worker{index}.jsonl"
                if panel_path.exists() and read_jsonl(panel_path) != panel:
                    raise ValueError("evaluation shard panel changed on resume")
                panel_path.write_text("".join(json.dumps(p, ensure_ascii=False) + "\n" for p in panel))
                out = evaluation / "shards" / f"worker{index}"
                options = ["--model", "/models/Ouro-1.4B",
                           "--prompts", panel_path, "--frozen-io", frozen, "--out", out,
                           "--ssd-checkpoint", baseline / "model.pt",
                           "--history-checkpoint", run / "history" / "model.pt",
                           "--residual-checkpoint", run / "history_residual" / "model.pt",
                           "--expected-per-source", 300 // len(gpus), "--max-new-tokens", 256,
                           "--max-prompt-tokens", 512, "--seed", 0, *common]
                eval_jobs.append(spec(f"evaluation{index}", "evaluate_history", options, gpu, out / "summary.json"))
            pool("evaluation", eval_jobs)
            merge_evaluation(evaluation, prompts, gpus)
            status.update(state="complete", stage="complete", completed_at=now(),
                          summary=str(evaluation / "summary.json"))
            save()
        except BaseException as error:
            status.update(state="failed", error=repr(error), failed_at=now())
            save()
            raise


if __name__ == "__main__":
    main()
