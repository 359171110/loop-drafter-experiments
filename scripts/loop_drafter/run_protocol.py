"""Sequential capture, four-epoch baseline, equal-budget branches, speed panel."""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys

from .evaluate_protocol import atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-name", default="run_seed0")
    parser.add_argument("--expected-train-requests", type=int, default=8192)
    parser.add_argument("--eval-per-source", type=int, default=300)
    parser.add_argument("--baseline-epochs", type=int, default=4)
    parser.add_argument("--branch-epochs", type=int, default=2)
    parser.add_argument("--requests-per-update", type=int, default=8)
    parser.add_argument("--max-anchors-per-request", type=int, default=64)
    parser.add_argument("--microbatch", type=int, default=16)
    parser.add_argument("--capture-batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=256)
    parser.add_argument("--memory-fraction", type=float, default=.18)
    args = parser.parse_args()
    if min(args.expected_train_requests, args.eval_per_source, args.baseline_epochs,
           args.branch_epochs, args.requests_per_update, args.max_anchors_per_request,
           args.microbatch, args.capture_batch_size) < 1 or args.max_new_tokens < 2:
        raise ValueError("invalid protocol sizes")
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    lock = (root / "protocol.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    run = root / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    repo = Path(__file__).resolve().parents[2]
    config = vars(args) | {"seed": 0, "target_dtype": "float32", "source_layout": "canonical",
                           "loss": "uniform_ce", "checkpoint_selection": "fixed_final_epoch",
                           "development_panel": None, "additional_seeds": []}
    config_path = run / "protocol.json"
    if config_path.exists() and json.loads(config_path.read_text()) != config:
        raise ValueError("protocol settings changed; use a new output root")
    atomic_json(config_path, config)
    status_path = run / "status.json"
    previous = json.loads(status_path.read_text()) if status_path.exists() else {}
    completed = list(previous.get("completed_stages", []))
    status = {"state": "running", "started_at": previous.get("started_at", now()),
              "resumed_at": now() if previous else None, "pid": os.getpid(),
              "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "python": sys.executable, "completed_stages": completed, "config": config}

    def save_status():
        status["updated_at"] = now()
        atomic_json(status_path, status)

    def stage(name, module, arguments, artifact):
        if name in completed:
            if not Path(artifact).exists():
                raise FileNotFoundError(f"completed stage {name} is missing {artifact}")
            return
        command = [sys.executable, "-m", f"scripts.loop_drafter.{module}"] + [str(a) for a in arguments]
        status.update(stage=name, stage_started_at=now(), command=command, stage_log=str(run / f"{name}.log"))
        save_status()
        print(json.dumps({"stage": name, "started_at": status["stage_started_at"]}), flush=True)
        with (run / f"{name}.log").open("a") as log:
            log.write("\n" + json.dumps({"started_at": now(), "command": command}) + "\n")
            log.flush()
            try:
                # Keep the root lock alive if the parent is interrupted while a
                # child is still writing checkpoints. A second launcher cannot
                # then start a concurrent writer in the same output directory.
                child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT,
                                         pass_fds=(lock.fileno(),))
                status["child_pid"] = child.pid
                save_status()
                result = child.wait()
            except BaseException as error:
                status.update(state="failed", error=repr(error), failed_at=now())
                save_status()
                raise
        status.pop("child_pid", None)
        if result or not Path(artifact).exists():
            status.update(state="failed", returncode=result or 1, failed_at=now())
            save_status()
            raise SystemExit(result or 1)
        completed.append(name)
        save_status()

    capture = root / "capture_fp32"
    common = ["--device", "cuda:0", "--memory-fraction", args.memory_fraction, "--num-threads", 8]
    stage("capture", "capture_protocol", ["--model", args.model,
          "--prompts", root / "prompts/train.jsonl", "--reserves", root / "prompts/reserves.jsonl",
          "--out", capture, "--batch-size", args.capture_batch_size,
          "--max-new-tokens", args.max_new_tokens] + common, capture / "capture_summary.json")
    train_common = ["--data", capture / "manifest.jsonl", "--frozen-io", capture / "frozen_io.pt",
                    "--expected-requests", args.expected_train_requests, "--seed", 0,
                    "--requests-per-update", args.requests_per_update,
                    "--max-anchors-per-request", args.max_anchors_per_request,
                    "--batch-size", args.microbatch] + common

    def train(name, role, mode, epochs, parent=None):
        out = run / name
        options = ["--out", out, "--role", role, "--feature-mode", mode, "--epochs", epochs] + train_common
        if parent:
            options += ["--init-checkpoint", parent]
        if (out / "latest.pt").exists():
            options += ["--resume"]
        stage(name, "train_protocol", options, out / "model.pt")
        return out / "model.pt"

    baseline = train("baseline", "baseline", "final", args.baseline_epochs)
    branches = [train(f"branch_{mode}", "branch", mode, args.branch_epochs, baseline)
                for mode in ("final", "multiloop", "trajectory")]
    stage("speed_evaluation", "evaluate_protocol", ["--model", args.model,
          "--prompts", root / "prompts/eval.jsonl", "--frozen-io", capture / "frozen_io.pt",
          "--checkpoints", *branches, "--out", run / "evaluation",
          "--expected-per-source", args.eval_per_source,
          "--max-new-tokens", args.max_new_tokens, "--seed", 0] + common, run / "evaluation/summary.json")
    status.update(state="complete", finished_at=now())
    save_status()
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
