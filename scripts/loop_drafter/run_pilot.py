"""Run the bounded loop-drafter pilot sequentially on one selected GPU.

Each stage has its own log. A nonzero exit stops dependent work and records the
failure; capture is resumable and training resumes from its latest checkpoint.
"""

import argparse
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import subprocess
import sys


def now():
    return datetime.now(timezone.utc).isoformat()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", required=True)
    parser.add_argument("--root", required=True)
    parser.add_argument("--run-name", default="pilot_100")
    parser.add_argument("--steps", type=int, default=100)
    parser.add_argument("--per-split-limit", type=int, default=0)
    parser.add_argument("--capture-tokens", type=int, default=96)
    parser.add_argument("--eval-limit", type=int, default=16)
    parser.add_argument("--eval-tokens", type=int, default=64)
    parser.add_argument("--memory-fraction", type=float, default=.18)
    args = parser.parse_args()
    if min(args.steps, args.eval_limit, args.eval_tokens) < 1 or args.per_split_limit < 0:
        raise ValueError("invalid pilot sizes")
    root = Path(args.root).resolve()
    run = root / args.run_name
    run.mkdir(parents=True, exist_ok=True)
    # Lock the shared capture root as well as the run, preventing two GPU jobs
    # from accidentally writing the same trajectory manifest.
    lock = (root / "pilot.lock").open("a")
    fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    repo = Path(__file__).resolve().parents[2]
    command_prefix = [sys.executable, "-m", "scripts.loop_drafter."]
    status = {"state": "running", "started_at": now(), "pid": os.getpid(),
              "config": vars(args), "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
              "python": sys.executable, "completed_stages": []}

    def write_status():
        status["updated_at"] = now()
        temporary = run / "status.json.tmp"
        temporary.write_text(json.dumps(status, indent=2) + "\n")
        temporary.replace(run / "status.json")

    def stage(name, module, arguments):
        command = command_prefix[:2] + [command_prefix[2] + module] + [str(v) for v in arguments]
        status.update(stage=name, command=command, stage_started_at=now())
        write_status()
        print(json.dumps({"stage": name, "started_at": status["stage_started_at"]}), flush=True)
        with (run / f"{name}.log").open("a") as log:
            log.write("\n" + json.dumps({"started_at": now(), "command": command}) + "\n")
            log.flush()
            child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
            status["child_pid"] = child.pid
            write_status()
            code = child.wait()
        status.pop("child_pid", None)
        if code:
            status.update(state="failed", returncode=code, failed_at=now())
            write_status()
            raise SystemExit(code)
        status["completed_stages"].append(name)
        write_status()

    stage("target_validation", "target", ["--model", args.model, "--device", "cuda:0",
          "--dtype", "float32", "--attention-backend", "sdpa", "--memory-limit-gib", 17,
          "--output", run / "target_validation.json", "--new-tokens", 8])
    capture = root / "capture_fp32"
    common = ["--device", "cuda:0", "--memory-fraction", args.memory_fraction]
    stage("capture", "capture", ["--model", args.model, "--prompts",
          root / "prompts/pilot_train.jsonl", root / "prompts/pilot_eval.jsonl",
          "--out", capture, "--max-new-tokens", args.capture_tokens,
          "--target-dtype", "float32", "--per-split-limit", args.per_split_limit] + common)
    # Freeze the selected trajectories for this run. A smoke run after a full
    # capture must still use only its requested number of records per split.
    rows = [json.loads(line) for line in (capture / "manifest.jsonl").read_text().splitlines() if line.strip()]
    selected, split_counts = [], {}
    for row in rows:
        split = row["split"]
        if args.per_split_limit and split_counts.get(split, 0) >= args.per_split_limit:
            continue
        selected.append(row)
        split_counts[split] = split_counts.get(split, 0) + 1
    manifest = run / "manifest.jsonl"
    manifest.write_text("".join(json.dumps(row) + "\n" for row in selected))
    status["data_records"] = split_counts
    write_status()
    checkpoints = []
    for mode in ("final", "multiloop", "trajectory"):
        output = run / mode
        options = ["--data", manifest, "--frozen-io", capture / "frozen_io.pt",
                   "--out", output, "--feature-mode", mode, "--max-steps", args.steps,
                   "--epochs", max(4, args.steps), "--max-anchors-per-request", 32,
                   "--eval-max-anchors", 32, "--batch-size", 16,
                   "--eval-every", 50, "--save-every", 50] + common
        if (output / "latest.pt").exists():
            options.append("--resume")
        stage(f"train_{mode}", "train", options)
        checkpoints.append(output / "latest.pt")
    stage("evaluate", "evaluate", ["--model", args.model, "--manifest", manifest,
          "--frozen-io", capture / "frozen_io.pt", "--checkpoints", *checkpoints,
          "--out", run / "evaluation", "--target-dtype", "float32",
          "--max-new-tokens", args.eval_tokens, "--limit", args.eval_limit] + common)
    status.update(state="complete", finished_at=now())
    write_status()
    print(json.dumps(status), flush=True)


if __name__ == "__main__":
    main()
