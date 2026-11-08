import json
from collections import Counter
from pathlib import Path
from unittest.mock import patch

import pytest

from . import run_loop_memory_experiment as runner
from .run_loop_memory_experiment import METHODS, MODES, job_specs, merge_evaluation, panels


def test_unequal_benchmark_sizes_are_partitioned_without_replacement():
    prompts = [{"id": f"{source}:{i}", "source": source}
               for source, count in (("gsm8k", 300), ("humaneval", 164), ("chat", 300))
               for i in range(count)]
    shards = panels(prompts, 8)
    assert sorted(row["id"] for shard in shards for row in shard) == sorted(row["id"] for row in prompts)
    counts = [Counter(row["source"] for row in shard) for shard in shards]
    for source in counts[0]:
        assert max(c[source] for c in counts) - min(c[source] for c in counts) <= 1
    assert panels(prompts, 8) == shards


def comparison_fixture():
    from .loop_memory_draft import LoopMemoryConfig
    recipe = json.loads(Path(runner.__file__).with_name("data_protocol_v2.json").read_text())
    return {"inputs": {key: {"path": f"/{key}"} for key in
            ("data", "frozen_io", "data_index", "core_sources", "reasoning_extension_sources")},
            "target_model": "/ouro", "python": "/runtime/python", "max_prompt_tokens": 2048,
            "max_new_tokens": 256, "training": runner.training_settings(recipe),
            "expected_requests": 16384, "architecture": LoopMemoryConfig().dict()}


def suites_fixture():
    return {name: [{"id": f"{name}:{i}", "source": name} for i in range(16)]
            for name in ("core", "reasoning_extension")}


def test_eight_gpu_commands_preserve_scratch_budget_and_separate_panels(tmp_path):
    jobs = job_specs(tmp_path, comparison_fixture(), suites_fixture(), list(map(str, range(8))), .18, 8)
    training = jobs[:4]
    assert [j["name"] for j in training] == list(MODES)
    assert [j["gpus"] for j in training] == [["0", "4"], ["1", "5"], ["2", "6"], ["3", "7"]]
    for job in training:
        cmd = job["command"]
        assert cmd[0] == "/runtime/python"
        assert "--nproc_per_node=2" in cmd and "--from-scratch" in cmd
        assert "--init-checkpoint" not in cmd
        assert cmd[cmd.index("--sampling-epoch-offset") + 1] == "0"
        assert cmd[cmd.index("--epochs") + 1] == "4"
        assert cmd[cmd.index("--expected-requests") + 1] == "16384"
    assert len(jobs[4:]) == 16
    for name in suites_fixture():
        selected = [j for j in jobs if j.get("panel") == name]
        assert len(selected) == 8
        assert all(f"evaluation/{name}/shards/" in j["artifact"] for j in selected)
        for job in selected:
            command = job["command"]
            path = command[command.index("--prompts") + 1]
            assert all(row["id"].startswith(name + ":") for row in runner.read_jsonl(path))
            assert command[command.index("--tokenizer-manifest") + 1] == f"/{name}_sources"


def test_only_unstarted_preparation_can_change_worker_count(tmp_path):
    comparison, suites = comparison_fixture(), suites_fixture()
    job_specs(tmp_path, comparison, suites, ["0", "1"], .18, 8)
    with pytest.raises(ValueError, match="shard changed"):
        job_specs(tmp_path, comparison, suites, ["0", "1", "2", "3"], .18, 8)
    jobs = job_specs(tmp_path, comparison, suites, ["0", "1", "2", "3"], .18, 8,
                     allow_repartition=True)
    assert len(jobs) == 12
    for job in jobs[4:]:
        path = job["command"][job["command"].index("--prompts") + 1]
        assert len(Path(path).read_text().splitlines()) == 4


def preparation_fixture(tmp_path, ready=False):
    from .loop_memory_draft import LoopMemoryConfig
    recipe = json.loads(Path(runner.__file__).with_name("data_protocol_v2.json").read_text())
    recipe["run_name"] = "scratch_test"
    recipe["training_data"]["effective_requests"] = 8
    recipe["training"]["updates_per_variant"] = 4
    recipe["training"]["request_exposures_per_variant"] = 32
    recipe["training"]["frozen_io"] = str(tmp_path / "frozen_io.pt")
    Path(recipe["training"]["frozen_io"]).write_bytes(b"target IO, not an old drafter")
    target = tmp_path / "target"
    target.mkdir()
    (target / "config.json").write_text("{}")
    recipe["capture"]["target"] = str(target)
    prompt_path = tmp_path / "prompts" / "train.jsonl"
    prompt_path.parent.mkdir()
    prompt_path.write_text("".join(json.dumps({"id": str(i), "source": "train", "input_ids": [1, 2]}) + "\n" for i in range(8)))
    recipe["training_data"]["prompt_manifest"] = str(prompt_path)
    capture = tmp_path / "capture"
    capture.mkdir()
    recipe["training_data"]["trajectory_manifest"] = str(capture / "manifest.jsonl")
    for name in ("core", "reasoning_extension"):
        panel = tmp_path / name
        panel.mkdir()
        rows = [{"id": f"{name}:{i}", "source": name, "split": "eval", "input_ids": [1, 2]}
                for i in range(8)]
        (panel / "prompts.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
        (panel / "sources.json").write_text(json.dumps({"expected_counts": {name: 8}}))
        recipe["evaluation"][name].update(manifest=str(panel / "prompts.jsonl"), requests=8)
    repo = tmp_path / "repo"
    for directory in ("loop_drafter", "fest_hybrid", "fest_hybrid_prefix_tree"):
        (repo / "scripts" / directory).mkdir(parents=True)
        (repo / "scripts" / directory / "model.py").write_text("# frozen source\n")
    if ready:
        entries = []
        for i in range(8):
            path = capture / f"{i}.pt"
            path.write_bytes(b"not loaded: cached validated shapes")
            stat = path.stat()
            entries.append(dict(path=str(path), id=str(i), source="train", prompt_length=2,
                                token_count=9, available_anchors=6, file_size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        (capture / "manifest.jsonl").write_text("".join(json.dumps({k: entry[k] for k in
                   ("path", "id", "source", "prompt_length")} | {"split": "train", "generated_length": 7}) + "\n"
                   for entry in entries))
        (capture / "data_index.json").write_text(json.dumps(entries))
        (capture / "config.json").write_text(json.dumps({"config": LoopMemoryConfig().dict()}))
    path = tmp_path / "recipe.json"
    path.write_text(json.dumps(recipe))
    run = tmp_path / recipe["run_name"]
    run.mkdir()
    return path, run, repo


def test_missing_capture_prepares_commands_and_execute_never_launches(tmp_path):
    recipe, run, repo = preparation_fixture(tmp_path)
    with patch.object(runner.subprocess, "Popen", side_effect=AssertionError("must not launch")):
        runner.main(["--protocol", str(recipe), "--root", str(tmp_path), "--gpus", "0,1"])
        status = json.loads((run / "status.json").read_text())
        assert status["state"] == "awaiting_trajectories" and not status["training_started"]
        assert {row["name"] for row in status["missing_inputs"]} == {"data", "data_index", "index_config"}
        assert not (run / "comparison_protocol.json").exists()
        jobs = json.loads((run / "execution.json").read_text())["jobs"]
        assert all("--init-checkpoint" not in job["command"] for job in jobs)
        assert len(jobs) == 8
        with pytest.raises(RuntimeError, match="awaiting_trajectories"):
            runner.main(["--protocol", str(recipe), "--root", str(tmp_path), "--gpus", "0,1", "--execute"])


def test_complete_index_freezes_four_epochs_and_rejects_changes(tmp_path):
    recipe, run, repo = preparation_fixture(tmp_path, ready=True)
    comparison, protocol, suites, preparation = runner.prepare_protocol(recipe, run, repo)
    assert preparation["state"] == "prepared"
    assert comparison["parent_checkpoint"] is None and protocol["initialization"] == "from_scratch"
    assert [r["sampling_epoch"] for r in comparison["sampling"]] == [0, 1, 2, 3]
    assert sum(r["requests"] for r in comparison["sampling"]) == 32
    assert sum(r["updates"] for r in comparison["sampling"]) == 4
    assert all(r["anchors"] == 48 and r["valid_labels"] == 160 for r in comparison["sampling"])
    assert set(suites) == {"core", "reasoning_extension"}
    assert runner.prepare_protocol(recipe, run, repo)[1] == protocol
    io_path = json.loads(recipe.read_text())["training"]["frozen_io"]
    Path(io_path).write_bytes(b"changed")
    with pytest.raises(ValueError, match="frozen input changed"):
        runner.prepare_protocol(recipe, run, repo)


def test_legacy_recipe_is_rejected_before_touching_parent(tmp_path):
    recipe, run, repo = preparation_fixture(tmp_path)
    data = json.loads(recipe.read_text())
    data["training"].update(initialization="warmstart", parent_checkpoint="/absent/old/model.pt")
    recipe.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="only supports from_scratch"):
        runner.prepare_protocol(recipe, run, repo)


def test_execution_stages_resume_and_failure_stop_evaluation(tmp_path):
    comparison, suites = comparison_fixture(), suites_fixture()
    jobs = job_specs(tmp_path, comparison, suites, ["0", "1", "2", "3"], .18, 8)
    status = {"jobs": {}}
    calls = []
    class Child:
        pid = 123
        def __init__(self, command, **kwargs):
            assert "pass_fds" in kwargs
            self.command, self.returncode = command, 0
            calls.append(command)
            out = Path(command[command.index("--out") + 1])
            out.mkdir(parents=True, exist_ok=True)
            (out / ("model.pt" if "--from-scratch" in command else "summary.json")).write_text("{}")
        def poll(self):
            return self.returncode
    with (tmp_path / "lock").open("a") as lock, patch.object(runner.subprocess, "Popen", Child):
        runner.execute(jobs, tmp_path, tmp_path, status, lock, 1)
        assert all("--from-scratch" in cmd for cmd in calls[:4])
        assert all("--from-scratch" not in cmd for cmd in calls[4:])
        runner.execute(jobs, tmp_path, tmp_path, status, lock, 1)
        assert len(calls) == len(jobs)
    status = {"jobs": {}}
    calls.clear()
    class Failed(Child):
        def __init__(self, command, **kwargs):
            super().__init__(command, **kwargs)
            self.returncode = 1
    with (tmp_path / "lock").open("a") as lock, patch.object(runner.subprocess, "Popen", Failed):
        with pytest.raises(RuntimeError, match="failed"):
            runner.execute(jobs, tmp_path, tmp_path, status, lock, 1)
    assert calls and all("--from-scratch" in cmd for cmd in calls)


def make_shards(root):
    prompts = [{"id": "p0", "source": "gsm8k"}, {"id": "p1", "source": "humaneval"}]
    (root / "panels").mkdir()
    for i, prompt in enumerate(prompts):
        (root / "panels" / f"worker{i}.jsonl").write_text(json.dumps(prompt) + "\n")
        shard = root / "shards" / f"worker{i}"
        shard.mkdir(parents=True)
        records = [prompt | {"method": method, "tokens": [1, 2], "prefill_seconds": .1,
                             "decode_seconds": (i + 1) * (2 if method == "ar" else 1),
                             "proposal_seconds": 0 if method == "ar" else .1,
                             "verify_seconds": 0 if method == "ar" else .8,
                             "memory_update_seconds": 0 if method in ("ar", "none") else .05,
                             "rounds": [] if method == "ar" else [
                                 {"committed_tokens": 2, "accepted_children": 1}]}
                   for method in METHODS]
        (shard / "requests.jsonl").write_text("".join(json.dumps(r) + "\n" for r in records))
        (shard / "summary.json").write_text(json.dumps({"hardware": {"cuda_visible_devices": str(i)}}))
    return prompts


def test_merge_uses_paired_ar_and_rejects_duplicate_results(tmp_path):
    prompts = make_shards(tmp_path)
    result = merge_evaluation(tmp_path, prompts, ["0", "1"])
    row = next(r for r in result["results"] if r["source"] == "__all__" and r["method"] == "trajectory")
    assert row["requests"] == row["exact_matches"] == 2
    assert row["decode_latency_speedup"] == 2
    cost = next(r for r in result["component_costs"] if r["source"] == "__all__" and r["method"] == "trajectory")
    assert cost["proposal_seconds"] == .2
    assert cost["verify_seconds"] == 1.6
    assert cost["memory_update_seconds"] == .1
    assert cost["milliseconds_per_verify"]["memory_update_seconds"] == 50
    path = tmp_path / "shards" / "worker0" / "requests.jsonl"
    with path.open("a") as stream:
        stream.write(path.read_text().splitlines()[0] + "\n")
    with pytest.raises(ValueError, match="duplicate"):
        merge_evaluation(tmp_path, prompts, ["0", "1"])


def test_train_only_does_not_launch_or_merge_evaluation(tmp_path):
    recipe, run, _ = preparation_fixture(tmp_path, ready=True)
    selected = []
    def training(jobs, *args):
        selected.extend(jobs)
        assert len(jobs) == 4 and all(j['stage'] == 'training' for j in jobs)
    with patch.object(runner, 'execute', side_effect=training), \
         patch.object(runner, 'merge_evaluation', side_effect=AssertionError('must not evaluate')):
        runner.main(['--protocol', str(recipe), '--root', str(tmp_path), '--gpus', '0,1', '--execute', '--train-only'])
    assert selected
    assert json.loads((run / 'status.json').read_text())['state'] == 'training_complete'
