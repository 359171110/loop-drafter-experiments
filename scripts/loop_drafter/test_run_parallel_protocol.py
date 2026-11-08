"""CPU-only checks of prompt sharding, paired merging, and subprocess scheduling."""

from collections import Counter
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest
from unittest.mock import patch

from . import run_parallel_protocol as protocol


METHODS = ("ar", "final", "multiloop", "trajectory")
GPUS = ("4", "5", "6", "7")


def prompts(per_source):
    return [dict(id=f"{source}:{i}", source=source, domain=source,
                 split="eval", prompt=f"fake {source} prompt {i}")
            for source in ("math", "code", "chat") for i in range(per_source)]


def write_jsonl(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))


def write_results(out, panel, gpu):
    out.mkdir(parents=True, exist_ok=True)
    rows = [dict(id=entry["id"], source=entry["source"], method=method,
                 tokens=[1, 2, 3, 4, 5], prefill_seconds=1.,
                 decode_seconds=(int(gpu) - 3) * (4. if method == "ar" else 2.),
                 rounds=[])
            for entry in panel for method in METHODS]
    write_jsonl(out / "requests.jsonl", rows)
    (out / "summary.json").write_text(json.dumps({"hardware": {"cuda_visible_devices": gpu}}))
    return rows


class EvaluationPanelsTests(unittest.TestCase):
    def test_actual_300_per_source_panel_is_balanced_unique_and_reproducible(self):
        original = prompts(300)
        before = [row["id"] for row in original]
        panels = protocol.evaluation_panels(original, 4, 300)
        self.assertEqual([len(panel) for panel in panels], [225] * 4)
        for panel in panels:
            self.assertEqual(Counter(row["source"] for row in panel),
                             {"math": 75, "code": 75, "chat": 75})
        combined = [row["id"] for panel in panels for row in panel]
        self.assertEqual(len(set(combined)), 900)
        self.assertCountEqual(combined, before)
        self.assertEqual(panels, protocol.evaluation_panels(original, 4, 300))
        self.assertEqual([row["id"] for row in original], before)

    def test_rejects_duplicate_ids_wrong_counts_and_uneven_panels(self):
        duplicate = prompts(4)
        duplicate[1]["id"] = duplicate[0]["id"]
        for panel, workers, count in ((duplicate, 4, 4), (prompts(4)[:-1], 4, 4),
                                      (prompts(5), 4, 5), ([], 4, 4)):
            with self.subTest(workers=workers, count=count, size=len(panel)):
                with self.assertRaises(ValueError):
                    protocol.evaluation_panels(panel, workers, count)


class MergeEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = prompts(4)
        self.panels = protocol.evaluation_panels(self.prompts, 4, 4)
        for index, (panel, gpu) in enumerate(zip(self.panels, GPUS)):
            write_jsonl(self.root / "panels" / f"worker{index}.jsonl", panel)
            write_results(self.root / "shards" / f"worker{index}", panel, gpu)

    def rows_path(self, index):
        return self.root / "shards" / f"worker{index}" / "requests.jsonl"

    def merge(self):
        return protocol.merge_evaluation(self.root, self.prompts, GPUS)

    def test_complete_merge_preserves_same_gpu_pairs_and_summed_time_rates(self):
        summary = self.merge()
        rows = protocol.read_jsonl(self.root / "requests.jsonl")
        self.assertEqual(len(rows), 48)
        self.assertEqual(len({(row["id"], row["method"]) for row in rows}), 48)
        for prompt in self.prompts:
            matched = [row for row in rows if row["id"] == prompt["id"]]
            self.assertEqual({row["method"] for row in matched}, set(METHODS))
            self.assertEqual(len({row["physical_gpu"] for row in matched}), 1)
        combined = next(row for row in summary["results"]
                        if row["source"] == "__all__" and row["method"] == "final")
        self.assertEqual(combined["paired_requests"], 12)
        self.assertEqual(combined["decode_tokens_per_second"], 48 / 60)
        self.assertEqual(combined["decode_latency_speedup"], 2.)
        self.assertEqual(summary["source_counts"], {"math": 4, "code": 4, "chat": 4})
        self.assertEqual(len(summary["results_by_gpu"]), 4)
        self.assertIn("not aggregate multi-GPU serving throughput", summary["metric_notes"]["throughput"])

    def test_rejects_missing_method(self):
        path = self.rows_path(0)
        write_jsonl(path, protocol.read_jsonl(path)[:-1])
        with self.assertRaisesRegex(ValueError, "incomplete evaluation shard"):
            self.merge()

    def test_rejects_duplicate_record_even_when_set_of_keys_is_complete(self):
        path = self.rows_path(0)
        rows = protocol.read_jsonl(path)
        write_jsonl(path, rows + [rows[0]])
        with self.assertRaisesRegex(ValueError, "duplicate or invalid"):
            self.merge()

    def test_rejects_method_moved_to_another_gpu(self):
        left, right = [protocol.read_jsonl(self.rows_path(i)) for i in (0, 1)]
        left[1], right[1] = right[1], left[1]
        write_jsonl(self.rows_path(0), left)
        write_jsonl(self.rows_path(1), right)
        with self.assertRaisesRegex(ValueError, "incomplete evaluation shard"):
            self.merge()

    def test_rejects_changed_gpu_and_wrong_source(self):
        summary_path = self.root / "shards/worker0/summary.json"
        original = summary_path.read_text()
        summary_path.write_text(json.dumps({"hardware": {"cuda_visible_devices": "0"}}))
        with self.assertRaisesRegex(ValueError, "moved to another GPU"):
            self.merge()
        summary_path.write_text(original)
        path = self.rows_path(0)
        rows = protocol.read_jsonl(path)
        rows[0]["source"] = "wrong-source"
        write_jsonl(path, rows)
        with self.assertRaisesRegex(ValueError, "duplicate or invalid"):
            self.merge()

    def test_rejects_globally_missing_prompt_despite_locally_complete_shards(self):
        removed = self.panels[0][0]["id"]
        write_jsonl(self.root / "panels/worker0.jsonl", self.panels[0][1:])
        path = self.rows_path(0)
        write_jsonl(path, [row for row in protocol.read_jsonl(path) if row["id"] != removed])
        with self.assertRaisesRegex(ValueError, "does not cover the full panel"):
            self.merge()


class FakeProcesses:
    """Simulate file contracts while recording launch-before-poll concurrency."""
    def __init__(self, root):
        self.root = root
        self.calls = []
        self.events = []

    @staticmethod
    def value(command, option):
        return command[command.index(option) + 1]

    def __call__(self, command, **kwargs):
        module = next(part.rsplit(".", 1)[-1] for part in command
                      if part.startswith("scripts.loop_drafter."))
        if module == "capture_parallel":
            name = "capture_" + self.value(command, "--phase")
        else:
            out = Path(self.value(command, "--out"))
            name = out.name if module == "train_protocol" else f"{module}:{out.parent.name}:{out.name}"
        gpu = kwargs["env"]["CUDA_VISIBLE_DEVICES"]
        inherited = [os.fstat(fd) for fd in kwargs["pass_fds"]]
        self.calls.append(dict(name=name, module=module, command=command, gpu=gpu,
                               inherited=[(stat.st_dev, stat.st_ino) for stat in inherited]))
        self.events.append(("launch", name))
        if name == "capture_prepare":
            quotas = {"math": 3072, "code": 3072, "chat": 2048}
            workers = []
            for index in range(4):
                directory = self.root / "capture_parallel" / f"worker{index}"
                directory.mkdir(parents=True)
                workers.append(dict(index=index, prompts=str(directory / "prompts.jsonl"),
                                    reserves=str(directory / "reserves.jsonl"), out=str(directory / "capture"),
                                    quotas={source: count // 4 for source, count in quotas.items()}))
            (self.root / "capture_parallel/plan.json").write_text(json.dumps(
                dict(existing_counts={}, quotas=quotas, workers=workers)))
        elif name == "capture_merge":
            directory = self.root / "capture_fp32"
            directory.mkdir(exist_ok=True)
            (directory / "capture_summary.json").write_text("{}")
            (directory / "manifest.jsonl").write_text("")
            (directory / "frozen_io.pt").write_text("fake frozen target IO")
        elif module == "capture_protocol":
            out.mkdir(parents=True)
            (out / "capture_summary.json").write_text("{}")
            (out / "progress.json").write_text(json.dumps({"source_counts": {"math": 768, "code": 768, "chat": 512}}))
        elif module == "train_protocol":
            out.mkdir(parents=True)
            (out / "model.pt").write_text("fake trained model")
        elif module == "evaluate_protocol":
            panel = protocol.read_jsonl(self.value(command, "--prompts"))
            write_results(out, panel, gpu)
        else:
            raise AssertionError(module)
        owner = self

        class Process:
            pid = 40000 + len(owner.calls)
            returncode = None

            def poll(self):
                owner.events.append(("poll", name))
                self.returncode = 0
                return 0

        return Process()


class OrchestrationTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Only the fake evaluation panel is small; training keeps the real 8192/8 budget.
        write_jsonl(self.root / "prompts/eval.jsonl", prompts(4))
        (self.root / "capture_fp32").mkdir()

    def invoke(self, processes):
        argv = ["run_parallel_protocol", "--model", "/fake/ouro", "--root", str(self.root),
                "--eval-per-source", "4"]
        with patch.object(protocol.sys, "argv", argv), \
             patch.object(protocol.subprocess, "Popen", processes), \
             patch.object(protocol.os, "killpg", side_effect=AssertionError("no real process may be signalled")):
            protocol.main()

    def test_four_capture_workers_distributed_baseline_and_concurrent_shared_parent_branches(self):
        fake = FakeProcesses(self.root)
        self.invoke(fake)
        capture = [call for call in fake.calls if call["module"] == "capture_protocol"]
        self.assertEqual(len(capture), 4)
        self.assertEqual([call["gpu"] for call in capture], list(GPUS))
        for option in ("--prompts", "--reserves", "--out"):
            paths = [FakeProcesses.value(call["command"], option) for call in capture]
            self.assertEqual(len(set(paths)), 4)
        baseline = next(call for call in fake.calls if call["name"] == "baseline")
        self.assertEqual(baseline["gpu"], ",".join(GPUS))
        self.assertIn("torch.distributed.run", baseline["command"])
        self.assertIn("--nproc_per_node=4", baseline["command"])
        branches = [call for call in fake.calls if call["name"].startswith("branch_")]
        self.assertEqual([call["gpu"] for call in branches], list(GPUS[:3]))
        parent = str(self.root / "run_seed0/baseline/model.pt")
        for call in [baseline, *branches]:
            self.assertEqual(FakeProcesses.value(call["command"], "--requests-per-update"), "8")
            self.assertEqual(FakeProcesses.value(call["command"], "--expected-requests"), "8192")
        for call in branches:
            self.assertEqual(FakeProcesses.value(call["command"], "--init-checkpoint"), parent)
            self.assertNotIn("torch.distributed.run", call["command"])
        launches = [i for i, event in enumerate(fake.events) if event[0] == "launch" and event[1].startswith("branch_")]
        polls = [i for i, event in enumerate(fake.events) if event[0] == "poll" and event[1].startswith("branch_")]
        self.assertLess(max(launches), min(polls))
        lock = (self.root / "protocol.lock").stat()
        self.assertTrue(all(call["inherited"] == [(lock.st_dev, lock.st_ino)] for call in fake.calls))
        status = json.loads((self.root / "run_seed0/status.json").read_text())
        self.assertEqual(status["state"], "complete")
        self.assertEqual(status["config"]["additional_seeds"], [])

    def test_resume_skips_completed_jobs_even_if_stage_markers_were_not_committed(self):
        self.invoke(FakeProcesses(self.root))
        path = self.root / "run_seed0/status.json"
        status = json.loads(path.read_text())
        status["completed_stages"] = []
        path.write_text(json.dumps(status))
        resumed = FakeProcesses(self.root)
        self.invoke(resumed)
        self.assertEqual(resumed.calls, [])
        self.assertEqual(json.loads(path.read_text())["state"], "complete")


if __name__ == "__main__":
    unittest.main()
