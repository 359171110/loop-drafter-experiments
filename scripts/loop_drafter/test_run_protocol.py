"""CPU orchestration checks with fake subprocesses and real stage artifacts."""
from contextlib import redirect_stdout
import io
import json
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from . import run_protocol


class FakeProcesses:
    """Create only the output names actually produced by the child modules."""
    def __init__(self, fail_stage=None, missing_artifact=None, launch_failure=None, wait_failure=None):
        self.fail_stage = fail_stage
        self.missing_artifact = missing_artifact
        self.launch_failure = launch_failure
        self.wait_failure = wait_failure
        self.calls = []

    @staticmethod
    def value(command, option):
        return command[command.index(option) + 1]

    def __call__(self, command, **kwargs):
        module = command[2].rsplit(".", 1)[-1]
        out = Path(self.value(command, "--out"))
        stage = {"capture_protocol": "capture", "evaluate_protocol": "speed_evaluation"}.get(module, out.name)
        # Inspect the descriptor while Popen would inherit it. The OS descriptor
        # can be closed after main returns, so retain its file identity instead.
        inherited = [os.fstat(fd) for fd in kwargs.get("pass_fds", ())]
        self.calls.append(dict(stage=stage, module=module, command=command, cwd=kwargs["cwd"],
                               inherited_files=[(stat.st_dev, stat.st_ino) for stat in inherited]))
        if stage == self.launch_failure:
            raise OSError("simulated child launch failure")
        owner = self

        class Process:
            pid = 10000 + len(owner.calls)

            def wait(self):
                if stage == owner.wait_failure:
                    raise KeyboardInterrupt("simulated interruption while child remains active")
                out.mkdir(parents=True, exist_ok=True)
                if module == "train_protocol":
                    # A failed training process may leave a valid resumable state.
                    (out / "latest.pt").write_text("fake optimizer checkpoint")
                if stage == owner.fail_stage:
                    return 17
                if stage != owner.missing_artifact:
                    if module == "capture_protocol":
                        (out / "capture_summary.json").write_text(json.dumps(dict(complete=True)))
                        (out / "manifest.jsonl").write_text("")
                        (out / "frozen_io.pt").write_text("fake target weights")
                    elif module == "train_protocol":
                        (out / "model.pt").write_text("fake completed model")
                    elif module == "evaluate_protocol":
                        (out / "summary.json").write_text("{}")
                    else:
                        raise AssertionError(f"unexpected child module {module}")
                return 0

        return Process()


class RunProtocolTests(unittest.TestCase):
    def invoke(self, root, processes):
        argv = ["run_protocol", "--model", "/fake/ouro", "--root", str(root)]
        with patch.object(run_protocol.sys, "argv", argv), \
                patch.object(run_protocol.subprocess, "Popen", processes), \
                redirect_stdout(io.StringIO()):
            run_protocol.main()

    def status(self, root):
        return json.loads((Path(root) / "run_seed0/status.json").read_text())

    def test_baseline_then_all_shared_parent_branches_then_evaluation(self):
        with tempfile.TemporaryDirectory() as root:
            processes = FakeProcesses()
            self.invoke(root, processes)
            stages = [call["stage"] for call in processes.calls]
            self.assertEqual(stages, ["capture", "baseline", "branch_final", "branch_multiloop",
                                      "branch_trajectory", "speed_evaluation"])
            lock_stat = (Path(root) / "protocol.lock").stat()
            for call in processes.calls:
                self.assertEqual(call["inherited_files"], [(lock_stat.st_dev, lock_stat.st_ino)])
            run = Path(root) / "run_seed0"
            baseline = str(run / "baseline/model.pt")
            train_calls = [call for call in processes.calls if call["module"] == "train_protocol"]
            for call in train_calls[1:]:
                self.assertEqual(FakeProcesses.value(call["command"], "--init-checkpoint"), baseline)
                self.assertEqual(FakeProcesses.value(call["command"], "--role"), "branch")
            self.assertNotIn("--init-checkpoint", train_calls[0]["command"])
            evaluation = processes.calls[-1]["command"]
            start, end = evaluation.index("--checkpoints") + 1, evaluation.index("--out")
            self.assertEqual(evaluation[start:end], [str(run / f"branch_{mode}/model.pt")
                                                    for mode in ("final", "multiloop", "trajectory")])
            status = self.status(root)
            self.assertEqual(status["state"], "complete")
            self.assertEqual(status["completed_stages"], stages)
            self.assertIsNone(status["config"]["development_panel"])
            self.assertEqual(status["config"]["additional_seeds"], [])
            self.assertEqual(status["config"]["seed"], 0)

    def test_failed_branch_blocks_later_stages_and_resume_continues_its_checkpoint(self):
        with tempfile.TemporaryDirectory() as root:
            processes = FakeProcesses(fail_stage="branch_multiloop")
            with self.assertRaises(SystemExit) as raised:
                self.invoke(root, processes)
            self.assertEqual(raised.exception.code, 17)
            self.assertEqual([call["stage"] for call in processes.calls],
                             ["capture", "baseline", "branch_final", "branch_multiloop"])
            status = self.status(root)
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["completed_stages"], ["capture", "baseline", "branch_final"])
            resumed = FakeProcesses()
            self.invoke(root, resumed)
            self.assertEqual([call["stage"] for call in resumed.calls],
                             ["branch_multiloop", "branch_trajectory", "speed_evaluation"])
            self.assertIn("--resume", resumed.calls[0]["command"])
            self.assertNotIn("--resume", resumed.calls[1]["command"])
            self.assertEqual(self.status(root)["state"], "complete")

    def test_identical_completed_run_skips_every_subprocess(self):
        with tempfile.TemporaryDirectory() as root:
            self.invoke(root, FakeProcesses())
            resumed = FakeProcesses()
            self.invoke(root, resumed)
            self.assertEqual(resumed.calls, [])
            self.assertEqual(self.status(root)["state"], "complete")

    def test_zero_exit_without_required_artifact_is_a_failed_stage(self):
        with tempfile.TemporaryDirectory() as root:
            processes = FakeProcesses(missing_artifact="baseline")
            with self.assertRaises(SystemExit) as raised:
                self.invoke(root, processes)
            self.assertEqual(raised.exception.code, 1)
            self.assertEqual([call["stage"] for call in processes.calls], ["capture", "baseline"])
            self.assertEqual(self.status(root)["completed_stages"], ["capture"])
            self.assertEqual(self.status(root)["state"], "failed")

    def test_child_launch_exception_records_failure_and_blocks_following_stages(self):
        with tempfile.TemporaryDirectory() as root:
            processes = FakeProcesses(launch_failure="baseline")
            with self.assertRaisesRegex(OSError, "simulated child launch failure"):
                self.invoke(root, processes)
            self.assertEqual([call["stage"] for call in processes.calls], ["capture", "baseline"])
            status = self.status(root)
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["stage"], "baseline")
            self.assertEqual(status["completed_stages"], ["capture"])
            self.assertIn("OSError", status["error"])
            self.assertIn("failed_at", status)
            self.assertNotIn("child_pid", status)

    def test_wait_interruption_records_failure_and_preserves_active_child_identity(self):
        with tempfile.TemporaryDirectory() as root:
            processes = FakeProcesses(wait_failure="baseline")
            with self.assertRaisesRegex(KeyboardInterrupt, "child remains active"):
                self.invoke(root, processes)
            status = self.status(root)
            self.assertEqual(status["state"], "failed")
            self.assertEqual(status["completed_stages"], ["capture"])
            self.assertIn("KeyboardInterrupt", status["error"])
            self.assertEqual(status["child_pid"], 10002)
            self.assertEqual([call["stage"] for call in processes.calls], ["capture", "baseline"])


if __name__ == "__main__":
    unittest.main()
