"""CPU checks for fixed shards, preservation, source quotas, and canonical merge."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
from safetensors.torch import save_file

from .capture_parallel import _jsonl, merge, prepare
from .capture_protocol import _lock, parser, run_capture
from .test_capture_protocol import OracleTarget, record


class StopAfterBatch(OracleTarget):
    def forward(self, input_ids, cache=None):
        if cache is None and self.calls:
            raise RuntimeError("simulated interrupted capture")
        return super().forward(input_ids, cache)


class ParallelCaptureTests(unittest.TestCase):
    def setup_root(self, root, primary, reserves):
        (root / "prompts").mkdir()
        for name, rows in (("train", primary), ("reserves", reserves)):
            (root / f"prompts/{name}.jsonl").write_text(
                "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows))
        model = root / "model"
        model.mkdir()
        save_file({"model.embed_tokens.weight": torch.randn(64, 2), "lm_head.weight": torch.randn(64, 2)},
                  str(model / "model.safetensors"))
        return model

    def capture_args(self, model, prompts, reserves, out):
        return parser().parse_args([
            "--model", str(model), "--prompts", str(prompts), "--reserves", str(reserves),
            "--out", str(out), "--device", "cpu", "--max-new-tokens", "6",
            "--max-prompt-tokens", "4", "--num-threads", "1",
        ])

    def run_workers(self, plan):
        summaries = []
        for worker in plan["workers"]:
            args = self.capture_args(plan["capture_config"]["model"], worker["prompts"],
                                     worker["reserves"], worker["out"])
            with redirect_stdout(io.StringIO()):
                summaries.append(run_capture(args, target_factory=lambda _: OracleTarget()))
        return summaries

    def test_empty_capture_four_workers_and_canonical_merge(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = [record(f"{source}-{i}", source, "2 7") for i in range(4) for source in ("a", "b", "c")]
            reserves = [record(f"r-{source}-{i}", source, "3 8") for i in range(4) for source in ("a", "b", "c")]
            model = self.setup_root(root, primary, reserves)
            plan = prepare(root, workers=4, model=model, max_new_tokens=6, max_prompt_tokens=4)
            self.assertEqual(plan["existing_counts"], {"a": 0, "b": 0, "c": 0})
            self.assertEqual(len(plan["workers"]), 4)
            self.assertTrue(all(worker["quotas"] == {"a": 1, "b": 1, "c": 1} for worker in plan["workers"]))
            io_path = root / "capture_fp32/frozen_io.pt"
            frozen_io = torch.load(io_path, weights_only=True)
            self.assertEqual(frozen_io["embedding"].dtype, torch.float32)
            self.assertEqual(frozen_io["model_path"], str(model))
            for worker in plan["workers"]:
                link = Path(worker["out"]) / "frozen_io.pt"
                self.assertTrue(link.is_symlink())
                self.assertEqual(link.resolve(), io_path)
            self.assertTrue(all(summary["complete"] for summary in self.run_workers(plan)))
            config_before = (root / "capture_fp32/capture_config.json").read_bytes()
            summary = merge(root)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["source_counts"], {"a": 4, "b": 4, "c": 4})
            rows = _jsonl(root / "capture_fp32/manifest.jsonl")
            self.assertEqual([row["id"] for row in rows], [row["id"] for row in primary])
            self.assertTrue(all("capture_parallel/worker_" in row["path"] for row in rows))
            before = (root / "capture_fp32/manifest.jsonl").read_bytes()
            self.assertEqual(prepare(root, workers=4), plan)
            self.assertTrue(merge(root)["complete"])
            self.assertEqual((root / "capture_fp32/manifest.jsonl").read_bytes(), before)
            self.assertEqual((root / "capture_fp32/capture_config.json").read_bytes(), config_before)

    def test_migration_preserves_successes_and_replaces_historical_rejections(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            primary = [record(f"{source}-{i}", source, "2 7") for i in range(4) for source in ("a", "b", "c")]
            primary[0]["prompt"] = "1 7"
            reserves = [record(f"r-{source}-{i}", source, "3 8") for i in range(4) for source in ("a", "b", "c")]
            model = self.setup_root(root, primary, reserves)
            args = self.capture_args(model, root / "prompts/train.jsonl", root / "prompts/reserves.jsonl",
                                     root / "capture_fp32")
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(RuntimeError, "interrupted"):
                run_capture(args, target_factory=lambda _: StopAfterBatch())
            old_rows = _jsonl(root / "capture_fp32/manifest.jsonl")
            self.assertEqual(len(old_rows), 3)
            old_files = {row["path"]: Path(row["path"]).read_bytes() for row in old_rows}
            plan = prepare(root, workers=2)
            self.assertEqual(plan["existing_records"], 3)
            self.assertEqual(plan["promoted_reserve_ids"], ["r-a-0"])
            for worker in plan["workers"]:
                self.assertNotIn("a-0", worker["primary_ids"] + worker["reserve_ids"])
            # Live aggregated progress cannot alter shard assignment.
            (root / "capture_fp32/progress.json").write_text('{"done":999,"source_counts":{"a":999}}')
            self.assertEqual(prepare(root, workers=2), plan)
            self.assertTrue(all(summary["complete"] for summary in self.run_workers(plan)))
            summary = merge(root)
            self.assertEqual(summary["records"], 12)
            self.assertEqual(summary["rejections"], 1)
            rows = _jsonl(root / "capture_fp32/manifest.jsonl")
            for old in old_rows:
                self.assertIn(old, rows)
                self.assertEqual(Path(old["path"]).read_bytes(), old_files[old["path"]])
            expected = [row["id"] for row in primary[1:]] + ["r-a-0"]
            self.assertEqual([row["id"] for row in rows], expected)

    def test_incomplete_worker_does_not_publish_partial_manifest(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.setup_root(root, [record("bad", "a", "1 7"), record("ok", "b", "2 8")], [])
            plan = prepare(root, workers=2, model=model, max_new_tokens=6, max_prompt_tokens=4)
            self.assertFalse(all(summary["complete"] for summary in self.run_workers(plan)))
            with self.assertRaisesRegex(ValueError, "fulfill"):
                merge(root)
            self.assertFalse((root / "capture_fp32/manifest.jsonl").exists())
            self.assertFalse((root / "capture_fp32/capture_summary.json").exists())

    def test_changed_inputs_refuse_resume_and_active_original_refuses_prepare(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model = self.setup_root(root, [record("id\u2028one", "source\u2029a", "2\u20287")], [])
            capture = root / "capture_fp32"
            capture.mkdir()
            with _lock(capture), self.assertRaisesRegex(RuntimeError, "Another capture"):
                prepare(root, model=model)
            plan = prepare(root, workers=1, model=model, max_new_tokens=6, max_prompt_tokens=4)
            self.assertEqual(plan["quotas"], {"source\u2029a": 1})
            path = root / "prompts/train.jsonl"
            path.write_text(path.read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "contents changed"):
                prepare(root, workers=1)


if __name__ == "__main__":
    unittest.main()
