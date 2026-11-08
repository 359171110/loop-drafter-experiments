"""CPU-only rate, pairing, rotation, and resume tests; no model is loaded."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import random
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch

from . import evaluate_protocol as protocol
from . import target as target_module


def record(identifier, source, method, length, decode, prefill, rounds=()):
    return dict(id=identifier, source=source, method=method, tokens=list(range(length)),
                decode_seconds=decode, prefill_seconds=prefill, rounds=list(rounds))


def select_summary(rows, source, method):
    return next(row for row in protocol.aggregate(rows)
                if row["source"] == source and row["method"] == method)


class AggregateTests(unittest.TestCase):
    def test_rates_and_cycle_statistics_use_summed_denominators(self):
        rows = [record("p1", "math", "ar", 5, 2., 1.),
                record("p2", "math", "ar", 11, 18., 3.),
                record("p1", "math", "final", 5, 1., 2.,
                       [dict(committed_tokens=2, accepted_children=1)]),
                record("p2", "math", "final", 11, 3., 2.,
                       [dict(committed_tokens=3, accepted_children=2),
                        dict(committed_tokens=2, accepted_children=1),
                        dict(committed_tokens=1, accepted_children=0)])]
        result = select_summary(rows, "math", "final")
        self.assertEqual(result["generated_tokens"], 16)
        self.assertEqual(result["decode_tokens"], 14)
        self.assertAlmostEqual(result["decode_tokens_per_second"], 14 / 4)
        self.assertAlmostEqual(result["end_to_end_tokens_per_second"], 16 / 8)
        self.assertAlmostEqual(result["decode_latency_speedup"], 20 / 4)
        self.assertAlmostEqual(result["end_to_end_latency_speedup"], 24 / 8)
        self.assertEqual(result["verification_cycles"], 4)
        self.assertEqual(result["mean_committed_tokens_per_verify"], 2.)
        self.assertEqual(result["mean_accepted_children_per_verify"], 1.)
        # Unequal durations distinguish summed throughput from mean request TPS.
        self.assertNotEqual(result["decode_tokens_per_second"], (4 / 1 + 10 / 3) / 2)

    def test_paired_latency_excludes_unpaired_requests_but_rate_does_not(self):
        rows = [record("p1", "chat", "ar", 5, 8., 1.),
                record("p1", "chat", "trajectory", 5, 2., 1.),
                record("pending", "chat", "trajectory", 3, 100., 10.)]
        result = select_summary(rows, "chat", "trajectory")
        self.assertEqual(result["requests"], 2)
        self.assertEqual(result["paired_requests"], 1)
        self.assertEqual(result["exact_matches"], 1)
        self.assertAlmostEqual(result["decode_latency_speedup"], 4.)
        self.assertAlmostEqual(result["end_to_end_latency_speedup"], 3.)
        self.assertAlmostEqual(result["decode_tokens_per_second"], 6 / 102)

    def test_pairing_requires_both_source_and_id(self):
        rows = [record("shared-id", "math", "ar", 5, 10., 1.),
                record("shared-id", "code", "ar", 5, 100., 1.),
                record("shared-id", "math", "final", 5, 5., 1.),
                record("shared-id", "unpaired-source", "final", 5, 2., 1.)]
        result = select_summary(rows, "math", "final")
        self.assertEqual(result["decode_latency_speedup"], 2.)
        unpaired = select_summary(rows, "unpaired-source", "final")
        self.assertEqual(unpaired["paired_requests"], 0)
        self.assertEqual(unpaired["exact_matches"], 0)
        self.assertIsNone(unpaired["decode_latency_speedup"])

    def test_all_sources_uses_total_time_and_identity_is_only_diagnostic(self):
        rows = [record("a", "math", "ar", 4, 1., 1.),
                record("b", "code", "ar", 10, 9., 1.),
                record("a", "math", "multiloop", 4, 1., 1.),
                record("b", "code", "multiloop", 10, 3., 1.)]
        rows[-1]["tokens"][-1] = 999
        result = select_summary(rows, "__all__", "multiloop")
        self.assertEqual(result["decode_tokens"], 12)
        self.assertEqual(result["decode_tokens_per_second"], 3.)
        self.assertEqual(result["exact_matches"], 1)
        self.assertEqual(result["paired_requests"], 2)
        self.assertNotIn("accuracy", result)
        self.assertNotIn("task_accuracy", result)

    def test_single_output_and_zero_time_do_not_divide_by_zero(self):
        rows = [record("a", "math", "ar", 1, 0., 0.)]
        result = select_summary(rows, "math", "ar")
        self.assertEqual(result["decode_tokens"], 0)
        self.assertIsNone(result["decode_tokens_per_second"])
        self.assertIsNone(result["end_to_end_tokens_per_second"])
        self.assertIsNone(result["mean_committed_tokens_per_verify"])
        self.assertEqual(protocol.aggregate([]), [])


class FakeDraft:
    def __init__(self, mode):
        self.config = SimpleNamespace(feature_mode=mode, source_layout="aligned")
        self.config.dict = lambda: dict(feature_mode=mode, source_layout="aligned")

    def eval(self):
        return self


class MainProtocolTests(unittest.TestCase):
    methods = ["ar", "final", "multiloop", "trajectory"]

    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        # Small fake panel exercises the loop; production default stays 300/source.
        self.prompts = [dict(id=f"p{i}", source="math" if i < 2 else "code",
                             domain="math" if i < 2 else "code", split="eval", prompt=f"prompt {i}")
                        for i in range(4)]
        self.prompt_path = self.root / "eval.jsonl"
        self.prompt_path.write_text("".join(json.dumps(row) + "\n" for row in self.prompts))
        self.out = self.root / "out"
        checkpoints = [self.root / f"{mode}.pt" for mode in self.methods[1:]]
        for path in checkpoints:
            path.write_bytes(b"fake checkpoint used only for file identity")
        self.argv = ["evaluate_protocol", "--model", "local-fake-model", "--prompts", str(self.prompt_path),
                     "--checkpoints", *map(str, checkpoints),
                     "--frozen-io", "fake-frozen-io.pt", "--out", str(self.out),
                     "--expected-per-source", "2", "--max-new-tokens", "16", "--device", "cpu",
                     "--num-threads", "1"]

    def run_main(self, *, stop_after=None, checkpoint_step=100):
        calls = []

        def decode(method, ids, budget):
            if budget == 16:
                if stop_after is not None and len(calls) == stop_after:
                    raise RuntimeError("simulated interruption")
                calls.append((f"p{int(ids[0, 0])}", method))
            return dict(tokens=[1, 2, 3], prefill_seconds=.25, decode_seconds=.5,
                        rounds=[] if method == "ar" else [dict(committed_tokens=3, accepted_children=2)])

        def checkpoint(path, **kwargs):
            mode = Path(path).stem
            return FakeDraft(mode), {"cursor": {"step": checkpoint_step, "epoch": 1, "next_request": 0}}

        target = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
        with patch("sys.argv", self.argv), \
             patch.object(protocol.torch, "set_num_threads"), \
             patch.object(target_module.OuroTarget, "from_pretrained", return_value=target), \
             patch.object(protocol, "load_checkpoint", side_effect=checkpoint), \
             patch.object(protocol, "prompt_ids", side_effect=lambda tok, row, maximum: torch.tensor([[int(row["id"][1:])]])), \
             patch.object(protocol, "decode_ar", side_effect=lambda target, ids, budget, eos: decode("ar", ids, budget)), \
             patch.object(protocol, "decode_speculative", side_effect=lambda target, draft, ids, budget, eos: decode(draft.config.feature_mode, ids, budget)), \
             redirect_stdout(io.StringIO()):
            try:
                protocol.main()
            except RuntimeError as error:
                if stop_after is None or str(error) != "simulated interruption":
                    raise
        return calls

    def test_four_methods_rotate_and_resume_appends_only_missing_rows(self):
        first = self.run_main(stop_after=5)
        partial = protocol.read_jsonl(self.out / "requests.jsonl")
        self.assertEqual(len(partial), 5)
        progress = json.loads((self.out / "progress.json").read_text())
        self.assertEqual(progress["completed_method_requests"], 5)
        second = self.run_main()
        rows = protocol.read_jsonl(self.out / "requests.jsonl")
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({(r["id"], r["method"]) for r in rows}), 16)
        self.assertEqual(rows[:5], partial)
        self.assertEqual(len(second), 11)
        shuffled = list(self.prompts)
        random.Random(0).shuffle(shuffled)
        expected = []
        for index, row in enumerate(shuffled):
            offset = index % 4
            expected.extend((row["id"], method) for method in self.methods[offset:] + self.methods[:offset])
        self.assertEqual(first + second, expected)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertIn("no answer correctness scoring", summary["metric_notes"]["output_comparison"])
        self.assertEqual(summary["config"]["seed"], 0)
        self.assertEqual(self.run_main(), [])

    def test_resume_recovers_incomplete_final_json_line(self):
        first = self.run_main(stop_after=5)
        path = self.out / "requests.jsonl"
        committed_prefix = path.read_bytes()
        partial = protocol.read_jsonl(path)
        with path.open("ab") as stream:
            stream.write(b'{"id": "interrupted", "method":')

        resumed = self.run_main()

        rows = protocol.read_jsonl(path)
        self.assertTrue(path.read_bytes().startswith(committed_prefix))
        self.assertEqual(rows[:5], partial)
        self.assertEqual(len(resumed), 11)
        self.assertEqual(len(rows), 16)
        self.assertEqual(len(set(first + resumed)), 16)
        self.assertEqual(len({(row["id"], row["method"]) for row in rows}), 16)
        progress = json.loads((self.out / "progress.json").read_text())
        self.assertEqual(progress["completed_method_requests"], 16)
        self.assertEqual(self.run_main(), [])

    def test_resume_rejects_log_source_mismatch(self):
        self.run_main()
        path = self.out / "requests.jsonl"
        rows = protocol.read_jsonl(path)
        rows[0]["source"] = "incorrect-dataset"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaises(ValueError):
            self.run_main()

    def test_resume_rejects_replaced_checkpoint_at_same_path(self):
        self.run_main(checkpoint_step=100)
        with self.assertRaises(ValueError):
            self.run_main(checkpoint_step=200)

    def test_resume_rejects_changed_prompt_contents(self):
        self.run_main()
        changed = [dict(row) for row in self.prompts]
        changed[0]["prompt"] += " changed"
        self.prompt_path.write_text("".join(json.dumps(row) + "\n" for row in changed))
        with self.assertRaises(ValueError):
            self.run_main()


if __name__ == "__main__":
    unittest.main()
