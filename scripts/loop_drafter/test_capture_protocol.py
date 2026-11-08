"""Deterministic CPU oracles for batched EOS, causal alignment, and resume."""

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import torch
from torch import nn

from .capture_protocol import capture_batch, main, parser, run_capture


class OracleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.embed = nn.Embedding(64, 2)
        self.head = nn.Linear(2, 64, bias=False)

    def get_input_embeddings(self):
        return self.embed

    def get_output_embeddings(self):
        return self.head


class OracleTarget:
    """First prompt token sets generation length; states encode actual prefixes."""

    def __init__(self):
        self.model = OracleModel()
        self.tokenizer = SimpleNamespace(eos_token_id=0, chat_template=None,
                                         encode=lambda text, **kwargs: [int(value) for value in text.split()])
        self.calls = []

    def forward(self, input_ids, cache=None):
        ids = input_ids.cpu()
        batch, length = ids.shape
        previous = 0 if cache is None else cache["ids"].shape[1]
        if cache is None:
            cache = {"ids": ids.clone(), "prompt_length": length, "limits": ids[:, 0].clone()}
        else:
            cache["ids"] = torch.cat((cache["ids"], ids), dim=1)
        self.calls.append({"tokens": ids.clone(), "previous": previous})
        logits = torch.full((batch, length, 64), -100.)
        hidden = torch.empty(batch, length, 4, 2)
        for row in range(batch):
            for column in range(length):
                position = previous + column
                generated_before = position + 1 - cache["prompt_length"]
                token = 0 if generated_before + 1 >= cache["limits"][row] else 20 + generated_before
                logits[row, column, token] = 100
                prefix_sum = int(cache["ids"][row, :position + 1].sum())
                for round_index in range(4):
                    hidden[row, column, round_index] = torch.tensor([position, prefix_sum + round_index])
        return SimpleNamespace(logits=logits, round_hidden=hidden, cache=cache)


def record(identifier, source, prompt, domain="general"):
    return dict(id=identifier, source=source, domain=domain, split="train", prompt=prompt)


class CaptureProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def arguments(self, root, primary, reserves=None):
        prompts = root / "prompts.jsonl"
        prompts.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in primary))
        options = ["--model", "oracle", "--prompts", str(prompts), "--out", str(root / "out"),
                   "--device", "cpu", "--max-new-tokens", "6", "--max-prompt-tokens", "4", "--num-threads", "1"]
        if reserves is not None:
            reserve_path = root / "reserves.jsonl"
            reserve_path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in reserves))
            options += ["--reserves", str(reserve_path)]
        return parser().parse_args(options)

    def capture(self, args, target=None):
        target = target or OracleTarget()
        with redirect_stdout(io.StringIO()):
            result = run_capture(args, target_factory=lambda _: target)
        return result, target

    def test_batch_eos_state_alignment_and_independent_storage(self):
        target = OracleTarget()
        prompts = torch.tensor([[1, 7], [2, 8], [4, 9], [9, 6]])
        results = capture_batch(target, prompts, 5, {0})
        self.assertEqual([len(tokens) - 2 for tokens, _ in results], [1, 2, 4, 5])
        self.assertEqual(len(target.calls), 5)
        self.assertEqual([call["tokens"][0, 0].item() for call in target.calls[1:]], [0, 0, 0, 0])
        for tokens, hidden in results:
            self.assertEqual(tuple(hidden.shape), (len(tokens) - 1, 4, 2))
            self.assertEqual(hidden.dtype, torch.bfloat16)
            self.assertEqual(hidden.untyped_storage().nbytes(), hidden.numel() * hidden.element_size())
            for position in range(len(tokens) - 1):
                for round_index in range(4):
                    self.assertEqual(hidden[position, round_index].tolist(),
                                     [position, int(tokens[:position + 1].sum()) + round_index])
        self.assertEqual(len({hidden.data_ptr() for _, hidden in results}), 4)

    def test_all_rows_stop_without_an_extra_forward(self):
        target = OracleTarget()
        results = capture_batch(target, torch.tensor([[1, 7], [1, 8]]), 9, {0})
        self.assertEqual(len(target.calls), 1)
        self.assertEqual([tokens.tolist() for tokens, _ in results], [[1, 7, 0], [1, 8, 0]])

    def test_quota_reserves_length_buckets_and_complete_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root,
                [record("a", "s1", "1 7"), record("b", "s1", "2 8"),
                 record("c", "s2", "3 9 10"), record("d", "s2", "2 1 2 3 4")],
                [record("ra", "s1", "4 8"), record("rd", "s2", "2 9 10")])
            summary, target = self.capture(args)
            self.assertTrue(summary["complete"])
            self.assertEqual(summary["source_counts"], {"s1": 2, "s2": 2})
            self.assertEqual(summary["rejections"], 2)
            prefill = [call["tokens"] for call in target.calls if call["previous"] == 0]
            self.assertEqual([tuple(tokens.shape) for tokens in prefill], [(2, 2), (1, 3), (1, 2), (1, 3)])
            rows = [json.loads(line) for line in (root / "out/manifest.jsonl").read_text().splitlines()]
            self.assertEqual({row["id"] for row in rows}, {"b", "c", "ra", "rd"})
            self.assertEqual({Path(row["path"]).name for row in rows}, {"000001.pt", "000002.pt", "000004.pt", "000005.pt"})
            self.assertTrue(all(row["domain"] == "general" for row in rows))
            io_mtime = (root / "out/frozen_io.pt").stat().st_mtime_ns
            with redirect_stdout(io.StringIO()):
                resumed = run_capture(args, target_factory=lambda _: self.fail("completed resume loaded target"))
            self.assertTrue(resumed["complete"])
            self.assertEqual((root / "out/frozen_io.pt").stat().st_mtime_ns, io_mtime)
            self.assertEqual(resumed["all_generated_tokens"], 12)
            Path(args.prompts).write_text(Path(args.prompts).read_text() + "\n")
            with self.assertRaisesRegex(ValueError, "snapshot"):
                self.capture(args)

    def test_orphan_record_and_partial_manifest_resume_without_recapture(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root, [record("a", "s", "2 7"), record("b", "s", "3 8")])
            self.capture(args)
            manifest = root / "out/manifest.jsonl"
            lines = manifest.read_text().splitlines()
            manifest.write_text(lines[0] + "\n" + '{"id":')
            with redirect_stdout(io.StringIO()):
                resumed = run_capture(args, target_factory=lambda _: self.fail("orphan resume loaded target"))
            self.assertTrue(resumed["complete"])
            restored = [json.loads(line) for line in manifest.read_text().splitlines()]
            self.assertEqual([row["id"] for row in restored], ["a", "b"])

    def test_rejections_are_not_repeated_and_missing_reserve_fails_cli(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root, [record("a", "s1", "1 7"), record("b", "s2", "2 8")])
            summary, _ = self.capture(args)
            self.assertFalse(summary["complete"])
            self.assertEqual(summary["deficits"], {"s1": 1})
            with redirect_stdout(io.StringIO()):
                resumed = run_capture(args, target_factory=lambda _: self.fail("exhausted resume loaded target"))
            self.assertEqual(resumed["rejections"], 1)
            with patch("sys.argv", ["capture_protocol", "--model", "oracle", "--prompts", args.prompts,
                                    "--out", args.out]), patch(
                    f"{main.__module__}.run_capture", return_value=summary):
                with redirect_stdout(io.StringIO()), self.assertRaises(SystemExit):
                    main()

    def test_unicode_line_separators_inside_json_strings_survive_resume(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = self.arguments(root, [record("id\u2028a", "source\u2029one", "2\u20287", "domain\u2029one")])
            summary, _ = self.capture(args)
            self.assertTrue(summary["complete"])
            with redirect_stdout(io.StringIO()):
                resumed = run_capture(args, target_factory=lambda _: self.fail("complete resume loaded target"))
            self.assertEqual(resumed["source_counts"], {"source\u2029one": 1})


if __name__ == "__main__":
    unittest.main()
