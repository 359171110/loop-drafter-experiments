"""CPU tests for accepted-history rollback and paired resumable evaluation."""

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

from . import evaluate_history as evaluation
from . import target as target_module
from .test_evaluate import EOS, FakeDraft, FakeTarget, update_state


class FakeHistoryDraft(FakeDraft):
    def __init__(self, prompt_length, *, window=64, **kwargs):
        super().__init__(prompt_length, **kwargs)
        self.config.window = window
        self.histories = []

    def generate_draft(self, history, history_mask, bonus, max_children=None):
        if history.ndim != 3 or history.shape[0] != 1 or history.shape[2] != 4:
            raise AssertionError("expected batch-one final-round history")
        if not 1 <= history.shape[1] <= self.config.window:
            raise AssertionError("history exceeds window")
        if history.dtype != torch.bfloat16:
            raise AssertionError("history must match BF16 capture snapshots")
        if history_mask.dtype != torch.bool or history_mask.shape != history.shape[:2]:
            raise AssertionError("wrong history mask")
        if not history_mask.all():
            raise AssertionError("decoder should pass unpadded actual states")
        if not history[:, :, -1].eq(3).all():
            raise AssertionError("history must select round four")
        self.histories.append(history.clone())
        taps = history[:, -1:, :].expand(-1, 4, -1)
        return super().generate_draft(taps, bonus, max_children=max_children)


def expected_history(tokens, window):
    state, rows = 0, []
    for index, token in enumerate(tokens):
        state = update_state(state, token)
        rows.append([state, index + 1, token, 3])
    return torch.tensor([rows[-window:]], dtype=torch.bfloat16)


class HistoryDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def run_case(self, budget, *, prompt_length=3, window=64, failures=(None,),
                 eos_after=None, wrong_eos=False, children=None, return_new_cache=False):
        prompt = torch.tensor([[index % 23 + 1 for index in range(prompt_length)]], dtype=torch.long)
        target = FakeTarget(prompt_length, eos_after, return_new_cache)
        draft = FakeHistoryDraft(prompt_length, window=window, depth=5,
                                 failures=failures, eos_after=eos_after, wrong_eos=wrong_eos)
        reference = evaluation.decode_ar(FakeTarget(prompt_length, eos_after), prompt, budget, {EOS})
        result = evaluation.decode_history_speculative(target, draft, prompt, budget, {EOS}, children)
        self.assertEqual(result["tokens"], reference["tokens"])
        self.assertLessEqual(len(result["tokens"]), budget)
        self.assertEqual(result["target_calls"], len(target.forward_calls))
        self.assertEqual(len(result["rounds"]), len(draft.histories))
        committed = 0
        for history, proposal, forward, crop, ledger in zip(
                draft.histories, draft.calls, target.forward_calls[1:],
                target.crop_events, result["rounds"]):
            prefix = prompt[0].tolist() + result["tokens"][:committed]
            # Check every row, not only the last tap: rejected descendants must
            # never reappear in later context even after a sliding-window crop.
            torch.testing.assert_close(history, expected_history(prefix, window), rtol=0, atol=0)
            self.assertEqual(forward["prefix"], prefix)
            self.assertEqual(proposal["source_length"], len(prefix))
            self.assertEqual(forward["input"], [proposal["bonus"]] + proposal["proposals"])
            self.assertEqual(ledger["committed_tokens"], ledger["accepted_children"] + 1)
            committed += ledger["committed_tokens"]
            self.assertEqual(crop["kept"], prompt[0].tolist() + result["tokens"][:committed])
            self.assertEqual(crop["length"], prompt_length + committed)
        self.assertIn(len(result["tokens"]) - committed, (0, 1))
        return result, draft, target

    def test_prefill_uses_all_available_h4_and_truncates_long_prompts(self):
        for prompt_length in (1, 3, 64, 79):
            with self.subTest(prompt_length=prompt_length):
                _, draft, _ = self.run_case(15, prompt_length=prompt_length)
                self.assertEqual(draft.histories[0].shape[1], min(prompt_length, 64))

    def test_rejections_roll_back_entire_history_and_target_cache(self):
        for failures in ((0,), (1,), (2,), (3,), (4,), (4, 0, None, 2, 1)):
            for new_cache in (False, True):
                with self.subTest(failures=failures, new_cache=new_cache):
                    self.run_case(41, prompt_length=61, failures=failures, return_new_cache=new_cache)

    def test_correct_children_cross_history_window_boundary(self):
        result, draft, _ = self.run_case(45, prompt_length=3, window=8)
        self.assertTrue(any(history.shape[1] == 8 for history in draft.histories))
        self.assertTrue(all(row["accepted_children"] == row["proposed_children"]
                            for row in result["rounds"]))

    def test_eos_at_bonus_and_each_child_never_commits_after_eos(self):
        for eos_after in (1, 2, 3, 4, 5, 6, 7, 11):
            for failures in ((None,), (0,), (4, 1, None)):
                with self.subTest(eos_after=eos_after, failures=failures):
                    result, draft, target = self.run_case(20, eos_after=eos_after, failures=failures)
                    self.assertEqual(len(result["tokens"]), eos_after)
                    self.assertEqual(result["tokens"][-1], EOS)
                    if eos_after == 1:
                        self.assertFalse(draft.histories)
                        self.assertEqual(len(target.forward_calls), 1)

    def test_wrong_proposed_eos_does_not_end_generation(self):
        result, _, _ = self.run_case(18, failures=(0, 2, 4), wrong_eos=True)
        self.assertEqual(len(result["tokens"]), 18)
        self.assertNotIn(EOS, result["tokens"])

    def test_budget_and_child_limits(self):
        for budget in (1, 2, 5, 6, 7, 12):
            for children in (0, 1, 3, 5):
                with self.subTest(budget=budget, children=children):
                    result, draft, _ = self.run_case(budget, failures=(3, 0, None), children=children)
                    self.assertEqual(len(result["tokens"]), budget)
                    self.assertTrue(all(call["count"] <= children for call in draft.calls))
                    if budget == 1:
                        self.assertFalse(draft.calls)

    def test_invalid_input_or_budget_fails_before_target_call(self):
        target = FakeTarget(1)
        draft = FakeHistoryDraft(1)
        for ids, budget, children in ((torch.tensor([[1]]), 0, None),
                                     (torch.tensor([[1], [2]]), 3, None),
                                     (torch.tensor([[1]]), 3, 6)):
            with self.assertRaises(ValueError):
                evaluation.decode_history_speculative(target, draft, ids, budget, {EOS}, children)
        self.assertFalse(target.forward_calls)

    def test_actual_history_module_interface_matches_ar_in_both_modes(self):
        from .history_draft import HistoryConfig, HistoryDraft
        prompt = torch.tensor([[2, 3, 5]], dtype=torch.long)
        embedding = torch.randn(31, 4)
        reference = evaluation.decode_ar(FakeTarget(3), prompt, 17, {EOS})
        for mode in ("history", "history_residual"):
            with self.subTest(mode=mode):
                config = HistoryConfig(feature_mode=mode, target_hidden=4, vocab_size=31,
                                       width=8, mlp_width=16, layers=1, heads=2, window=8, depth=5)
                draft = HistoryDraft(config, embedding).eval()
                result = evaluation.decode_history_speculative(FakeTarget(3), draft, prompt, 17, {EOS})
                self.assertEqual(result["tokens"], reference["tokens"])


class MainHistoryTests(unittest.TestCase):
    def setUp(self):
        self.temp = TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.prompts = [dict(id=f"p{i}", source="math" if i < 2 else "code",
                             domain="math" if i < 2 else "code", split="eval", prompt=f"prompt {i}")
                        for i in range(4)]
        self.prompt_path = self.root / "eval.jsonl"
        self.prompt_path.write_text("".join(json.dumps(row) + "\n" for row in self.prompts))
        self.out = self.root / "out"
        self.paths = {mode: self.root / f"{mode}.pt" for mode in evaluation.METHODS[1:]}
        for path in self.paths.values():
            path.write_bytes(b"fake checkpoint identity")
        self.argv = ["evaluate_history", "--model", "local-fake-model", "--prompts", str(self.prompt_path),
                     "--ssd-checkpoint", str(self.paths["ssd"]),
                     "--history-checkpoint", str(self.paths["history"]),
                     "--residual-checkpoint", str(self.paths["history_residual"]),
                     "--frozen-io", "fake-frozen-io.pt", "--out", str(self.out),
                     "--expected-per-source", "2", "--max-new-tokens", "16", "--device", "cpu",
                     "--num-threads", "1"]

    def run_main(self, *, stop_after=None, checkpoint_step=100, swapped=False, gpu_assignment="5"):
        calls, loaders = [], []

        def decode(method, ids, budget):
            if budget == 16:
                if stop_after is not None and len(calls) == stop_after:
                    raise RuntimeError("simulated interruption")
                calls.append((f"p{int(ids[0, 0])}", method))
            return dict(tokens=[1, 2, 3], prefill_seconds=.25, decode_seconds=.5,
                        rounds=[] if method == "ar" else [dict(committed_tokens=3, accepted_children=2)])

        def checkpoint(path, *, kind, **kwargs):
            mode = Path(path).stem
            loaders.append((mode, kind))
            feature_mode = "final" if mode == "ssd" else mode
            if swapped and mode == "history":
                feature_mode = "history_residual"
            config = SimpleNamespace(feature_mode=feature_mode)
            config.dict = lambda: dict(feature_mode=feature_mode)
            draft = SimpleNamespace(config=config, method=mode)
            draft.eval = lambda: draft
            return draft, {"cursor": {"step": checkpoint_step, "epoch": 4, "next_request": 0}}

        target = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
        with patch("sys.argv", self.argv), \
             patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": gpu_assignment}), \
             patch.object(evaluation.torch, "set_num_threads"), \
             patch.object(target_module.OuroTarget, "from_pretrained", return_value=target), \
             patch.object(evaluation, "load_checkpoint", side_effect=lambda path, **kw: checkpoint(path, kind="ssd", **kw)), \
             patch.object(evaluation, "load_history_checkpoint", side_effect=lambda path, **kw: checkpoint(path, kind="history", **kw)), \
             patch.object(evaluation, "prompt_ids", side_effect=lambda tok, row, maximum: torch.tensor([[int(row["id"][1:])]])), \
             patch.object(evaluation, "decode_ar", side_effect=lambda target, ids, budget, eos: decode("ar", ids, budget)), \
             patch.object(evaluation, "decode_speculative", side_effect=lambda target, draft, ids, budget, eos: decode(draft.method, ids, budget)), \
             patch.object(evaluation, "decode_history_speculative", side_effect=lambda target, draft, ids, budget, eos: decode(draft.method, ids, budget)), \
             redirect_stdout(io.StringIO()):
            try:
                evaluation.main()
            except RuntimeError as error:
                if stop_after is None or str(error) != "simulated interruption":
                    raise
        self.assertEqual(loaders, [("ssd", "ssd"), ("history", "history"), ("history_residual", "history")])
        return calls

    def test_rotation_pairing_resume_and_truncated_last_record(self):
        first = self.run_main(stop_after=5)
        path = self.out / "requests.jsonl"
        initial = evaluation.read_jsonl(path)
        with path.open("ab") as stream:
            stream.write(b'{"id": "partial"')
        rest = self.run_main()
        rows = evaluation.read_jsonl(path)
        self.assertEqual(rows[:5], initial)
        self.assertEqual(len(rows), 16)
        self.assertEqual(len({(row["id"], row["method"]) for row in rows}), 16)
        shuffled = list(self.prompts)
        random.Random(0).shuffle(shuffled)
        expected = []
        for index, prompt in enumerate(shuffled):
            offset = index % 4
            expected.extend((prompt["id"], method)
                            for method in evaluation.METHODS[offset:] + evaluation.METHODS[:offset])
        self.assertEqual(first + rest, expected)
        self.assertTrue(all(row["device"] == "cpu" and row["cuda_visible_devices"] == "5" for row in rows))
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["config"]["seed"], 0)
        self.assertEqual(summary["source_counts"], {"math": 2, "code": 2})
        self.assertIn("no answer correctness scoring", summary["metric_notes"]["output_comparison"])
        for row in summary["results"]:
            self.assertEqual(row["exact_matches"], row["requests"])
        self.assertEqual(self.run_main(), [])

    def test_swapped_history_checkpoint_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "checkpoint has feature_mode"):
            self.run_main(swapped=True)

    def test_resume_rejects_changed_checkpoint_and_gpu_assignment(self):
        self.run_main(stop_after=5)
        with self.assertRaisesRegex(ValueError, "checkpoint identity changed"):
            self.run_main(checkpoint_step=200)
        with self.assertRaisesRegex(ValueError, "same GPU assignment"):
            self.run_main(gpu_assignment="6")

    def test_resume_rejects_unknown_or_duplicate_rows(self):
        self.run_main(stop_after=5)
        path = self.out / "requests.jsonl"
        rows = evaluation.read_jsonl(path)
        path.write_text("".join(json.dumps(row) + "\n" for row in rows + rows[:1]))
        with self.assertRaisesRegex(ValueError, "duplicate evaluation"):
            self.run_main()
        rows[0]["source"] = "different-source"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        with self.assertRaisesRegex(ValueError, "do not belong"):
            self.run_main()


if __name__ == "__main__":
    unittest.main()
