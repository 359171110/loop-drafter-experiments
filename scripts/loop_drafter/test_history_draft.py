"""CPU tests for committed history, padding, causal queries and residual control."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch
from torch.nn import functional as F

from .history_draft import HistoryConfig, HistoryDraft, load_history_checkpoint, make_history_batch
from .train import make_batch


class HistoryDraftTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(11)
        self.config = HistoryConfig(target_hidden=8, vocab_size=23, width=16,
                                    mlp_width=24, layers=2, heads=4, window=4, depth=3)
        self.embedding = torch.randn(23, 8)
        self.lm_head = torch.randn(23, 8)
        self.record = dict(token_ids=torch.tensor([2, 3, 4, 5, 6, 7, 8, 9, 10]), prompt_length=3,
                           round_hidden=torch.randn(8, 4, 8))

    def model(self, mode="history"):
        return HistoryDraft(replace(self.config, feature_mode=mode), self.embedding, self.lm_head)

    @staticmethod
    def inputs(batch):
        return batch["history"], batch["history_mask"], batch["bonus"]

    def test_history_window_and_labels_follow_the_processed_anchor(self):
        batch = make_history_batch(self.record, [2, 6], depth=3, window=4)
        original = make_batch(self.record, [2, 6], depth=3)
        for key in ("labels", "valid", "bonus"):
            torch.testing.assert_close(batch[key], original[key], rtol=0, atol=0)
        self.assertEqual(batch["history_mask"].tolist(), [[False, True, True, True], [True] * 4])
        torch.testing.assert_close(batch["history"][0, 0], torch.zeros(8), rtol=0, atol=0)
        torch.testing.assert_close(batch["history"][0, 1:], self.record["round_hidden"][:3, -1], rtol=0, atol=0)
        torch.testing.assert_close(batch["history"][1], self.record["round_hidden"][3:7, -1], rtol=0, atol=0)
        self.assertEqual(batch["bonus"].tolist(), [5, 9])
        self.assertEqual(batch["labels"].tolist(), [[6, 7, 8], [10, -100, -100]])
        self.assertNotIn("previous", batch)

    def test_future_hidden_and_gold_children_cannot_change_logits(self):
        batch = make_history_batch(self.record, [2], depth=3, window=4)
        changed = dict(self.record, round_hidden=self.record["round_hidden"].clone(),
                       token_ids=self.record["token_ids"].clone())
        changed["round_hidden"][3:] = 999
        changed["token_ids"][4:] = 19
        other = make_history_batch(changed, [2], depth=3, window=4)
        self.assertFalse(torch.equal(batch["labels"], other["labels"]))
        for mode in ("history", "history_residual"):
            model = self.model(mode).eval()
            torch.testing.assert_close(model(*self.inputs(batch)), model(*self.inputs(other)), rtol=0, atol=0)

    def test_later_query_embeddings_cannot_affect_earlier_predictions(self):
        batch = make_history_batch(self.record, [2, 4], depth=3, window=4)
        model = self.model().eval()
        original = model(*self.inputs(batch))
        with torch.no_grad():
            model.offset_embedding[2].add_(torch.randn(16) * 10)
        changed = model(*self.inputs(batch))
        torch.testing.assert_close(original[:, :2], changed[:, :2], rtol=0, atol=0)
        self.assertFalse(torch.allclose(original[:, 2], changed[:, 2]))
        _, _, allowed, valid, _ = model.prepare_queries(*self.inputs(batch))
        for row in range(len(valid)):
            for query in torch.where(valid[row])[0].tolist():
                self.assertFalse(bool(allowed[row, query, query + 1:].any()))
                self.assertFalse(bool(allowed[row, query, ~valid[row]].any()))

    def test_padding_is_inert_and_batched_matches_unpadded_individual_requests(self):
        history = torch.randn(2, 4, 8)
        mask = torch.tensor([[False, False, False, True], [True, True, True, True]])
        bonus = torch.tensor([3, 8])
        changed = history.clone().masked_fill(~mask[..., None], float("nan"))
        for mode in ("history", "history_residual"):
            model = self.model(mode).eval()
            actual = model(changed, mask, bonus)
            expected = model(history, mask, bonus)
            self.assertTrue(bool(torch.isfinite(actual).all()))
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            for index in range(2):
                one_history = history[index:index + 1, mask[index]]
                one_mask = torch.ones(1, one_history.shape[1], dtype=torch.bool)
                single = model(one_history, one_mask, bonus[index:index + 1])
                torch.testing.assert_close(actual[index:index + 1], single, rtol=1e-5, atol=2e-6)

    def test_invalid_memory_has_zero_gradient_and_target_io_remain_frozen(self):
        embedding, lm_head = self.embedding.clone().requires_grad_(True), self.lm_head.clone().requires_grad_(True)
        model = HistoryDraft(self.config, embedding, lm_head)
        history = torch.randn(2, 4, 8, requires_grad=True)
        mask = torch.tensor([[False, False, True, True], [False, True, True, True]])
        logits = model(history, mask, torch.tensor([3, 8]))
        F.cross_entropy(logits.flatten(0, 1), torch.tensor([4, 5, 6, 7, 8, 9])).backward()
        torch.testing.assert_close(history.grad[~mask], torch.zeros_like(history.grad[~mask]), rtol=0, atol=0)
        self.assertTrue(bool(torch.isfinite(history.grad).all()))
        self.assertIsNone(embedding.grad)
        self.assertIsNone(lm_head.grad)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(bool(torch.isfinite(parameter.grad).all()), name)

    def test_two_modes_have_identical_weights_and_only_the_requested_hidden_residual(self):
        torch.manual_seed(42)
        plain = self.model("history").eval()
        torch.manual_seed(42)
        residual = self.model("history_residual").eval()
        self.assertEqual(plain.state_dict().keys(), residual.state_dict().keys())
        for name, value in plain.state_dict().items():
            torch.testing.assert_close(value, residual.state_dict()[name], rtol=0, atol=0)
        batch = make_history_batch(self.record, [2, 4], depth=3, window=4)
        predicted = plain.predict_hidden(*self.inputs(batch))
        changed = residual.predict_hidden(*self.inputs(batch))
        expected = batch["history"][:, -1:, :].expand(-1, 3, -1)
        torch.testing.assert_close(changed - predicted, expected, rtol=1e-5, atol=2e-7)
        self.assertEqual(sum(p.numel() for p in plain.parameters()), sum(p.numel() for p in residual.parameters()))

    def test_parallel_generate_calls_forward_once_and_truncates_only_output(self):
        model = self.model().eval()
        batch = make_history_batch(self.record, [2, 4], depth=3, window=4)
        expected = model(*self.inputs(batch)).argmax(dim=-1)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            proposed = model.generate_draft(*self.inputs(batch), max_children=2)
            self.assertEqual(forward.call_count, 1)
            torch.testing.assert_close(proposed, expected[:, :2], rtol=0, atol=0)
        with patch.object(model, "forward", wraps=model.forward) as forward:
            empty = model.generate_draft(*self.inputs(batch), max_children=0)
            self.assertEqual(tuple(empty.shape), (2, 0))
            forward.assert_not_called()

    def test_checkpoint_roundtrip_omits_frozen_io_and_preserves_mode(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            frozen_io, path = root / "frozen_io.pt", root / "model.pt"
            torch.save(dict(embedding=self.embedding, lm_head=self.lm_head), frozen_io)
            model = self.model("history_residual").eval()
            self.assertFalse(any(name.startswith("shared_") for name in model.state_dict()))
            torch.save(dict(schema="loop_history_drafter_checkpoint_v1", config=model.config.dict(),
                            model=model.state_dict(), frozen_io=str(frozen_io), cursor=dict(step=7)), path)
            loaded, payload = load_history_checkpoint(path)
            batch = make_history_batch(self.record, [2, 4], depth=3, window=4)
            torch.testing.assert_close(model(*self.inputs(batch)), loaded(*self.inputs(batch)), rtol=0, atol=0)
            self.assertEqual(loaded.config.feature_mode, "history_residual")
            self.assertEqual(payload["cursor"]["step"], 7)
            torch.testing.assert_close(loaded.shared_lm_head, self.lm_head, rtol=0, atol=0)

    def test_rejects_empty_memory_and_invalid_query_count(self):
        model = self.model()
        history = torch.randn(1, 4, 8)
        with self.assertRaisesRegex(ValueError, "at least one"):
            model(history, torch.zeros(1, 4, dtype=torch.bool), torch.tensor([3]))
        with self.assertRaisesRegex(ValueError, "max_children"):
            model.generate_draft(history, torch.ones(1, 4, dtype=torch.bool), torch.tensor([3]), max_children=4)


if __name__ == "__main__":
    unittest.main()
