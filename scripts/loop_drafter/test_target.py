"""CPU checks using the checkpoint's real Ouro architecture with tiny weights."""

from __future__ import annotations

import copy
import os
from pathlib import Path
import unittest

import torch
from transformers import AutoConfig

try:
    from .target import DEFAULT_MODEL, OuroTarget, ouro_model_class, validate_target
except ImportError:
    from target import DEFAULT_MODEL, OuroTarget, ouro_model_class, validate_target


class OuroTargetTest(unittest.TestCase):
    """Exercise actual recurrent attention and all logical KV slots on CPU."""

    @classmethod
    def setUpClass(cls) -> None:
        model_path = os.environ.get("OURO_TEST_MODEL", DEFAULT_MODEL)
        if "OURO_TEST_MODEL" not in os.environ and not Path(model_path).is_dir():
            raise unittest.SkipTest("Set OURO_TEST_MODEL to a local Ouro checkpoint to test its model code")
        torch.set_num_threads(2)
        torch.manual_seed(43)
        config = AutoConfig.from_pretrained(model_path, trust_remote_code=True, local_files_only=True)
        config.hidden_size = 32
        config.intermediate_size = 64
        config.num_hidden_layers = 2
        config.num_attention_heads = 4
        config.num_key_value_heads = 4
        config.head_dim = 8
        config.vocab_size = 37
        config.max_position_embeddings = 128
        config.layer_types = ["full_attention"] * 2
        config.pad_token_id = 0
        config._attn_implementation = "eager"
        cls.target = OuroTarget(ouro_model_class(model_path)(config).float())

    def test_serial_block_rejection_and_native_head(self) -> None:
        report = validate_target(
            self.target, torch.tensor([[2, 5, 8, 13], [4, 1, 9, 7]]),
            new_tokens=7, block_size=3, atol=2e-5, rtol=2e-5,
        )
        self.assertTrue(report["passed"], report)
        self.assertEqual(report["rows"][0]["round_hidden_shape"], [1, 4, 4, 32])
        self.assertEqual(report["rows"][0]["accepted_before_rejection"], 1)

    def test_round_states_are_exact_checkpoint_outputs(self) -> None:
        ids = torch.tensor([[1, 4, 7]])
        before = {name: value.clone() for name, value in self.target.model.named_parameters()}
        wrapped = self.target(ids)
        with torch.no_grad():
            _, states, _ = self.target.model.model(
                input_ids=ids, past_key_values=self.target.new_cache(), use_cache=True,
            )
        torch.testing.assert_close(wrapped.round_hidden, torch.stack(states, 2), atol=0, rtol=0)
        for name, value in self.target.model.named_parameters():
            torch.testing.assert_close(value, before[name], atol=0, rtol=0)
            self.assertFalse(value.requires_grad)

    def test_all_rounds_crop_to_zero_and_restart(self) -> None:
        ids = torch.tensor([[2, 3, 4]])
        first = self.target(ids)
        self.assertEqual(self.target.cache_lengths(first.cache), [3] * 8)
        self.target.crop_cache(first.cache, 0)
        second = self.target(ids, first.cache)
        torch.testing.assert_close(first.logits, second.logits, atol=0, rtol=0)
        with self.assertRaises(ValueError):
            self.target.crop_cache(second.cache, 4)

    def test_rejects_an_actual_round_count_change(self) -> None:
        self.target.model.model.total_ut_steps = 3
        try:
            with self.assertRaises(RuntimeError):
                self.target(torch.tensor([[2, 3]]))
        finally:
            self.target.model.model.total_ut_steps = 4

    def test_equal_length_batch_has_independent_cache_rows(self) -> None:
        ids = torch.tensor([[2, 4, 6], [3, 5, 7]])
        batch = self.target(ids)
        expected = torch.cat([self.target(row[None]).logits for row in ids], dim=0)
        torch.testing.assert_close(batch.logits, expected, atol=2e-5, rtol=2e-5)

    def test_sdpa_serial_block_and_recovery(self) -> None:
        config = copy.deepcopy(self.target.model.config)
        config._attn_implementation = "sdpa"
        target = OuroTarget(type(self.target.model)(config).float())
        report = validate_target(target, [torch.tensor([[2, 5, 8, 13]])],
                                 new_tokens=5, block_size=3, atol=2e-5, rtol=2e-5)
        self.assertEqual(report["attention_backend"], "sdpa")
        self.assertTrue(report["passed"], report)

    def test_changed_future_proposal_cannot_affect_earlier_positions(self) -> None:
        config = copy.deepcopy(self.target.model.config)
        config._attn_implementation = "sdpa"
        target = OuroTarget(type(self.target.model)(config).to(torch.bfloat16))
        prefill = target(torch.tensor([[1, 2, 3, 4, 5]]))
        left = target(torch.tensor([[6, 7, 8, 9]]), target.clone_cache(prefill.cache))
        right = target(torch.tensor([[6, 7, 16, 19]]), target.clone_cache(prefill.cache))
        torch.testing.assert_close(left.logits[:, :2], right.logits[:, :2], atol=0, rtol=0)
        torch.testing.assert_close(left.round_hidden[:, :2], right.round_hidden[:, :2], atol=0, rtol=0)


if __name__ == "__main__":
    unittest.main()
