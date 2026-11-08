"""CPU checks for v1 warmstart, token causality and incremental loop memory."""
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from .draft import Config, LoopDraft
from .loop_memory_draft import (LoopMemoryConfig, LoopMemoryDraft, fork_from_ssd,
                                load_loop_memory_checkpoint, make_loop_memory_batch)


class LoopMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(5)
        self.config = LoopMemoryConfig(target_hidden=8, vocab_size=23, width=8,
            mlp_width=16, layers=2, init_rank=4, prefix_width=3, depth=3,
            state_dim=3, state_head_dim=4, memory_width=8, memory_heads=2, window=4)
        self.embedding = torch.randn(23, 8)
        self.history = torch.randn(2, 4, 4, 8)
        self.mask = torch.tensor([[False, False, True, True], [True, True, True, True]])
        self.taps = self.history[:, -1]
        self.previous = torch.tensor([[2, 4, 6], [3, 5, 7]])
        self.bonus = self.previous[:, 0]

    def model(self, mode="trajectory", active=False):
        torch.manual_seed(11)
        model = LoopMemoryDraft(replace(self.config, memory_mode=mode), self.embedding).eval()
        if active:
            with torch.no_grad():
                model.memory_final_attention.output.weight.normal_(std=.1)
                model.memory_transition_attention.output.weight.normal_(std=.1)
        return model

    def score(self, model, history=None, previous=None):
        return model(self.taps, self.bonus, self.history if history is None else history,
                     self.mask, self.previous if previous is None else previous)

    def test_zero_outputs_preserve_original_ssd_and_seeded_parameters(self):
        base = self.model("none")
        original = LoopDraft(Config(**{k: v for k, v in base.config.dict().items()
                                     if k in Config.__dataclass_fields__}), self.embedding).eval()
        original.load_state_dict({k: v for k, v in base.state_dict().items() if not k.startswith("memory_")})
        expected = original(self.taps, self.bonus, self.previous)
        for mode in ("none", "final", "final_capacity", "trajectory"):
            model = self.model(mode)
            for name, value in model.state_dict().items():
                torch.testing.assert_close(value, base.state_dict()[name], atol=0, rtol=0)
            torch.testing.assert_close(self.score(model), expected, atol=0, rtol=0)
            actual = model.generate_draft(self.taps, self.bonus, self.history, self.mask)
            torch.testing.assert_close(actual, original.generate_draft(self.taps, self.bonus), atol=0, rtol=0)

    def test_teacher_prefix_causality_and_online_head_equivalence(self):
        model = self.model(active=True)
        expected = self.score(model)
        future = self.previous.clone()
        future[:, 2] = 19
        torch.testing.assert_close(self.score(model, previous=future)[:, :2], expected[:, :2], atol=0, rtol=0)
        memory = model.prepare_memory(self.history, self.mask)
        features = model.backbone(self.taps, self.bonus)
        for depth in range(3):
            torch.testing.assert_close(model.step_logits(features, self.previous[:, :depth + 1], memory),
                                       expected[:, depth], atol=2e-6, rtol=2e-6)
        with patch.object(model, "backbone", wraps=model.backbone) as backbone:
            generated = model.generate_draft(self.taps, self.bonus, memory=memory)
            self.assertEqual(backbone.call_count, 1)
        for depth in range(3):
            prefix = torch.cat((self.bonus[:, None], generated[:, :depth]), 1)
            torch.testing.assert_close(generated[:, depth], model.step_logits(features, prefix, memory).argmax(-1))

    def test_masked_nan_padding_and_batch_equals_unpadded(self):
        model = self.model(active=True)
        corrupt = self.history.clone()
        corrupt[~self.mask] = float("nan")
        expected = self.score(model)
        torch.testing.assert_close(self.score(model, history=corrupt), expected, atol=0, rtol=0)
        for i in range(2):
            history = self.history[i:i + 1, self.mask[i]]
            output = model(self.taps[i:i + 1], self.bonus[i:i + 1], history,
                           torch.ones(history.shape[:2], dtype=torch.bool), self.previous[i:i + 1])
            torch.testing.assert_close(output, expected[i:i + 1], atol=2e-6, rtol=2e-6)
        with self.assertRaisesRegex(ValueError, "confirmed token"):
            model.prepare_memory(self.history, torch.zeros_like(self.mask))

    def test_controls_use_only_final_but_trajectory_reads_previous_states(self):
        changed = self.history.clone()
        changed[:, :, :3] = torch.randn_like(changed[:, :, :3]) * 10
        for mode in ("none", "final", "final_capacity"):
            model = self.model(mode, active=True)
            torch.testing.assert_close(self.score(model), self.score(model, history=changed), atol=0, rtol=0)
        model = self.model("trajectory", active=True)
        self.assertGreater(float((self.score(model) - self.score(model, history=changed)).abs().max()), 1e-5)
        final, state, delta = model.normalized_history(self.history, self.mask)
        clean = self.history.masked_fill(~self.mask[..., None, None], 0)
        scale = torch.rsqrt(clean[:, :, -1:].square().mean(-1, keepdim=True) + self.config.normalization_eps)
        torch.testing.assert_close(delta, (clean[:, :, 1:] - clean[:, :, :-1]) * scale)
        torch.testing.assert_close(state, clean[:, :, :-1] * scale)
        capacity = self.model("final_capacity")
        f, s, d = capacity.normalized_history(self.history, self.mask)
        torch.testing.assert_close(s, f[:, :, None].expand_as(s))
        self.assertEqual(int(torch.count_nonzero(d)), 0)

    def test_zero_projection_gets_gradient_then_encoder_learns(self):
        model = self.model()
        labels = torch.tensor([4, 6, 8, 5, 7, 9])
        loss = torch.nn.functional.cross_entropy(self.score(model).flatten(0, 1), labels)
        loss.backward()
        for branch in (model.memory_final_attention, model.memory_transition_attention):
            self.assertGreater(float(branch.output.weight.grad.abs().sum()), 0)
        self.assertIsNone(model.shared_embed.grad)
        optimizer = torch.optim.SGD(model.parameters(), lr=.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        torch.nn.functional.cross_entropy(self.score(model).flatten(0, 1), labels).backward()
        for name, parameter in model.named_parameters():
            if name.startswith("memory_"):
                self.assertIsNotNone(parameter.grad, name)
                self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(parameter.grad.abs().sum()), 0, name)

    def test_incremental_cache_sliding_matches_full_rebuild(self):
        model = self.model(active=True)
        memory = model.prepare_memory(self.history, self.mask)
        addition = torch.randn(2, 2, 4, 8)
        with patch.object(model.memory_transition_encoder, "forward", wraps=model.memory_transition_encoder.forward) as encoder:
            appended = model.append_memory(memory, addition)
            self.assertEqual(encoder.call_count, 1)
            self.assertEqual(encoder.call_args.args[0].shape[1], 2)
        self.assertEqual(memory.mask.shape[1], 4)
        combined = torch.cat((self.history, addition), 1)[:, -4:]
        mask = torch.cat((self.mask, torch.ones(2, 2, dtype=torch.bool)), 1)[:, -4:]
        rebuilt = model.prepare_memory(combined, mask)
        features = model.backbone(addition[:, -1], self.bonus)
        for depth in range(3):
            prefix = self.previous[:, :depth + 1]
            torch.testing.assert_close(model.step_logits(features, prefix, appended),
                                       model.step_logits(features, prefix, rebuilt), atol=2e-6, rtol=2e-6)
        self.assertIs(model.append_memory(memory, addition[:, :0]), memory)
        long = torch.randn(2, 7, 4, 8)
        appended = model.append_memory(memory, long)
        rebuilt = model.prepare_memory(long[:, -4:], torch.ones(2, 4, dtype=torch.bool))
        torch.testing.assert_close(model.step_logits(features, self.bonus[:, None], appended),
                                   model.step_logits(features, self.bonus[:, None], rebuilt), atol=2e-6, rtol=2e-6)

    def test_batch_labels_and_confirmed_only_history(self):
        record = dict(token_ids=torch.tensor([10, 11, 12, 13, 14, 15]), prompt_length=3,
                      round_hidden=torch.arange(5 * 4 * 8).reshape(5, 4, 8).float())
        batch = make_loop_memory_batch(record, [2, 3], 3, 4)
        self.assertEqual(batch["bonus"].tolist(), [13, 14])
        self.assertEqual(batch["previous"].tolist(), [[13, 14, 15], [14, 15, 0]])
        self.assertEqual(batch["labels"].tolist(), [[14, 15, -100], [15, -100, -100]])
        self.assertEqual(batch["history_mask"].tolist(), [[False, True, True, True], [True] * 4])
        changed = {**record, "round_hidden": record["round_hidden"].clone(), "token_ids": record["token_ids"].clone()}
        changed["round_hidden"][3:] = -999
        changed["token_ids"][4:] = 1
        original = make_loop_memory_batch(record, [2], 3, 4)
        altered = make_loop_memory_batch(changed, [2], 3, 4)
        for key in ("history", "history_mask", "taps", "bonus"):
            torch.testing.assert_close(original[key], altered[key], atol=0, rtol=0)
        model = self.model(active=True)
        torch.testing.assert_close(model.forward_batch(original)[:, 0], model.forward_batch(altered)[:, 0], atol=0, rtol=0)

    def test_checkpoint_fork_roundtrip_and_budget(self):
        base = self.model("none")
        base_config = {k: v for k, v in base.config.dict().items() if k in Config.__dataclass_fields__}
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            io, parent_path = directory / "io.pt", directory / "parent.pt"
            torch.save(dict(embedding=self.embedding), io)
            torch.save(dict(schema="loop_drafter_checkpoint_v1", config=base_config,
                            model={k: v for k, v in base.state_dict().items() if not k.startswith("memory_")},
                            frozen_io=str(io), cursor=dict(step=4096, epoch=4), protocol="parent_protocol",
                            settings={"epochs": 4}, data_index=[{"id": "a"}], optimizer={}), parent_path)
            parent_bytes = parent_path.read_bytes()
            all_states = []
            for mode in ("none", "final", "final_capacity", "trajectory"):
                child, payload = fork_from_ssd(parent_path, mode, memory_width=8, memory_heads=2, window=4)
                child.eval()
                all_states.append(child.state_dict())
                self.assertNotIn("optimizer", payload)
                self.assertEqual(payload["cursor"], dict(step=4096, epoch=4))
                self.assertEqual(payload["protocol"], "parent_protocol")
                torch.testing.assert_close(self.score(child), self.score(base), atol=0, rtol=0)
                checkpoint = directory / f"{mode}.pt"
                torch.save(payload, checkpoint)
                loaded, _ = load_loop_memory_checkpoint(checkpoint)
                loaded.eval()
                torch.testing.assert_close(self.score(loaded), self.score(child), atol=0, rtol=0)
            for name in all_states[0]:
                for state in all_states[1:]:
                    torch.testing.assert_close(state[name], all_states[0][name], atol=0, rtol=0)
            self.assertEqual(parent_path.read_bytes(), parent_bytes)
        with torch.device("meta"):
            large = LoopMemoryDraft(LoopMemoryConfig(), torch.empty(49152, 2048))
        self.assertLessEqual(large.memory_parameter_count(), 5_000_000)
        self.assertGreater(large.memory_parameter_count(), 4_000_000)


if __name__ == "__main__":
    unittest.main()
