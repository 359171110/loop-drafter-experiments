"""CPU checks of the temporal contract, feature controls, and reusable heads."""
from dataclasses import replace
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest

import torch

from .draft import Config, LoopDraft, fork_from_final, load_checkpoint
from .train import anchor_positions, make_batch, prompt_group, save_checkpoint, split_paths


class LoopDraftTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(2)
        self.config = Config(target_hidden=8, vocab_size=23, rounds=4, width=8,
                             mlp_width=16, layers=2, init_rank=4, prefix_width=3,
                             depth=3, state_dim=3, state_head_dim=4)
        self.embedding = torch.randn(23, 8)

    def test_equal_projection_budget_and_difference_magnitude(self):
        models = [LoopDraft(replace(self.config, feature_mode=mode), self.embedding)
                  for mode in ("final", "multiloop", "trajectory")]
        self.assertEqual(len({sum(p.numel() for p in m.parameters()) for m in models}), 1)
        h = torch.ones(1, 4, 8)
        h[:, 0] = .7
        h[:, 1] = .8
        h[:, 2] = .9
        features = models[-1].source_features(h)
        torch.testing.assert_close(features[:, 1:], features[:, :1].expand(-1, 3, -1) * .1)
        altered = h.clone()
        altered[:, :3] = -20
        torch.testing.assert_close(models[0].source_features(h), models[0].source_features(altered))

    def test_full_prefix_causality_and_head_equivalence(self):
        model = LoopDraft(self.config, self.embedding).eval()
        taps = torch.randn(2, 4, 8)
        previous = torch.tensor([[2, 4, 6], [3, 5, 7]])
        bonus = previous[:, 0]
        original = model(taps, bonus, previous)
        changed = previous.clone()
        changed[:, 2] = 19
        torch.testing.assert_close(original[:, :2], model(taps, bonus, changed)[:, :2])
        features = model.backbone(taps, bonus)
        for depth in range(3):
            torch.testing.assert_close(original[:, depth], model.step_logits(features, previous[:, :depth + 1]))
        generated = model.generate_draft(taps, bonus)
        self.assertEqual(tuple(generated.shape), (2, 3))
        for depth in range(3):
            prefix = torch.cat((bonus[:, None], generated[:, :depth]), 1)
            torch.testing.assert_close(generated[:, depth], model.step_logits(features, prefix).argmax(-1))

    def test_anchor_boundary_tail_and_future_hidden(self):
        record = dict(token_ids=torch.tensor([10, 11, 12, 13, 14, 15]), prompt_length=3,
                      round_hidden=torch.arange(5 * 4 * 8).view(5, 4, 8).float())
        self.assertEqual(anchor_positions(record), [2, 3])
        batch = make_batch(record, [2, 3], depth=3)
        self.assertEqual(batch["bonus"].tolist(), [13, 14])
        self.assertEqual(batch["labels"].tolist(), [[14, 15, -100], [15, -100, -100]])
        self.assertEqual(batch["previous"].tolist(), [[13, 14, 15], [14, 15, 0]])
        torch.testing.assert_close(batch["taps"], record["round_hidden"][[2, 3]])
        changed = dict(record, round_hidden=record["round_hidden"].clone())
        changed["round_hidden"][3:] = -999
        torch.testing.assert_close(make_batch(record, [2], 3)["taps"], make_batch(changed, [2], 3)["taps"])

    def test_gradients_reach_drafter_but_not_frozen_target(self):
        embedding = self.embedding.clone().requires_grad_(True)
        model = LoopDraft(self.config, embedding)
        previous = torch.tensor([[2, 4, 6], [3, 5, 7]])
        logits = model(torch.randn(2, 4, 8), previous[:, 0], previous)
        torch.nn.functional.cross_entropy(logits.flatten(0, 1), torch.tensor([4, 6, 8, 5, 7, 9])).backward()
        self.assertIsNone(embedding.grad)
        for name, parameter in model.named_parameters():
            self.assertIsNotNone(parameter.grad, name)
            self.assertTrue(torch.isfinite(parameter.grad).all(), name)

    def test_checkpoint_roundtrip_and_prompt_grouped_split(self):
        model = LoopDraft(self.config, self.embedding).eval()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            io = directory / "io.pt"
            torch.save(dict(embedding=self.embedding), io)
            path = directory / "checkpoint.pt"
            optimizer = torch.optim.AdamW(model.parameters())
            save_checkpoint(path, model, optimizer, dict(step=0, epoch=0, next_request=0),
                            SimpleNamespace(frozen_io=str(io)), dict(train=[], validation=[]))
            loaded, payload = load_checkpoint(path)
            self.assertEqual(payload["cursor"], dict(step=0, epoch=0, next_request=0))
            taps, previous = torch.randn(2, 4, 8), torch.tensor([[2, 4, 6], [3, 5, 7]])
            torch.testing.assert_close(model(taps, previous[:, 0], previous),
                                       loaded(taps, previous[:, 0], previous))
            paths = []
            for i, ids in enumerate(([1, 2, 3, 4], [1, 2, 5, 6], [3, 4, 5, 6], [5, 6, 7, 8])):
                p = directory / f"record_{i}.pt"
                torch.save(dict(token_ids=torch.tensor(ids), prompt_length=2), p)
                paths.append(p)
            train, valid = split_paths(paths, .5, 7)
            keys = lambda ps: {prompt_group(torch.load(p, weights_only=True)) for p in ps}
            self.assertFalse(keys(train) & keys(valid))
            self.assertEqual(split_paths(paths, .5, 7), (train, valid))
            declared = {str(p): ("eval" if p == paths[-1] else "train") for p in paths}
            self.assertEqual(split_paths(paths, .9, 123, declared), (paths[:-1], paths[-1:]))
            declared[str(paths[0])] = "eval"
            with self.assertRaisesRegex(ValueError, "same prompt"):
                split_paths(paths, .9, 123, declared)

    def test_canonical_slots_share_final_rms_and_zero_initial_projection(self):
        taps = torch.randn(2, 4, 8)
        taps[:, -1] *= 4
        scale = torch.rsqrt(taps[:, -1:].square().mean(-1, keepdim=True)
                            + self.config.normalization_eps) / 2
        expected = {
            "final": torch.cat((taps[:, -1:], torch.zeros_like(taps[:, :-1])), 1),
            "multiloop": torch.cat((taps[:, -1:], taps[:, :-1]), 1),
            "trajectory": torch.cat((taps[:, -1:], taps[:, 1:] - taps[:, :-1]), 1),
        }
        for mode in expected:
            model = LoopDraft(replace(self.config, source_layout="canonical", feature_mode=mode), self.embedding)
            torch.testing.assert_close(model.source_features(taps), expected[mode] * scale, atol=0, rtol=0)
            self.assertEqual(int(torch.count_nonzero(model.source_rank.weight[:, 8:])), 0)
        self.assertEqual(Config().source_layout, "legacy")
        with self.assertRaisesRegex(ValueError, "four rounds"):
            replace(self.config, source_layout="canonical", rounds=3)

    def test_canonical_forks_equal_parent_and_have_independent_storage(self):
        config = replace(self.config, source_layout="canonical", feature_mode="final", normalization_eps=3e-5)
        parent = LoopDraft(config, self.embedding).eval()
        with torch.no_grad():
            parent.source_weight.copy_(torch.linspace(.3, 1.7, 8))
            # Unused final blocks can contain arbitrary old optimizer residue.
            # All forks clear them while retaining the parent's function.
            parent.source_rank.weight[:, 8:].normal_()
            parent.head_proj.weight.add_(.15)
        taps, previous = torch.randn(2, 4, 8), torch.tensor([[2, 4, 6], [3, 5, 7]])
        expected = parent(taps, previous[:, 0], previous)
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            io, path = directory / "io.pt", directory / "parent.pt"
            torch.save(dict(embedding=self.embedding), io)
            save_checkpoint(path, parent, torch.optim.AdamW(parent.parameters()),
                            dict(step=11, epoch=2, next_request=3),
                            SimpleNamespace(frozen_io=str(io)), dict(train=[], validation=[]))
            original_bytes = path.read_bytes()
            children = []
            for mode in ("final", "multiloop", "trajectory"):
                child, payload = fork_from_final(path, mode)
                child.eval()
                children.append(child)
                self.assertEqual(child.config.feature_mode, mode)
                self.assertEqual(child.config.source_layout, "canonical")
                self.assertEqual(child.config.normalization_eps, parent.config.normalization_eps)
                self.assertEqual(payload["cursor"], dict(step=11, epoch=2, next_request=3))
                self.assertEqual(payload["fork"]["parent_cursor"]["step"], 11)
                self.assertNotIn("optimizer", payload)
                torch.testing.assert_close(child.source_weight, parent.source_weight, atol=0, rtol=0)
                self.assertEqual(int(torch.count_nonzero(child.source_rank.weight[:, 8:])), 0)
                torch.testing.assert_close(child(taps, previous[:, 0], previous), expected, atol=0, rtol=0)
                child_path = directory / f"{mode}.pt"
                torch.save(payload, child_path)
                loaded, _ = load_checkpoint(child_path)
                torch.testing.assert_close(loaded(taps, previous[:, 0], previous), expected, atol=0, rtol=0)
                if mode != "final":
                    with self.assertRaisesRegex(ValueError, "canonical final parent"):
                        fork_from_final(child_path, "trajectory")
            for name in dict(parent.named_parameters()):
                parameters = [dict(model.named_parameters())[name] for model in [parent, *children]]
                self.assertEqual(len({parameter.data_ptr() for parameter in parameters}), 4, name)
            self.assertEqual(len({model.shared_embed.data_ptr() for model in [parent, *children]}), 4)
            self.assertEqual(path.read_bytes(), original_bytes)

    def test_canonical_extra_projection_learns_only_when_features_exist(self):
        taps = torch.randn(2, 4, 8)
        previous = torch.tensor([[2, 4, 6], [3, 5, 7]])
        labels = torch.tensor([4, 6, 8, 5, 7, 9])
        base = LoopDraft(replace(self.config, source_layout="canonical", feature_mode="final"), self.embedding)
        for mode in ("final", "multiloop", "trajectory"):
            model = LoopDraft(replace(base.config, feature_mode=mode), self.embedding)
            model.load_state_dict(base.state_dict())
            logits = model(taps, previous[:, 0], previous)
            torch.nn.functional.cross_entropy(logits.flatten(0, 1), labels).backward()
            extra = model.source_rank.weight.grad[:, 8:]
            self.assertTrue(torch.isfinite(extra).all())
            self.assertGreater(float(model.source_rank.weight.grad[:, :8].abs().sum()), 0)
            if mode == "final":
                self.assertEqual(float(extra.abs().sum()), 0)
            else:
                for block in extra.split(8, dim=1):
                    self.assertGreater(float(block.abs().sum()), 0)

    def test_canonical_modes_preserve_full_prefix_causality(self):
        taps = torch.randn(2, 4, 8)
        previous = torch.tensor([[2, 4, 6], [3, 5, 7]])
        for mode in ("final", "multiloop", "trajectory"):
            model = LoopDraft(replace(self.config, source_layout="canonical", feature_mode=mode), self.embedding).eval()
            # Exercise causality after additional source information is active.
            with torch.no_grad():
                model.source_rank.weight[:, 8:].normal_(std=.02)
            original = model(taps, previous[:, 0], previous)
            changed = previous.clone()
            changed[:, 2] = 19
            torch.testing.assert_close(original[:, :2], model(taps, previous[:, 0], changed)[:, :2], atol=0, rtol=0)
            features = model.backbone(taps, previous[:, 0])
            for depth in range(3):
                torch.testing.assert_close(original[:, depth], model.step_logits(features, previous[:, :depth + 1]))

    def test_legacy_checkpoint_without_layout_loads_and_cannot_be_forked(self):
        model = LoopDraft(replace(self.config, feature_mode="final"), self.embedding).eval()
        with tempfile.TemporaryDirectory() as directory:
            directory = Path(directory)
            io, path = directory / "io.pt", directory / "legacy.pt"
            torch.save(dict(embedding=self.embedding), io)
            config = model.config.dict()
            config.pop("source_layout")
            torch.save(dict(schema="loop_drafter_checkpoint_v1", config=config,
                            model=model.state_dict(), frozen_io=str(io)), path)
            loaded, _ = load_checkpoint(path)
            self.assertEqual(loaded.config.source_layout, "legacy")
            taps, previous = torch.randn(2, 4, 8), torch.tensor([[2, 4, 6], [3, 5, 7]])
            torch.testing.assert_close(model(taps, previous[:, 0], previous),
                                       loaded(taps, previous[:, 0], previous), atol=0, rtol=0)
            with self.assertRaisesRegex(ValueError, "canonical final parent"):
                fork_from_final(path, "trajectory")


if __name__ == "__main__":
    unittest.main()
