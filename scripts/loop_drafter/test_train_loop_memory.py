"""CPU checks of scratch/legacy initialization, resume, sampling and CE."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest
from unittest.mock import patch

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from . import train_loop_memory as training
from .draft import Config, LoopDraft
from .loop_memory_draft import (LoopMemoryConfig, LoopMemoryDraft, fork_from_ssd,
                                load_loop_memory_checkpoint, make_loop_memory_batch)
from .train import anchor_positions
from .train_protocol import (Settings, epoch_order, index_training_data, prepare_requests,
                             shard_indices, train_epochs as train_ssd)


def distributed_worker(rank, rendezvous, job_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=2)
    try:
        job = torch.load(job_path, weights_only=True)
        config, settings = LoopMemoryConfig(**job["config"]), Settings(**job["settings"])
        model = LoopMemoryDraft(config, job["embedding"])
        model.load_state_dict(job["initial"])
        order = epoch_order(len(job["entries"]), 0, 4)[:settings.requests_per_update]
        requests = prepare_requests(job["entries"], shard_indices(order, rank, 2), settings, 4)
        stats = training.synchronized_gradients(model, requests, 2, torch.device("cpu"))
        out = Path(job["out"])
        if rank == 0:
            out.mkdir(parents=True, exist_ok=True)
            torch.save(dict(stats=stats, gradients={name: p.grad for name, p in model.named_parameters()}), out / "grads.pt")
        with redirect_stdout(io.StringIO()):
            training.train_epochs(model, job["entries"], settings, out / "full", job["frozen_io"], parent=job["parent"])
            split = LoopMemoryDraft(config, job["embedding"])
            split.load_state_dict(job["initial"])
            training.train_epochs(split, job["entries"], settings, out / "resume", job["frozen_io"],
                                  parent=job["parent"], stop_after_steps=1)
            resumed, payload = load_loop_memory_checkpoint(out / "resume/latest.pt")
            training.train_epochs(resumed, job["entries"], settings, out / "resume", job["frozen_io"], resume_payload=payload)
    finally:
        dist.destroy_process_group()


class LoopMemoryTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(0)
        random.seed(0)
        self.config = LoopMemoryConfig(target_hidden=8, vocab_size=23, width=8, mlp_width=16,
                                       layers=1, init_rank=4, prefix_width=3, depth=3,
                                       state_dim=3, state_head_dim=4, memory_width=8,
                                       memory_heads=2, window=4, memory_mode="trajectory")
        self.embedding = torch.randn(23, 8)
        self.settings = Settings(role="branch", epochs=2, requests_per_update=4,
                                 max_anchors_per_request=4, microbatch=2, peak_lr=5e-5, floor_lr=5e-6)

    def fixture(self, root):
        rows = []
        for i in range(5):
            length = 5 + 7 * i
            record = dict(id=f"request_{i}", source=f"source_{i % 2}", split="train", prompt_length=3,
                          token_ids=torch.randint(0, 23, (length,)), round_hidden=torch.randn(length - 1, 4, 8))
            path = root / f"{i}.pt"
            torch.save(record, path)
            rows.append(dict(path=str(path), id=record["id"], split="train"))
        manifest = root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        frozen_io = root / "frozen_io.pt"
        torch.save(dict(embedding=self.embedding), frozen_io)
        entries = index_training_data(manifest, self.config, 5)
        return entries, frozen_io, manifest

    def baseline(self, root, entries, frozen_io):
        config = Config(target_hidden=8, vocab_size=23, width=8, mlp_width=16, layers=1,
                        init_rank=4, prefix_width=3, depth=3, state_dim=3, state_head_dim=4,
                        source_layout="canonical", feature_mode="final")
        with redirect_stdout(io.StringIO()):
            train_ssd(LoopDraft(config, self.embedding), entries,
                      replace(self.settings, role="baseline", epochs=4, peak_lr=1e-4, floor_lr=1e-5),
                      root / "baseline", frozen_io)
        return root / "baseline/model.pt"

    def quiet(self, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return training.train_epochs(*args, **kwargs)

    def rows(self, root):
        with (root / "metrics.jsonl").open() as stream:
            return [json.loads(line) for line in stream]

    def nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.nested_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_global_label_normalization_with_teacher_prefixes(self):
        with tempfile.TemporaryDirectory() as temporary:
            entries, _, _ = self.fixture(Path(temporary))
            requests = prepare_requests(entries, [0, 1, 4], self.settings, 4)
            for mode in ("none", "final", "final_capacity", "trajectory"):
                with self.subTest(mode=mode):
                    model = LoopMemoryDraft(replace(self.config, memory_mode=mode), self.embedding)
                    reference = deepcopy(model)
                    actual = training.accumulate_gradients(model, requests, 2, torch.device("cpu"))
                    batches = [make_loop_memory_batch(record, roots, 3, 4) for _, record, roots in requests]
                    combined = {key: torch.cat([b[key] for b in batches]) for key in batches[0]}
                    logits = reference.forward_batch(combined)
                    loss = F.cross_entropy(logits.flatten(0, 1), combined["labels"].flatten(), ignore_index=-100)
                    loss.backward()
                    self.assertAlmostEqual(actual["loss"], loss.item(), places=6)
                    self.assertEqual(actual["valid_labels"], int(combined["valid"].sum()))
                    for (name, p), (_, ref) in zip(model.named_parameters(), reference.named_parameters()):
                        if p.grad is None or ref.grad is None:
                            self.assertIs(p.grad, ref.grad, name)
                        else:
                            torch.testing.assert_close(p.grad, ref.grad, rtol=3e-5, atol=3e-7, msg=name)

    def test_cli_four_arms_share_parent_fresh_optimizer_and_epochs_four_five(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, manifest = self.fixture(root)
            baseline = self.baseline(root, entries, frozen_io)
            parent = torch.load(baseline, weights_only=True)
            observed = []
            original = training.prepare_requests

            def inspect(*args, **kwargs):
                prepared = original(*args, **kwargs)
                observed.append([(entry["id"], roots) for entry, _, roots in prepared])
                return prepared

            expected = []
            for epoch in (4, 5):
                order = epoch_order(5, 0, epoch)
                for start in (0, 4):
                    prepared = prepare_requests(entries, order[start:start + 4], self.settings, epoch)
                    expected.append([(entry["id"], roots) for entry, _, roots in prepared])
            for mode in ("none", "final", "final_capacity", "trajectory"):
                observed.clear()
                argv = ["--data", str(manifest), "--frozen-io", str(frozen_io), "--out", str(root / mode),
                        "--memory-mode", mode, "--init-checkpoint", str(baseline), "--expected-requests", "5",
                        "--index-file", str(root / "baseline/data_index.json"), "--requests-per-update", "4",
                        "--max-anchors-per-request", "4", "--microbatch", "2", "--memory-width", "8",
                        "--memory-heads", "2", "--window", "4", "--num-threads", "1"]
                with patch.object(training, "prepare_requests", side_effect=inspect), redirect_stdout(io.StringIO()):
                    result = training.main(argv)
                self.assertEqual(result["event"], "complete")
                self.assertEqual(observed, expected)
                initial = torch.load(root / mode / "initial.pt", weights_only=True)
                self.assertEqual(initial["optimizer"]["state"], {})
                self.assertEqual(initial["cursor"]["step"], 0)
                self.assertEqual(initial["parent"]["epoch"], 4)
                self.assertEqual(initial["settings"]["peak_lr"], 5e-5)
                self.assertEqual(initial["sampling_epoch_offset"], 4)
                for name, value in parent["model"].items():
                    torch.testing.assert_close(initial["model"][name], value, rtol=0, atol=0, msg=name)
                rows = self.rows(root / mode)
                self.assertEqual([row["sampling_epoch"] for row in rows], [4, 4, 5, 5])
                self.assertEqual(rows[-1]["lr"], 5e-6)
                self.assertTrue((root / mode / "epoch2.pt").exists())
                with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "sampling epoch"):
                    training.main(argv + ["--resume", "--sampling-epoch-offset", "0"])

    def test_resume_matches_uninterrupted_and_preserves_sampling_offset(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, _ = self.fixture(root)
            parent = dict(checkpoint="ssd/model.pt", epoch=4, step=8)
            model = LoopMemoryDraft(self.config, self.embedding)
            initial = deepcopy(model.state_dict())
            torch_rng, python_rng = torch.get_rng_state(), random.getstate()
            self.quiet(model, entries, self.settings, root / "full", frozen_io, parent=parent)
            split = LoopMemoryDraft(self.config, self.embedding)
            split.load_state_dict(initial)
            torch.set_rng_state(torch_rng)
            random.setstate(python_rng)
            self.quiet(split, entries, self.settings, root / "resume", frozen_io, parent=parent, stop_after_steps=3)
            resumed, payload = load_loop_memory_checkpoint(root / "resume/latest.pt")
            with (root / "resume/metrics.jsonl").open("a") as stream:
                stream.write('{"event":"update","step":4,"loss":999}\n{"event":')
            self.quiet(resumed, entries, self.settings, root / "resume", frozen_io, resume_payload=payload)
            a = torch.load(root / "full/model.pt", weights_only=True)
            b = torch.load(root / "resume/model.pt", weights_only=True)
            for key in ("model", "optimizer", "cursor", "torch_rng", "python_rng", "parent"):
                self.nested_equal(a[key], b[key])
            strip = lambda rows: [{k: v for k, v in row.items() if k != "update_seconds"} for row in rows]
            self.assertEqual(strip(self.rows(root / "full")), strip(self.rows(root / "resume")))

    def test_scratch_initialization_has_identical_weights_predictions_and_frozen_target_io(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, _ = self.fixture(root)
            _, record, roots = prepare_requests(entries, [4], self.settings, 0)[0]
            batch = make_loop_memory_batch(record, roots, self.config.depth, self.config.window)
            initial, prediction, initialization_hash = None, None, None
            for mode in ("none", "final", "final_capacity", "trajectory"):
                with self.subTest(mode=mode), patch.object(training, "fork_from_ssd", side_effect=AssertionError("old drafter read")):
                    model = training.initialize_from_scratch(replace(self.config, memory_mode=mode), frozen_io, seed=0)
                    self.assertFalse(model.shared_embed.requires_grad)
                    self.assertFalse(model.shared_lm_head.requires_grad)
                    self.assertNotIn("shared_embed", model.state_dict())
                    torch.testing.assert_close(model.shared_embed, self.embedding, rtol=0, atol=0)
                    self.assertTrue(model.memory_final_attention.output.weight.eq(0).all())
                    self.assertTrue(model.memory_transition_attention.output.weight.eq(0).all())
                    with torch.no_grad():
                        logits = model.forward_batch(batch)
                    if initial is None:
                        initial, prediction = deepcopy(model.state_dict()), logits
                        initialization_hash = training.initial_state_hash(initial)
                    else:
                        self.nested_equal(initial, model.state_dict())
                        torch.testing.assert_close(prediction, logits, rtol=0, atol=0)
                        self.assertEqual(training.initial_state_hash(model.state_dict()), initialization_hash)

    def scratch_argv(self, manifest, frozen_io, out):
        result = ["--data", str(manifest), "--frozen-io", str(frozen_io), "--out", str(out),
                  "--memory-mode", "trajectory", "--from-scratch", "--expected-requests", "5",
                  "--requests-per-update", "4", "--max-anchors-per-request", "4", "--microbatch", "2",
                  "--memory-width", "8", "--memory-heads", "2", "--window", "4", "--num-threads", "1"]
        for name in training.BACKBONE_KEYS:
            result += ["--" + name.replace("_", "-"), str(getattr(self.config, name))]
        return result

    def test_scratch_cli_four_epochs_zero_offset_and_same_run_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, manifest = self.fixture(root)
            full_argv = self.scratch_argv(manifest, frozen_io, root / "scratch_full")
            split_argv = self.scratch_argv(manifest, frozen_io, root / "scratch_split")
            with patch.object(training, "fork_from_ssd", side_effect=AssertionError("old drafter read")), redirect_stdout(io.StringIO()):
                result = training.main(full_argv)
                paused = training.main(split_argv + ["--stop-after-steps", "3"])
                self.assertEqual(paused["step"], 3)
                training.main(split_argv + ["--resume"])
            self.assertEqual(result["total_steps"], 8)
            initial = torch.load(root / "scratch_full/initial.pt", weights_only=True)
            self.assertEqual(initial["protocol"], training.SCRATCH_PROTOCOL)
            self.assertTrue(initial["from_scratch"])
            self.assertIsNone(initial["parent"])
            self.assertEqual(initial["optimizer"]["state"], {})
            self.assertEqual(initial["cursor"]["step"], 0)
            self.assertEqual(initial["settings"]["role"], "baseline")
            self.assertEqual(initial["settings"]["epochs"], 4)
            self.assertEqual(initial["settings"]["peak_lr"], 1e-4)
            self.assertEqual(initial["settings"]["floor_lr"], 1e-5)
            self.assertEqual(initial["sampling_epoch_offset"], 0)
            self.assertEqual(initial["initialization"], dict(kind="from_scratch", seed=0,
                             initial_state_sha256=training.initial_state_hash(initial["model"])))
            config = json.loads((root / "scratch_full/config.json").read_text())
            self.assertEqual(config["initialization"], initial["initialization"])
            rows = self.rows(root / "scratch_full")
            self.assertEqual([r["sampling_epoch"] for r in rows], [0, 0, 1, 1, 2, 2, 3, 3])
            self.assertEqual(rows[-1]["lr"], 1e-5)
            full = torch.load(root / "scratch_full/model.pt", weights_only=True)
            split = torch.load(root / "scratch_split/model.pt", weights_only=True)
            for key in ("model", "optimizer", "cursor", "torch_rng", "python_rng", "parent", "from_scratch"):
                self.nested_equal(full[key], split[key])
            self.assertEqual(full["initialization"], initial["initialization"])
            self.assertEqual(split["initialization"], initial["initialization"])
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "initialization mode"):
                training.main([v for v in split_argv if v != "--from-scratch"] + ["--resume"])
            with redirect_stdout(io.StringIO()), self.assertRaisesRegex(ValueError, "sampling epoch"):
                training.main(split_argv + ["--resume", "--sampling-epoch-offset", "4"])
            resumed, payload = load_loop_memory_checkpoint(root / "scratch_split/latest.pt")
            with self.assertRaisesRegex(ValueError, "same run directory"):
                self.quiet(resumed, entries, Settings(**payload["settings"]), root / "copied_run", frozen_io,
                           resume_payload=payload, from_scratch=True)
            corrupted = deepcopy(payload)
            corrupted["parent"] = {"checkpoint": "old.pt"}
            with self.assertRaisesRegex(ValueError, "must not have a drafter parent"):
                training.validate_initialization(corrupted, True)
            corrupted = deepcopy(payload)
            del corrupted["initialization"]
            with self.assertRaisesRegex(ValueError, "initialization identity"):
                training.validate_initialization(corrupted, True)

    def test_new_initialization_mode_is_explicit_and_mutually_exclusive(self):
        base = ["--data", "manifest.jsonl", "--frozen-io", "io.pt", "--out", "new",
                "--memory-mode", "none"]
        with self.assertRaisesRegex(ValueError, "explicit --from-scratch"):
            training.main(base)
        with self.assertRaises(SystemExit), patch("sys.stderr", io.StringIO()):
            training.main(base + ["--from-scratch", "--init-checkpoint", "old.pt"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU gloo unavailable")
    def test_two_rank_global_gradient_and_empty_tail_resume(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, _ = self.fixture(root)
            model = LoopMemoryDraft(self.config, self.embedding)
            initial = deepcopy(model.state_dict())
            parent = dict(checkpoint="ssd/model.pt", epoch=4, step=8)
            order = epoch_order(5, 0, 4)[:4]
            stats = training.accumulate_gradients(model, prepare_requests(entries, order, self.settings, 4),
                                                   2, torch.device("cpu"))
            gradients = {name: None if p.grad is None else p.grad.clone() for name, p in model.named_parameters()}
            self.quiet(model, entries, self.settings, root / "single", frozen_io, parent=parent)
            job = dict(config=self.config.dict(), settings=asdict(self.settings), initial=initial,
                       embedding=self.embedding, entries=entries, frozen_io=str(frozen_io), parent=parent,
                       out=str(root / "two"))
            torch.save(job, root / "job.pt")
            mp.spawn(distributed_worker, args=(str(root / "gloo"), str(root / "job.pt")), nprocs=2, join=True)
            actual = torch.load(root / "two/grads.pt", weights_only=True)
            self.assertEqual(actual["stats"]["labels_by_depth"], stats["labels_by_depth"])
            for name, value in actual["gradients"].items():
                if gradients[name] is None:
                    self.assertTrue(value.eq(0).all())
                else:
                    torch.testing.assert_close(value, gradients[name], rtol=3e-5, atol=3e-7, msg=name)
            full = torch.load(root / "two/full/model.pt", weights_only=True)
            resumed = torch.load(root / "two/resume/model.pt", weights_only=True)
            single = torch.load(root / "single/model.pt", weights_only=True)
            for key in ("model", "optimizer", "cursor"):
                self.nested_equal(full[key], resumed[key])
            for name, value in full["model"].items():
                torch.testing.assert_close(value, single["model"][name], rtol=3e-5, atol=3e-7, msg=name)
            self.assertEqual([r["requests_by_rank"] for r in self.rows(root / "two/full")],
                             [[2, 2], [1, 0], [2, 2], [1, 0]])


if __name__ == "__main__":
    unittest.main()
