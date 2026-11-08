"""CPU checks for request batching, label weighting, full epochs and resumption."""
from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import asdict, replace
import io
import json
from pathlib import Path
import random
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from .draft import Config, LoopDraft, fork_from_final, load_checkpoint
from .train import anchor_positions, make_batch
from .train_protocol import (Settings, accumulate_gradients, epoch_order, index_training_data,
                             learning_rate, main, prepare_requests, shard_indices,
                             synchronized_gradients, train_epochs)


def distributed_training_worker(rank, world_size, rendezvous, job_path):
    """Spawned CPU worker: uneven microbatches, empty tail ranks, and resumption."""
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        job = torch.load(job_path, weights_only=True)
        config, settings = Config(**job["config"]), Settings(**job["settings"])
        root = Path(job["out"])
        model = LoopDraft(config, job["embedding"])
        model.load_state_dict(job["initial"])
        indices = epoch_order(len(job["entries"]), settings.seed, 0)[:settings.requests_per_update]
        local = shard_indices(indices, rank, world_size)
        requests = prepare_requests(job["entries"], local, settings, epoch=0)
        stats = synchronized_gradients(model, requests, settings.microbatch, torch.device("cpu"))
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            torch.save(dict(stats=stats, grads={name: p.grad for name, p in model.named_parameters()}), root / "gradients.pt")
        torch.set_rng_state(job["torch_rng"])
        random.setstate(job["python_rng"])
        with redirect_stdout(io.StringIO()):
            train_epochs(model, job["entries"], settings, root / "full", job["frozen_io"], save_every=1)
        states = [None] * world_size
        dist.all_gather_object(states, model.state_dict())
        for state in states:
            for name, value in state.items():
                if not torch.equal(value, states[0][name]):
                    raise AssertionError(f"ranks diverged after optimizer updates: {name}")
        split_model = LoopDraft(config, job["embedding"])
        split_model.load_state_dict(job["initial"])
        torch.set_rng_state(job["torch_rng"])
        random.setstate(job["python_rng"])
        with redirect_stdout(io.StringIO()):
            train_epochs(split_model, job["entries"], settings, root / "resumed", job["frozen_io"],
                         stop_after_steps=1, save_every=1)
            resumed, payload = load_checkpoint(root / "resumed/latest.pt")
            train_epochs(resumed, job["entries"], settings, root / "resumed", job["frozen_io"],
                         resume_payload=payload, save_every=1)
    finally:
        dist.destroy_process_group()


class TrainingProtocolTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        random.seed(19)
        self.config = Config(target_hidden=8, vocab_size=23, rounds=4,
                             feature_mode="final", source_layout="canonical",
                             width=8, mlp_width=16, layers=1, init_rank=4,
                             prefix_width=3, depth=3, state_dim=3, state_head_dim=4)
        self.embedding = torch.randn(23, 8)
        self.settings = Settings(epochs=3, requests_per_update=4,
                                 max_anchors_per_request=4, microbatch=2)

    def record(self, index, generated_length, split="train"):
        generator = torch.Generator().manual_seed(100 + index)
        length = 3 + generated_length
        return dict(id=f"request_{index}", source=f"source_{index % 3}", split=split,
                    token_ids=torch.randint(0, 23, (length,), generator=generator), prompt_length=3,
                    round_hidden=torch.randn(length - 1, 4, 8, generator=generator))

    def dataset(self, directory, count=11):
        directory = Path(directory)
        rows = []
        for index in range(count):
            path = directory / f"record_{index}.pt"
            record = self.record(index, 2 + index % 7)
            torch.save(record, path)
            rows.append(dict(path=str(path), id=record["id"], split="train"))
        manifest = directory / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        frozen_io = directory / "frozen_io.pt"
        torch.save(dict(embedding=self.embedding), frozen_io)
        entries = index_training_data(manifest, self.config, expected_requests=count)
        return entries, frozen_io, manifest

    def run_quiet(self, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return train_epochs(*args, **kwargs)

    def assert_nested_equal(self, left, right):
        if isinstance(left, torch.Tensor):
            torch.testing.assert_close(left, right, rtol=0, atol=0)
        elif isinstance(left, dict):
            self.assertEqual(left.keys(), right.keys())
            for key in left:
                self.assert_nested_equal(left[key], right[key])
        elif isinstance(left, (list, tuple)):
            self.assertEqual(len(left), len(right))
            for a, b in zip(left, right):
                self.assert_nested_equal(a, b)
        else:
            self.assertEqual(left, right)

    def test_global_label_normalization_matches_one_concatenated_loss(self):
        model = LoopDraft(self.config, self.embedding)
        reference = deepcopy(model)
        requests = []
        for index, length in enumerate((2, 4, 10)):
            record = self.record(index, length)
            roots = anchor_positions(record, max_anchors=5, seed=index)
            requests.append((dict(id=str(index)), record, roots))
        stats = accumulate_gradients(model, requests, microbatch=2, device=torch.device("cpu"))
        batches = [make_batch(record, roots, self.config.depth) for _, record, roots in requests]
        combined = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
        logits = reference(combined["taps"], combined["bonus"], combined["previous"])
        loss = F.cross_entropy(logits.flatten(0, 1), combined["labels"].flatten(), ignore_index=-100)
        loss.backward()
        self.assertEqual(stats["valid_labels"], int(combined["valid"].sum()))
        self.assertNotEqual(stats["request_labels"][0], stats["request_labels"][-1])
        self.assertAlmostEqual(stats["loss"], loss.item(), places=6)
        for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
            torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-4, atol=2e-7, msg=name)

    def test_complete_epochs_cover_every_request_once_without_short_sample_repeats(self):
        with tempfile.TemporaryDirectory() as temporary:
            entries, frozen_io, _ = self.dataset(temporary)
            out = Path(temporary) / "train"
            result = self.run_quiet(LoopDraft(self.config, self.embedding), entries, self.settings, out, frozen_io)
            self.assertEqual(result["event"], "complete")
            self.assertEqual(result["step"], 9)
            self.assertEqual(result["total_requests"], 33)
            rows = [json.loads(line) for line in (out / "metrics.jsonl").read_text().splitlines()]
            for epoch in range(3):
                epoch_rows = [row for row in rows if row["epoch"] == epoch + 1]
                self.assertEqual([row["requests"] for row in epoch_rows], [4, 4, 3])
                seen = [identity for row in epoch_rows for identity in row["request_ids"]]
                self.assertEqual(seen, [entries[i]["id"] for i in epoch_order(11, 0, epoch)])
                self.assertEqual(len(set(seen)), 11)
                for row in epoch_rows:
                    self.assertEqual(len(row["request_ids"]), len(set(row["request_ids"])))
                    for identity, count in zip(row["request_ids"], row["request_anchors"]):
                        entry = next(entry for entry in entries if entry["id"] == identity)
                        self.assertEqual(count, min(4, entry["available_anchors"]))
                checkpoint_model, payload = load_checkpoint(out / f"epoch{epoch + 1}.pt")
                self.assertEqual(payload["cursor"]["epoch"], epoch + 1)
                self.assertEqual(payload["cursor"]["next_request"], 0)
                self.assertEqual(checkpoint_model.config.source_layout, "canonical")
            self.assertEqual(result["total_anchors"], sum(row["anchors"] for row in rows))
            self.assertEqual(result["total_labels"], sum(row["valid_labels"] for row in rows))
            self.assertEqual(rows[-1]["lr"], self.settings.floor_lr)
            self.assertTrue((out / "initial.pt").exists())
            self.assertTrue((out / "model.pt").exists())

    def test_mid_epoch_resume_matches_uninterrupted_model_optimizer_rng_and_metrics(self):
        with tempfile.TemporaryDirectory() as temporary:
            entries, frozen_io, _ = self.dataset(temporary)
            root = Path(temporary)
            model = LoopDraft(self.config, self.embedding)
            initial = deepcopy(model.state_dict())
            torch_state, python_state = torch.get_rng_state(), random.getstate()
            self.run_quiet(model, entries, self.settings, root / "full", frozen_io, save_every=2)
            full = torch.load(root / "full/model.pt", weights_only=True)
            split_model = LoopDraft(self.config, self.embedding)
            split_model.load_state_dict(initial)
            torch.set_rng_state(torch_state)
            random.setstate(python_state)
            paused = self.run_quiet(split_model, entries, self.settings, root / "resumed", frozen_io,
                                    save_every=2, stop_after_steps=4)
            self.assertEqual((paused["step"], paused["epoch"], paused["next_request"]), (4, 1, 4))
            resumed_model, payload = load_checkpoint(root / "resumed/latest.pt")
            # Simulate a process crash after metrics but before a later checkpoint,
            # plus an interrupted JSON write. These rows must not double-count data.
            with (root / "resumed/metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(event="update", step=5, loss=999)) + "\n")
                handle.write('{"event":"update"')
            torch.manual_seed(987)
            random.seed(987)
            self.run_quiet(resumed_model, entries, self.settings, root / "resumed", frozen_io,
                           resume_payload=payload, save_every=2)
            resumed = torch.load(root / "resumed/model.pt", weights_only=True)
            for key in ("model", "optimizer", "cursor", "torch_rng", "python_rng"):
                self.assert_nested_equal(full[key], resumed[key])
            def training_rows(path):
                rows = [json.loads(line) for line in path.read_text().splitlines()]
                return [{key: value for key, value in row.items() if key != "update_seconds"} for row in rows]
            self.assertEqual(training_rows(root / "full/metrics.jsonl"), training_rows(root / "resumed/metrics.jsonl"))

    def test_resume_rejects_changed_budget_or_data_order(self):
        with tempfile.TemporaryDirectory() as temporary:
            entries, frozen_io, _ = self.dataset(temporary, count=5)
            out = Path(temporary) / "train"
            model = LoopDraft(self.config, self.embedding)
            self.run_quiet(model, entries, self.settings, out, frozen_io, stop_after_steps=1)
            model, payload = load_checkpoint(out / "latest.pt")
            with self.assertRaisesRegex(ValueError, "settings"):
                self.run_quiet(model, entries, replace(self.settings, epochs=4), out, frozen_io, resume_payload=payload)
            with self.assertRaisesRegex(ValueError, "data"):
                self.run_quiet(model, list(reversed(entries)), self.settings, out, frozen_io, resume_payload=payload)

    def test_training_only_manifest_and_nontraining_exclusion(self):
        with tempfile.TemporaryDirectory() as temporary:
            entries, _, manifest = self.dataset(temporary, count=3)
            path = Path(temporary) / "eval.pt"
            torch.save(self.record(50, 5, split="eval"), path)
            with manifest.open("a") as handle:
                handle.write(json.dumps(dict(path=str(path), split="eval")) + "\n")
            self.assertEqual(index_training_data(manifest, self.config, 3), entries)
            with self.assertRaisesRegex(ValueError, "expected 4"):
                index_training_data(manifest, self.config, 4)
            path = Path(temporary) / "empty.pt"
            torch.save(self.record(51, 1), path)
            with manifest.open("a") as handle:
                handle.write(json.dumps(dict(path=str(path), split="train")) + "\n")
            with self.assertRaisesRegex(ValueError, "no valid child anchors"):
                index_training_data(manifest, self.config)

    def test_branch_starts_from_parent_with_fresh_optimizer_and_schedule(self):
        with tempfile.TemporaryDirectory() as temporary:
            entries, frozen_io, manifest = self.dataset(temporary, count=4)
            root = Path(temporary)
            baseline_settings = replace(self.settings, epochs=1)
            self.run_quiet(LoopDraft(self.config, self.embedding), entries, baseline_settings, root / "baseline", frozen_io)
            parent_path = root / "baseline/model.pt"
            _, parent_payload = fork_from_final(parent_path, "trajectory")
            with redirect_stdout(io.StringIO()):
                result = main(["--role", "branch", "--feature-mode", "trajectory",
                               "--data", str(manifest), "--frozen-io", str(frozen_io),
                               "--out", str(root / "branch"), "--init-checkpoint", str(parent_path),
                               "--expected-requests", "4", "--num-threads", "1"])
            initial = torch.load(root / "branch/initial.pt", weights_only=True)
            self.assertEqual(initial["optimizer"]["state"], {})
            self.assertEqual(initial["cursor"]["step"], 0)
            self.assertEqual(result["step"], 2)
            self.assertEqual(initial["config"]["feature_mode"], "trajectory")
            self.assertEqual(initial["settings"]["peak_lr"], 5e-5)
            self.assertEqual(initial["settings"]["floor_lr"], 5e-6)
            for name, value in initial["model"].items():
                torch.testing.assert_close(value, parent_payload["model"][name], rtol=0, atol=0)

    def test_cli_refuses_forking_an_incomplete_baseline(self):
        with tempfile.TemporaryDirectory() as temporary:
            entries, frozen_io, manifest = self.dataset(temporary, count=4)
            out = Path(temporary) / "baseline"
            self.run_quiet(LoopDraft(self.config, self.embedding), entries, self.settings, out, frozen_io,
                           stop_after_steps=1)
            with self.assertRaisesRegex(ValueError, "completed baseline"):
                main(["--role", "branch", "--feature-mode", "final", "--data", str(manifest),
                      "--frozen-io", str(frozen_io), "--out", str(Path(temporary) / "branch"),
                      "--init-checkpoint", str(out / "latest.pt"), "--expected-requests", "4",
                      "--num-threads", "1"])

    def test_warmup_cosine_budget_and_stable_unique_anchor_sampling(self):
        settings = Settings()
        self.assertAlmostEqual(learning_rate(1, 100, settings), 2e-5)
        self.assertEqual(learning_rate(5, 100, settings), 1e-4)
        self.assertEqual(learning_rate(100, 100, settings), 1e-5)
        self.assertLess(learning_rate(50, 100, settings), learning_rate(6, 100, settings))
        with tempfile.TemporaryDirectory() as temporary:
            entries, _, _ = self.dataset(temporary, count=11)
            prepared = prepare_requests(entries, list(range(11)), self.settings, epoch=2)
            repeated = prepare_requests(entries, list(range(11)), self.settings, epoch=2)
            self.assertEqual([roots for _, _, roots in prepared], [roots for _, _, roots in repeated])
            for _, record, roots in prepared:
                self.assertEqual(len(roots), len(set(roots)))
                self.assertLessEqual(len(roots), 4)
                self.assertTrue(set(roots) <= set(anchor_positions(record)))

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU gloo unavailable")
    def test_gloo_two_and_four_ranks_match_global_loss_and_resume_with_empty_tail_ranks(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, _ = self.dataset(temporary, count=9)
            settings = replace(self.settings, epochs=2, requests_per_update=8)
            model = LoopDraft(self.config, self.embedding)
            initial = deepcopy(model.state_dict())
            torch_rng, python_rng = torch.get_rng_state(), random.getstate()
            indices = epoch_order(9, settings.seed, 0)[:8]
            requests = prepare_requests(entries, indices, settings, epoch=0)
            reference_stats = accumulate_gradients(model, requests, settings.microbatch, torch.device("cpu"))
            reference_gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
            self.run_quiet(model, entries, settings, root / "single", frozen_io, save_every=1)
            reference = torch.load(root / "single/model.pt", weights_only=True)
            reference_rows = [json.loads(line) for line in (root / "single/metrics.jsonl").read_text().splitlines()]
            for world_size in (2, 4):
                with self.subTest(world_size=world_size):
                    out = root / f"world{world_size}"
                    job = dict(config=self.config.dict(), settings=asdict(settings), initial=initial,
                               entries=entries, embedding=self.embedding, frozen_io=str(frozen_io),
                               out=str(out), torch_rng=torch_rng, python_rng=python_rng)
                    job_path = root / f"job{world_size}.pt"
                    torch.save(job, job_path)
                    mp.spawn(distributed_training_worker,
                             args=(world_size, str(root / f"gloo{world_size}"), str(job_path)),
                             nprocs=world_size, join=True)
                    gradients = torch.load(out / "gradients.pt", weights_only=True)
                    for name, value in gradients["grads"].items():
                        torch.testing.assert_close(value, reference_gradients[name], rtol=2e-4, atol=2e-7, msg=name)
                    for key in ("valid_labels", "anchors", "request_ids", "request_labels", "request_anchors",
                                "labels_by_depth", "correct_by_depth"):
                        self.assertEqual(gradients["stats"][key], reference_stats[key])
                    self.assertAlmostEqual(gradients["stats"]["loss"], reference_stats["loss"], places=6)
                    complete = torch.load(out / "full/model.pt", weights_only=True)
                    resumed = torch.load(out / "resumed/model.pt", weights_only=True)
                    self.assertEqual(complete["distributed"]["world_size"], world_size)
                    self.assertEqual(len(complete["rank_rng"]), world_size)
                    self.assertEqual(complete["cursor"], reference["cursor"])
                    for name, value in complete["model"].items():
                        self.assertFalse(name.startswith("module."))
                        torch.testing.assert_close(value, reference["model"][name], rtol=1e-5, atol=2e-7, msg=name)
                    for key in ("model", "optimizer", "cursor", "rank_rng", "torch_rng", "python_rng"):
                        self.assert_nested_equal(complete[key], resumed[key])
                    rows = [json.loads(line) for line in (out / "full/metrics.jsonl").read_text().splitlines()]
                    self.assertEqual([row["requests"] for row in rows], [8, 1, 8, 1])
                    for actual, expected in zip(rows, reference_rows):
                        self.assertEqual(sum(actual["requests_by_rank"]), actual["requests"])
                        if actual["requests"] == 1:
                            self.assertEqual(actual["requests_by_rank"], [1] + [0] * (world_size - 1))
                        for key in ("request_ids", "request_labels", "request_anchors", "labels_by_depth", "lr"):
                            self.assertEqual(actual[key], expected[key])
                        self.assertAlmostEqual(actual["loss"], expected["loss"], places=5)
                    loaded, _ = load_checkpoint(out / "full/model.pt")
                    self.assertEqual(loaded.config.source_layout, "canonical")


if __name__ == "__main__":
    unittest.main()
