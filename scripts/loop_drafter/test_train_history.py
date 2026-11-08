"""CPU checks of baseline-equivalent sampling, loss weighting and resumption."""
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

from . import train_history
from .history_draft import HistoryConfig, HistoryDraft, load_history_checkpoint, make_history_batch
from .train import anchor_positions, make_batch
from .train_protocol import Settings, epoch_order, index_training_data, prepare_requests, shard_indices
from .train_history import (accumulate_gradients, load_training_index, main,
                            synchronized_gradients, train_epochs)


def distributed_worker(rank, world_size, rendezvous, job_path):
    torch.set_num_threads(1)
    dist.init_process_group("gloo", init_method=f"file://{rendezvous}", rank=rank, world_size=world_size)
    try:
        job = torch.load(job_path, weights_only=True)
        config, settings = HistoryConfig(**job["config"]), Settings(**job["settings"])
        root = Path(job["out"])
        model = HistoryDraft(config, job["embedding"])
        model.load_state_dict(job["initial"])
        indices = epoch_order(len(job["entries"]), settings.seed, 0)[:settings.requests_per_update]
        requests = prepare_requests(job["entries"], shard_indices(indices, rank, world_size), settings, 0)
        stats = synchronized_gradients(model, requests, settings.microbatch, torch.device("cpu"))
        if rank == 0:
            root.mkdir(parents=True, exist_ok=True)
            torch.save(dict(stats=stats, grads={name: p.grad for name, p in model.named_parameters()}),
                       root / "gradients.pt")
        with redirect_stdout(io.StringIO()):
            train_epochs(model, job["entries"], settings, root / "full", job["frozen_io"], save_every=1)
        states = [None] * world_size
        dist.all_gather_object(states, model.state_dict())
        for state in states:
            for name, value in state.items():
                if not torch.equal(value, states[0][name]):
                    raise AssertionError(f"rank parameter divergence: {name}")
        split = HistoryDraft(config, job["embedding"])
        split.load_state_dict(job["initial"])
        with redirect_stdout(io.StringIO()):
            train_epochs(split, job["entries"], settings, root / "resumed", job["frozen_io"],
                         stop_after_steps=1, save_every=1)
            resumed, payload = load_history_checkpoint(root / "resumed/latest.pt")
            train_epochs(resumed, job["entries"], settings, root / "resumed", job["frozen_io"],
                         resume_payload=payload, save_every=1)
    finally:
        dist.destroy_process_group()


class HistoryTrainingTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.manual_seed(19)
        random.seed(19)
        self.config = HistoryConfig(target_hidden=8, vocab_size=23, width=8, mlp_width=16,
                                    layers=1, heads=2, window=4, depth=3)
        self.embedding = torch.randn(23, 8)
        self.settings = Settings(epochs=4, requests_per_update=4, max_anchors_per_request=4, microbatch=2)

    def record(self, index, generated_length):
        generator = torch.Generator().manual_seed(100 + index)
        length = 3 + generated_length
        return dict(id=f"request_{index}\u2028tail", source=f"source_{index % 3}", split="train",
                    token_ids=torch.randint(0, 23, (length,), generator=generator), prompt_length=3,
                    round_hidden=torch.randn(length - 1, 4, 8, generator=generator))

    def dataset(self, directory, count=5):
        root = Path(directory)
        rows = []
        for index in range(count):
            record = self.record(index, 2 + index * 7)
            path = root / f"record_{index}.pt"
            torch.save(record, path)
            rows.append(dict(path=str(path), id=record["id"], source=record["source"], split="train",
                             prompt_length=3, generated_length=2 + index * 7))
        manifest = root / "manifest.jsonl"
        manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        frozen_io = root / "frozen_io.pt"
        torch.save(dict(embedding=self.embedding), frozen_io)
        entries = index_training_data(manifest, self.config, count)
        return entries, frozen_io, manifest

    def run_quiet(self, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return train_epochs(*args, **kwargs)

    def rows(self, root):
        with (Path(root) / "metrics.jsonl").open() as handle:
            return [json.loads(line) for line in handle]

    def assert_nested_equal(self, a, b):
        if isinstance(a, torch.Tensor):
            torch.testing.assert_close(a, b, rtol=0, atol=0)
        elif isinstance(a, dict):
            self.assertEqual(a.keys(), b.keys())
            for key in a:
                self.assert_nested_equal(a[key], b[key])
        elif isinstance(a, (list, tuple)):
            self.assertEqual(len(a), len(b))
            for left, right in zip(a, b):
                self.assert_nested_equal(left, right)
        else:
            self.assertEqual(a, b)

    def test_both_modes_use_exact_baseline_request_order_roots_and_labels(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, _ = self.dataset(root)
            expected = []
            for epoch in range(self.settings.epochs):
                order = epoch_order(len(entries), self.settings.seed, epoch)
                for begin in range(0, len(order), self.settings.requests_per_update):
                    requests = prepare_requests(entries, order[begin:begin + self.settings.requests_per_update],
                                                self.settings, epoch)
                    expected.append([(entry["id"], roots) for entry, _, roots in requests])
            for mode in ("history", "history_residual"):
                with self.subTest(mode=mode):
                    seen = []

                    def observe(*args, **kwargs):
                        requests = prepare_requests(*args, **kwargs)
                        seen.append([(entry["id"], roots) for entry, _, roots in requests])
                        for _, record, roots in requests:
                            actual = make_history_batch(record, roots, 3, 4)
                            reference = make_batch(record, roots, 3)
                            self.assertTrue(torch.equal(actual["labels"], reference["labels"]))
                            self.assertTrue(torch.equal(actual["valid"], reference["valid"]))
                        return requests

                    torch.manual_seed(0)
                    model = HistoryDraft(replace(self.config, feature_mode=mode), self.embedding)
                    with patch.object(train_history, "prepare_requests", side_effect=observe):
                        result = self.run_quiet(model, entries, self.settings, root / mode, frozen_io)
                    self.assertEqual(seen, expected)
                    self.assertEqual(result["total_requests"], len(entries) * 4)
                    rows = self.rows(root / mode)
                    self.assertEqual([row["sampling_epoch"] for row in rows], [0, 0, 1, 1, 2, 2, 3, 3])
                    self.assertEqual(rows[-1]["lr"], self.settings.floor_lr)
                    payload = torch.load(root / mode / "model.pt", weights_only=True)
                    self.assertEqual(payload["schema"], "loop_history_drafter_checkpoint_v1")
                    self.assertEqual(payload["sampling_epoch_offset"], 0)
                    self.assertTrue(all((root / mode / f"epoch{i}.pt").exists() for i in range(1, 5)))

    def test_global_label_normalization_matches_concatenated_reference(self):
        for mode in ("history", "history_residual"):
            with self.subTest(mode=mode):
                model = HistoryDraft(replace(self.config, feature_mode=mode), self.embedding)
                reference = deepcopy(model)
                requests = []
                for index, length in enumerate((2, 4, 14)):
                    record = self.record(index, length)
                    requests.append((dict(id=str(index)), record, anchor_positions(record, 6, index)))
                stats = accumulate_gradients(model, requests, 2, torch.device("cpu"))
                batches = [make_history_batch(record, roots, 3, 4) for _, record, roots in requests]
                combined = {key: torch.cat([batch[key] for batch in batches]) for key in batches[0]}
                logits = reference(combined["history"], combined["history_mask"], combined["bonus"])
                loss = F.cross_entropy(logits.flatten(0, 1), combined["labels"].flatten(), ignore_index=-100)
                loss.backward()
                self.assertAlmostEqual(stats["loss"], loss.item(), places=6)
                self.assertEqual(stats["valid_labels"], int(combined["valid"].sum()))
                self.assertEqual(stats["labels_by_depth"], combined["valid"].sum(0).tolist())
                for (name, actual), (_, expected) in zip(model.named_parameters(), reference.named_parameters()):
                    torch.testing.assert_close(actual.grad, expected.grad, rtol=2e-5, atol=2e-7, msg=name)

    def test_offset_and_mid_epoch_resume_preserve_model_optimizer_rng_and_sampling(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, _ = self.dataset(root)
            settings = replace(self.settings, epochs=3)
            model = HistoryDraft(self.config, self.embedding)
            initial = deepcopy(model.state_dict())
            torch_rng, python_rng = torch.get_rng_state(), random.getstate()
            self.run_quiet(model, entries, settings, root / "full", frozen_io,
                           sampling_epoch_offset=4, save_every=2)
            full = torch.load(root / "full/model.pt", weights_only=True)
            split = HistoryDraft(self.config, self.embedding)
            split.load_state_dict(initial)
            torch.set_rng_state(torch_rng)
            random.setstate(python_rng)
            result = self.run_quiet(split, entries, settings, root / "resumed", frozen_io,
                                    sampling_epoch_offset=4, save_every=2, stop_after_steps=3)
            self.assertEqual((result["epoch"], result["next_request"]), (1, 4))
            resumed, payload = load_history_checkpoint(root / "resumed/latest.pt")
            with (root / "resumed/metrics.jsonl").open("a") as handle:
                handle.write(json.dumps(dict(event="update", step=4, loss=999)) + "\n")
                handle.write('{"event":"update"')
            torch.manual_seed(987)
            random.seed(987)
            self.run_quiet(resumed, entries, settings, root / "resumed", frozen_io,
                           resume_payload=payload, sampling_epoch_offset=4, save_every=2)
            recovered = torch.load(root / "resumed/model.pt", weights_only=True)
            for key in ("model", "optimizer", "cursor", "torch_rng", "python_rng"):
                self.assert_nested_equal(full[key], recovered[key])
            full_rows, resumed_rows = self.rows(root / "full"), self.rows(root / "resumed")
            strip_time = lambda rows: [{k: v for k, v in row.items() if k != "update_seconds"} for row in rows]
            self.assertEqual(strip_time(full_rows), strip_time(resumed_rows))
            self.assertEqual([row["sampling_epoch"] for row in resumed_rows], [4, 4, 5, 5, 6, 6])
            for epoch in range(3):
                expected_ids = [entries[i]["id"] for i in epoch_order(len(entries), settings.seed, epoch + 4)]
                self.assertEqual([i for row in resumed_rows[2 * epoch:2 * epoch + 2] for i in row["request_ids"]],
                                 expected_ids)
            a = prepare_requests(entries, [4], settings, 0)[0][2]
            b = prepare_requests(entries, [4], settings, 4)[0][2]
            self.assertNotEqual(a, b)

    def test_resume_rejects_sampling_offset_budget_and_data_changes(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, _ = self.dataset(root)
            self.run_quiet(HistoryDraft(self.config, self.embedding), entries, self.settings,
                           root / "out", frozen_io, stop_after_steps=1)
            model, payload = load_history_checkpoint(root / "out/latest.pt")
            for kwargs, expected in ((dict(sampling_epoch_offset=1), "sampling epoch"),
                                     (dict(settings=replace(self.settings, epochs=2)), "settings"),
                                     (dict(entries=list(reversed(entries))), "data")):
                args = dict(model=model, entries=entries, settings=self.settings, out=root / "out",
                            frozen_io=frozen_io, resume_payload=payload)
                args.update(kwargs)
                with self.subTest(expected=expected), self.assertRaisesRegex(ValueError, expected):
                    self.run_quiet(**args)

    def test_cached_baseline_index_checks_manifest_config_and_file_identity(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, _, manifest = self.dataset(root)
            index = root / "data_index.json"
            index.write_text(json.dumps(entries))
            (root / "config.json").write_text(json.dumps(dict(config=self.config.dict())))
            with patch.object(train_history, "index_training_data", side_effect=AssertionError("must use cache")):
                self.assertEqual(load_training_index(manifest, self.config, 5, index), entries)
            with self.assertRaisesRegex(ValueError, "configuration"):
                load_training_index(manifest, replace(self.config, target_hidden=12), 5, index)
            index.write_text(json.dumps(list(reversed(entries))))
            with self.assertRaisesRegex(ValueError, "order"):
                load_training_index(manifest, self.config, 5, index)
            index.write_text(json.dumps(entries))
            with Path(entries[0]["path"]).open("ab") as handle:
                handle.write(b"changed")
            with self.assertRaisesRegex(ValueError, "changed since indexing"):
                load_training_index(manifest, self.config, 5, index)

    def test_cpu_cli_and_resume_configuration(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            _, frozen_io, manifest = self.dataset(root, 3)
            args = ["--data", str(manifest), "--frozen-io", str(frozen_io), "--out", str(root / "out"),
                    "--feature-mode", "history_residual", "--epochs", "1", "--expected-requests", "3",
                    "--requests-per-update", "2", "--max-anchors-per-request", "3", "--microbatch", "2",
                    "--width", "8", "--mlp-width", "16", "--layers", "1", "--heads", "2",
                    "--window", "4", "--depth", "3", "--num-threads", "1"]
            with redirect_stdout(io.StringIO()):
                paused = main(args + ["--stop-after-steps", "1"])
                complete = main(args + ["--resume"])
            self.assertEqual(paused["event"], "paused")
            self.assertEqual(complete["event"], "complete")
            with self.assertRaisesRegex(ValueError, "model configuration"):
                main(args + ["--resume", "--window", "8"])

    @unittest.skipUnless(dist.is_available() and dist.is_gloo_available(), "CPU gloo unavailable")
    def test_two_rank_gradients_and_resume_match_single_rank_with_empty_tail(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            entries, frozen_io, _ = self.dataset(root)
            settings = replace(self.settings, epochs=2)
            model = HistoryDraft(self.config, self.embedding)
            initial = deepcopy(model.state_dict())
            order = epoch_order(len(entries), settings.seed, 0)[:settings.requests_per_update]
            stats = accumulate_gradients(model, prepare_requests(entries, order, settings, 0), 2, torch.device("cpu"))
            gradients = {name: p.grad.clone() for name, p in model.named_parameters()}
            self.run_quiet(model, entries, settings, root / "single", frozen_io, save_every=1)
            job = dict(config=self.config.dict(), settings=asdict(settings), initial=initial,
                       embedding=self.embedding, entries=entries, frozen_io=str(frozen_io), out=str(root / "two"))
            torch.save(job, root / "job.pt")
            mp.spawn(distributed_worker, args=(2, str(root / "gloo"), str(root / "job.pt")), nprocs=2, join=True)
            actual = torch.load(root / "two/gradients.pt", weights_only=True)
            self.assertEqual(actual["stats"]["request_ids"], stats["request_ids"])
            self.assertEqual(actual["stats"]["labels_by_depth"], stats["labels_by_depth"])
            self.assertAlmostEqual(actual["stats"]["loss"], stats["loss"], places=6)
            for name, value in actual["grads"].items():
                torch.testing.assert_close(value, gradients[name], rtol=2e-5, atol=2e-7, msg=name)
            single = torch.load(root / "single/model.pt", weights_only=True)
            full = torch.load(root / "two/full/model.pt", weights_only=True)
            resumed = torch.load(root / "two/resumed/model.pt", weights_only=True)
            self.assertEqual(single["cursor"], full["cursor"])
            for name, value in full["model"].items():
                torch.testing.assert_close(value, single["model"][name], rtol=2e-5, atol=2e-7, msg=name)
            for key in ("model", "optimizer", "cursor"):
                self.assert_nested_equal(full[key], resumed[key])
            rows = self.rows(root / "two/full")
            self.assertEqual([row["requests_by_rank"] for row in rows], [[2, 2], [1, 0], [2, 2], [1, 0]])


if __name__ == "__main__":
    unittest.main()
