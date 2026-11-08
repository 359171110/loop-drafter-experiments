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

from . import evaluate_loop_memory as evaluation
from . import target as target_module
from .test_evaluate import EOS, FakeDraft, FakeTarget, update_state


class FakeLoopMemoryDraft(FakeDraft):
    def __init__(self, prompt_length, *, window=64, **kwargs):
        super().__init__(prompt_length, **kwargs)
        self.config.window = window
        self.histories, self.appended = [], []
        self.prepared_tokens = 0

    def prepare_memory(self, history, mask):
        assert history.shape[2:] == (4, 4) and mask.all()
        assert history.dtype == torch.bfloat16
        self.prepared_tokens += history.shape[1]
        return history.clone()

    def append_memory(self, memory, accepted):
        assert accepted.ndim == 4 and accepted.shape[2:] == (4, 4)
        self.appended.append(accepted.clone())
        return torch.cat((memory, accepted), dim=1)[:, -self.config.window:]

    def generate_draft(self, taps, bonus, memory=None, max_children=None):
        assert memory.dtype == torch.bfloat16
        assert 1 <= memory.shape[1] <= self.config.window
        torch.testing.assert_close(taps, memory[:, -1], rtol=0, atol=0)
        self.histories.append(memory.clone())
        return super().generate_draft(taps, bonus, max_children=max_children)


def expected_history(tokens, window):
    state, rows = 0, []
    for index, token in enumerate(tokens):
        state = update_state(state, token)
        rows.append([[state, index + 1, token, r] for r in range(4)])
    return torch.tensor([rows[-window:]], dtype=torch.bfloat16)


class LoopMemoryDecoderTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)

    def run_case(self, budget, *, prompt_length=3, window=64, failures=(None,),
                 eos_after=None, wrong_eos=False, children=None, return_new_cache=False):
        prompt = torch.tensor([[index % 23 + 1 for index in range(prompt_length)]], dtype=torch.long)
        target = FakeTarget(prompt_length, eos_after, return_new_cache)
        draft = FakeLoopMemoryDraft(prompt_length, window=window, depth=5,
                                 failures=failures, eos_after=eos_after, wrong_eos=wrong_eos)
        reference = evaluation.decode_ar(FakeTarget(prompt_length, eos_after), prompt, budget, {EOS})
        result = evaluation.decode_loop_memory_speculative(target, draft, prompt, budget, {EOS}, children)
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
        self.assertEqual(draft.prepared_tokens, min(prompt_length, window))
        self.assertEqual(sum(x.shape[1] for x in draft.appended), committed)
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
        draft = FakeLoopMemoryDraft(1)
        for ids, budget, children in ((torch.tensor([[1]]), 0, None),
                                     (torch.tensor([[1], [2]]), 3, None),
                                     (torch.tensor([[1]]), 3, 6)):
            with self.assertRaises(ValueError):
                evaluation.decode_loop_memory_speculative(target, draft, ids, budget, {EOS}, children)
        self.assertFalse(target.forward_calls)

    def test_actual_loop_memory_interface_matches_ar_for_all_modes(self):
        from .loop_memory_draft import LoopMemoryConfig, LoopMemoryDraft
        prompt = torch.tensor([[2, 3, 5]], dtype=torch.long)
        embedding = torch.randn(31, 4)
        reference = evaluation.decode_ar(FakeTarget(3), prompt, 17, {EOS})
        for mode in evaluation.METHODS[1:]:
            config = LoopMemoryConfig(memory_mode=mode, target_hidden=4, vocab_size=31,
                                      width=8, mlp_width=16, layers=1, init_rank=4,
                                      prefix_width=3, depth=3, state_dim=3, state_head_dim=4,
                                      memory_width=8, memory_heads=2, window=8)
            draft = LoopMemoryDraft(config, embedding).eval()
            result = evaluation.decode_loop_memory_speculative(FakeTarget(3), draft, prompt, 17, {EOS})
            self.assertEqual(result["tokens"], reference["tokens"])

    def test_none_uses_original_decoder_without_memory_operations(self):
        draft = FakeDraft(3, depth=3, failures=(0, None, 2))
        draft.config.memory_mode = "none"
        draft.config.window = 8
        draft.prepare_memory = lambda *args: self.fail("none must not prepare memory")
        draft.append_memory = lambda *args: self.fail("none must not append memory")
        prompt = torch.tensor([[2, 3, 5]])
        reference = evaluation.decode_ar(FakeTarget(3), prompt, 17, {EOS})
        with patch.object(evaluation, "decode_speculative", wraps=evaluation.decode_speculative) as original:
            result = evaluation.decode_loop_memory_speculative(FakeTarget(3), draft, prompt, 17, {EOS})
        original.assert_called_once()
        self.assertEqual(result["tokens"], reference["tokens"])
        self.assertEqual(result["memory_update_seconds"], 0.)


class PromptEncodingTests(unittest.TestCase):
    def test_raw_completion_preserves_whitespace_and_frozen_ids_check_identity(self):
        class Tokenizer:
            name_or_path = "/local/model"
            chat_template = None
            def __len__(self):
                return 31
            def encode(self, text, add_special_tokens=False):
                self.last = (text, add_special_tokens)
                return [4, 8, 3]
        tok = Tokenizer()
        prompt = "def f(x):\n    \"\"\"Return x.\"\"\"\n    "
        row = dict(id="p", prompt=prompt, prompt_format="raw_completion")
        self.assertEqual(evaluation.encode_prompt(tok, row, 512).tolist(), [[4, 8, 3]])
        self.assertEqual(tok.last, (prompt, False))
        row.update(input_ids=[3, 2], tokenizer_path="/local/model")
        self.assertEqual(evaluation.encode_prompt(tok, row, 512).tolist(), [[3, 2]])
        row["tokenizer_path"] = "/different/model"
        with self.assertRaisesRegex(ValueError, "matching tokenizer"):
            evaluation.encode_prompt(tok, row, 512)
        row["tokenizer_path"] = "/local/model"
        with self.assertRaisesRegex(ValueError, "token budget"):
            evaluation.encode_prompt(tok, row, 1)

    def test_nonuniform_expected_counts(self):
        self.assertEqual(evaluation.expected_counts('{"gsm":300,"humaneval":164}'), {"gsm":300,"humaneval":164})
        self.assertIsNone(evaluation.expected_counts(None))
        with self.assertRaises(ValueError):
            evaluation.expected_counts('{"gsm":0}')


def protocol_payload(mode="none", *, scratch=False, frozen_io="fake-frozen-io.pt"):
    epochs = 4 if scratch else 2
    payload = dict(protocol=evaluation.SCRATCH_PROTOCOL if scratch else evaluation.LEGACY_PROTOCOL,
                   settings=dict(role="baseline" if scratch else "branch", epochs=epochs, seed=0,
                                 requests_per_update=2, max_anchors_per_request=64, microbatch=16,
                                 peak_lr=1e-4 if scratch else 5e-5),
                   config=dict(memory_mode=mode),
                   data_index=[dict(id=f"train-{i}", path=f"/records/{i}.pt", size=1024) for i in range(4)],
                   cursor=dict(step=epochs * 2, epoch=epochs, next_request=0, total_requests=epochs * 4,
                               total_anchors=epochs * 12, total_labels=epochs * 40),
                   sampling_epoch_offset=0 if scratch else 4,
                   parent=None if scratch else dict(checkpoint="ssd"), frozen_io=str(Path(frozen_io).resolve()))
    if scratch:
        payload.update(from_scratch=True,
                       initialization=dict(kind="from_scratch", seed=0, initial_state_sha256="a" * 64))
    return payload


class CheckpointProtocolTests(unittest.TestCase):
    def identity(self, payload):
        return evaluation.checkpoint_training_identity(payload, payload["config"], "fake-frozen-io.pt")

    def test_completed_scratch_and_legacy_contracts_are_supported(self):
        for scratch in (False, True):
            with self.subTest(scratch=scratch):
                reference = self.identity(protocol_payload(scratch=scratch))
                for mode in evaluation.METHODS[1:]:
                    evaluation.require_matching_training_identity(reference, self.identity(protocol_payload(mode, scratch=scratch)))
                self.assertEqual(reference["from_scratch"], scratch)
                self.assertEqual(reference["sampling_epoch_offset"], 0 if scratch else 4)

    def test_final_epoch_alone_cannot_hide_an_incomplete_training_budget(self):
        for change in ({"epoch": 3}, {"next_request": 2}, {"step": 7}, {"total_requests": 15}):
            with self.subTest(change=change):
                payload = protocol_payload(scratch=True)
                payload["cursor"].update(change)
                with self.assertRaisesRegex(ValueError, "fixed final|training data/update budget"):
                    self.identity(payload)
        payload = protocol_payload(scratch=True)
        payload["sampling_epoch_offset"] = 4
        with self.assertRaisesRegex(ValueError, "fixed final"):
            self.identity(payload)

    def test_scratch_requires_no_parent_and_a_valid_initialization_identity(self):
        for field, value in (("parent", {"checkpoint": "old"}), ("from_scratch", False),
                             ("initialization", None),
                             ("initialization", dict(kind="from_scratch", seed=1, initial_state_sha256="a" * 64)),
                             ("initialization", dict(kind="from_scratch", seed=0, initial_state_sha256="invalid"))):
            with self.subTest(field=field, value=value):
                payload = protocol_payload(scratch=True)
                payload[field] = value
                with self.assertRaises(ValueError):
                    self.identity(payload)

    def test_four_arms_cannot_mix_training_protocol_data_initialization_or_budget(self):
        reference = self.identity(protocol_payload(scratch=True))
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            evaluation.require_matching_training_identity(reference, self.identity(protocol_payload()))
        for kind in ("initialization", "data_order", "data_file", "settings", "config", "labels"):
            with self.subTest(kind=kind):
                payload = protocol_payload("trajectory", scratch=True)
                if kind == "initialization":
                    payload["initialization"]["initial_state_sha256"] = "b" * 64
                elif kind == "data_order":
                    payload["data_index"].reverse()
                elif kind == "data_file":
                    payload["data_index"][0]["size"] += 1
                elif kind == "settings":
                    payload["settings"]["max_anchors_per_request"] = 32
                elif kind == "config":
                    payload["config"]["window"] = 32
                else:
                    payload["cursor"]["total_labels"] += 1
                with self.assertRaisesRegex(ValueError, "must share"):
                    evaluation.require_matching_training_identity(reference, self.identity(payload))

    def test_frozen_io_and_loaded_config_are_bound_to_checkpoint(self):
        payload = protocol_payload(scratch=True)
        with self.assertRaisesRegex(ValueError, "frozen target IO differs"):
            evaluation.checkpoint_training_identity(payload, payload["config"], "different-io.pt")
        with self.assertRaisesRegex(ValueError, "loaded model config"):
            evaluation.checkpoint_training_identity(payload, {"memory_mode": "trajectory"}, "fake-frozen-io.pt")


class MainLoopMemoryTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.temporary = TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.prompts = [dict(id=f"p{i}", source="gsm" if i < 2 else "humaneval", split="eval",
                             prompt=f"def f{i}():\n    ", prompt_format="raw_completion") for i in range(3)]
        self.prompt_path = self.root / "prompts.jsonl"
        self.prompt_path.write_text("".join(json.dumps(row) + "\n" for row in self.prompts))
        self.out = self.root / "out"
        self.paths = {mode: self.root / f"{mode}.pt" for mode in evaluation.METHODS[1:]}
        for path in self.paths.values():
            path.write_bytes(b"checkpoint identity")
        self.argv = ["--model", "local-fake-model", "--prompts", str(self.prompt_path),
                     "--frozen-io", "fake-frozen-io.pt", "--out", str(self.out),
                     "--expected-counts", '{"gsm":2,"humaneval":1}', "--max-new-tokens", "16",
                     "--max-prompt-tokens", "2048", "--device", "cpu", "--num-threads", "1"]
        for mode, path in self.paths.items():
            self.argv.extend(["--" + mode.replace("_", "-") + "-checkpoint", str(path)])

    def run_main(self, stop_after=None, *, epoch=None, mismatch_parent=False, assignment="4", scratch=False,
                 mixed_protocol=False):
        calls = []

        def decode(method, ids, budget):
            if budget == 16:
                if stop_after is not None and len(calls) == stop_after:
                    raise RuntimeError("interrupted")
                calls.append((f"p{int(ids[0, 0])}", method))
            return dict(tokens=[1, 2, 3], prefill_seconds=.25, decode_seconds=.5,
                        rounds=[] if method == "ar" else [dict(committed_tokens=3, accepted_children=2)])

        def load(path, **kwargs):
            mode = Path(path).stem
            config = SimpleNamespace(memory_mode=mode)
            config.dict = lambda: dict(memory_mode=mode)
            draft = SimpleNamespace(config=config, method=mode)
            draft.eval = lambda: draft
            payload = protocol_payload(mode, scratch=scratch and not (mixed_protocol and mode == "trajectory"))
            if epoch is not None:
                payload["cursor"]["epoch"] = epoch
            if mismatch_parent and mode == "trajectory":
                payload["parent"] = dict(checkpoint="wrong")
            return draft, payload

        target = SimpleNamespace(tokenizer=SimpleNamespace(eos_token_id=0))
        with patch.dict("os.environ", {"CUDA_VISIBLE_DEVICES": assignment}), \
             patch.object(target_module.OuroTarget, "from_pretrained", return_value=target), \
             patch.object(evaluation, "load_loop_memory_checkpoint", side_effect=load), \
             patch.object(evaluation, "encode_prompt", side_effect=lambda tok, row, maximum: torch.tensor([[int(row["id"][1:])]])), \
             patch.object(evaluation, "decode_ar", side_effect=lambda target, ids, budget, eos: decode("ar", ids, budget)), \
             patch.object(evaluation, "decode_loop_memory_speculative", side_effect=lambda target, draft, ids, budget, eos: decode(draft.method, ids, budget)), \
             redirect_stdout(io.StringIO()):
            try:
                evaluation.main(self.argv)
            except RuntimeError as error:
                if str(error) != "interrupted" or stop_after is None:
                    raise
        return calls

    def test_five_method_rotation_nonuniform_panel_and_interrupted_resume(self):
        first = self.run_main(stop_after=6)
        path = self.out / "requests.jsonl"
        initial = evaluation.read_jsonl(path)
        with path.open("ab") as stream:
            stream.write(b'{"id":"partial"')
        rest = self.run_main()
        rows = evaluation.read_jsonl(path)
        self.assertEqual(rows[:6], initial)
        self.assertEqual(len(rows), 15)
        self.assertEqual(len({(r["id"], r["method"]) for r in rows}), 15)
        order = list(self.prompts)
        random.Random(0).shuffle(order)
        expected = []
        for i, prompt in enumerate(order):
            offset = i % len(evaluation.METHODS)
            expected.extend((prompt["id"], mode) for mode in evaluation.METHODS[offset:] + evaluation.METHODS[:offset])
        self.assertEqual(first + rest, expected)
        summary = json.loads((self.out / "summary.json").read_text())
        self.assertEqual(summary["source_counts"], {"gsm": 2, "humaneval": 1})
        self.assertEqual(summary["config"]["max_prompt_tokens"], 2048)
        self.assertTrue(all(r["exact_matches"] == r["requests"] for r in summary["results"]))
        self.assertEqual(self.run_main(), [])

    def test_incomplete_checkpoints_and_different_parents_are_rejected(self):
        with self.assertRaisesRegex(ValueError, "fixed final"):
            self.run_main(epoch=1)
        with self.assertRaisesRegex(ValueError, "same SSD baseline"):
            self.run_main(mismatch_parent=True)

    def test_scratch_final_checkpoints_run_and_resume_without_an_ssd_parent(self):
        first = self.run_main(stop_after=4, scratch=True)
        rest = self.run_main(scratch=True)
        self.assertEqual(len(first + rest), 15)
        summary = json.loads((self.out / "summary.json").read_text())
        identity = summary["hardware"]["training_identity"]
        self.assertEqual(identity["protocol"], evaluation.SCRATCH_PROTOCOL)
        self.assertIsNone(identity["parent"])
        self.assertEqual(identity["initialization"]["initial_state_sha256"], "a" * 64)
        self.assertEqual(identity["cursor"]["epoch"], 4)
        self.assertEqual(self.run_main(scratch=True), [])

    def test_mixed_scratch_and_legacy_checkpoints_are_rejected_before_decoding(self):
        with self.assertRaisesRegex(ValueError, "cannot mix"):
            self.run_main(scratch=True, mixed_protocol=True)
        self.assertFalse((self.out / "requests.jsonl").exists())

    def test_resume_requires_same_gpu_and_checkpoint_identity(self):
        self.run_main(stop_after=6)
        with self.assertRaisesRegex(ValueError, "same GPU assignment"):
            self.run_main(assignment="5")
        with self.paths["trajectory"].open("ab") as stream:
            stream.write(b"changed")
        with self.assertRaisesRegex(ValueError, "checkpoint identity changed"):
            self.run_main()


if __name__ == "__main__":
    unittest.main()
