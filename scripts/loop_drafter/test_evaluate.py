"""CPU state-machine tests for the actual speculative decoder.

The deterministic target depends on the entire prefix through a rolling state.
Consequently a rejected token left in its cache changes later outputs. Taps
encode that same state, allowing the fake drafter to detect an off-by-one source.
No real model weights, CUDA kernels, or trained checkpoints are required.
"""
from types import SimpleNamespace
import unittest

import torch

from .evaluate import decode_ar, decode_speculative


VOCAB = 31
EOS = 0


def update_state(state, token):
    return (state * 17 + int(token) + 1) % 97


def prefix_state(tokens):
    state = 0
    for token in tokens:
        state = update_state(state, token)
    return state


def next_token(state, length, prompt_length, eos_after):
    if eos_after is not None and length - prompt_length + 1 == eos_after:
        return EOS
    return (state * 11 + length * 7) % 29 + 1


class FakeCache:
    def __init__(self, tokens=()):
        self.tokens = list(tokens)

    def get_seq_length(self):
        return len(self.tokens)


class FakeTarget:
    def __init__(self, prompt_length, eos_after=None, return_new_cache=False):
        self.prompt_length = prompt_length
        self.eos_after = eos_after
        self.return_new_cache = return_new_cache
        self.forward_calls = []
        self.crop_events = []

    def forward(self, input_ids, cache=None):
        if input_ids.ndim != 2 or input_ids.shape[0] != 1 or not input_ids.shape[1]:
            raise AssertionError("fake target requires a nonempty batch-one input")
        cache = FakeCache() if cache is None else cache
        self.forward_calls.append(dict(prefix=list(cache.tokens), input=input_ids[0].tolist()))
        rows, hidden = [], []
        state = prefix_state(cache.tokens)
        for token in input_ids[0].tolist():
            cache.tokens.append(token)
            state = update_state(state, token)
            prediction = next_token(state, len(cache.tokens), self.prompt_length, self.eos_after)
            logits = torch.full((VOCAB,), -50.)
            logits[prediction] = 50.
            rows.append(logits)
            hidden.append([[state, len(cache.tokens), token, r] for r in range(4)])
        return SimpleNamespace(logits=torch.stack(rows)[None],
                               round_hidden=torch.tensor(hidden, dtype=torch.float32)[None],
                               cache=cache)

    def crop_cache(self, cache, length):
        if not 0 <= length <= len(cache.tokens):
            raise AssertionError("invalid crop boundary")
        self.crop_events.append(dict(before=list(cache.tokens), length=length,
                                     kept=list(cache.tokens[:length])))
        if self.return_new_cache:
            return FakeCache(cache.tokens[:length])
        del cache.tokens[length:]
        return None


class FakeDraft:
    def __init__(self, prompt_length, depth=3, eos_after=None, failures=(None,), wrong_eos=False):
        self.config = SimpleNamespace(depth=depth)
        self.prompt_length = prompt_length
        self.eos_after = eos_after
        self.failures = failures
        self.wrong_eos = wrong_eos
        self.calls = []

    def generate_draft(self, taps, bonus, max_children=None):
        count = self.config.depth if max_children is None else max_children
        if not 0 <= count <= self.config.depth:
            raise AssertionError("child budget exceeded")
        if taps.shape != (1, 4, 4) or bonus.shape != (1,):
            raise AssertionError("wrong drafter input shape")
        state, length = (int(value) for value in taps[0, -1, :2])
        expected_bonus = next_token(state, length, self.prompt_length, self.eos_after)
        if int(bonus[0]) != expected_bonus:
            raise AssertionError("source taps are not immediately before the known bonus")
        failure = self.failures[len(self.calls) % len(self.failures)]
        state, length = update_state(state, int(bonus[0])), length + 1
        proposals = []
        for child in range(count):
            wanted = next_token(state, length, self.prompt_length, self.eos_after)
            proposed = wanted
            if child == failure:
                proposed = EOS if self.wrong_eos and wanted != EOS else wanted % 29 + 1
                if proposed == wanted:
                    raise AssertionError("the designated rejection must be incorrect")
            proposals.append(proposed)
            state, length = update_state(state, proposed), length + 1
        self.calls.append(dict(bonus=int(bonus[0]), source_length=int(taps[0, -1, 1]),
                               count=count, proposals=proposals, failure=failure))
        return torch.tensor([proposals], dtype=torch.long, device=bonus.device)


class EvaluateTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.prompt = torch.tensor([[4, 8, 3]], dtype=torch.long)

    def run_case(self, budget, *, failures=(None,), eos_after=None, wrong_eos=False,
                 depth=3, children=None, return_new_cache=False):
        prompt_length = self.prompt.shape[1]
        ar_target = FakeTarget(prompt_length, eos_after)
        target = FakeTarget(prompt_length, eos_after, return_new_cache)
        draft = FakeDraft(prompt_length, depth, eos_after, failures, wrong_eos)
        ar = decode_ar(ar_target, self.prompt, budget, {EOS})
        spec = decode_speculative(target, draft, self.prompt, budget, {EOS}, children=children)
        self.assertEqual(spec["tokens"], ar["tokens"])
        self.assertLessEqual(len(spec["tokens"]), budget)
        self.assertEqual(spec["target_calls"], len(target.forward_calls))
        self.assertEqual(ar["target_calls"], len(ar_target.forward_calls))
        self.assertEqual(len(target.crop_events), len(spec["rounds"]))
        self.assertEqual(len(draft.calls), len(spec["rounds"]))
        committed = 0
        for ledger, crop, forward, proposal in zip(spec["rounds"], target.crop_events,
                                                 target.forward_calls[1:], draft.calls):
            self.assertEqual(forward["prefix"], self.prompt[0].tolist() + spec["tokens"][:committed])
            self.assertEqual(proposal["source_length"], prompt_length + committed)
            self.assertEqual(forward["input"], [proposal["bonus"]] + proposal["proposals"])
            self.assertEqual(ledger["proposed_children"], len(proposal["proposals"]))
            self.assertEqual(ledger["committed_tokens"], 1 + ledger["accepted_children"])
            self.assertLessEqual(ledger["accepted_children"], ledger["proposed_children"])
            committed += ledger["committed_tokens"]
            self.assertEqual(crop["length"], prompt_length + committed)
            self.assertEqual(crop["kept"], self.prompt[0].tolist() + spec["tokens"][:committed])
        # The only uncached output may be the already-known final bonus.
        self.assertIn(len(spec["tokens"]) - committed, (0, 1))
        return spec, draft, target

    def test_all_right_and_all_wrong_match_ar(self):
        for return_new in (False, True):
            for failure in (None, 0):
                with self.subTest(return_new=return_new, failure=failure):
                    result, _, _ = self.run_case(17, failures=(failure,), return_new_cache=return_new)
                    for row in result["rounds"]:
                        expected = row["proposed_children"] if failure is None else 0
                        self.assertEqual(row["accepted_children"], expected)

    def test_reject_at_each_depth_and_alternating_depths(self):
        for failures in ((0,), (1,), (2,), (2, 0, None, 1)):
            for return_new in (False, True):
                with self.subTest(failures=failures, return_new=return_new):
                    result, draft, target = self.run_case(23, failures=failures, return_new_cache=return_new)
                    for row, proposal, crop in zip(result["rounds"], draft.calls, target.crop_events):
                        failure, count = proposal["failure"], row["proposed_children"]
                        expected = count if failure is None else min(failure, count)
                        self.assertEqual(row["accepted_children"], expected)
                        self.assertEqual(len(crop["before"]) - crop["length"], count - expected)

    def test_eos_at_bonus_and_each_child_position(self):
        for eos_after in (1, 2, 3, 4, 5, 8, 11):
            for failures in ((None,), (0,), (2, None, 1)):
                with self.subTest(eos_after=eos_after, failures=failures):
                    result, draft, target = self.run_case(20, failures=failures, eos_after=eos_after)
                    self.assertEqual(len(result["tokens"]), eos_after)
                    self.assertEqual(result["tokens"][-1], EOS)
                    if eos_after == 1:
                        self.assertFalse(draft.calls)
                        self.assertFalse(target.crop_events)
                        self.assertEqual(result["target_calls"], 1)

    def test_wrong_proposed_eos_does_not_terminate_output(self):
        for eos_after in (None, 6, 10):
            with self.subTest(eos_after=eos_after):
                result, _, _ = self.run_case(14, failures=(0, 1, 2), eos_after=eos_after, wrong_eos=True)
                self.assertEqual(len(result["tokens"]), eos_after or 14)

    def test_budget_truncation_and_child_limit(self):
        for budget in (1, 2, 3, 4, 5, 7, 8, 9):
            for children in (0, 1, 2, 3):
                for failures in ((None,), (1, 0, None)):
                    with self.subTest(budget=budget, children=children, failures=failures):
                        result, draft, _ = self.run_case(budget, failures=failures, children=children)
                        self.assertEqual(len(result["tokens"]), budget)
                        self.assertTrue(all(call["count"] <= children for call in draft.calls))
                        if budget == 1:
                            self.assertEqual(result["rounds"], [])


if __name__ == "__main__":
    unittest.main()
