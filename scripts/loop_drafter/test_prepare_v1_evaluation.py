"""CPU checks for fixed benchmark membership, prompts and exact-overlap scope."""

import hashlib
import json
import unittest

from . import prepare_v1_evaluation as prepare


class FakeTokenizer:
    chat_template = "fake-template-for-tests"

    def encode(self, text, add_special_tokens):
        return ([999] if add_special_tokens else []) + [ord(char) for char in text]

    def apply_chat_template(self, messages, *, tokenize, add_generation_prompt, return_dict=False):
        assert add_generation_prompt and not return_dict
        text = "<user>" + messages[0]["content"] + "</user><assistant>"
        return self.encode(text, add_special_tokens=False) if tokenize else text


class PrepareV1Tests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.old = [dict(id=f"{source}:{i}", source=source, source_id=str(i), source_split="test",
                         split="eval", prompt=f"original {source} request {i}", domain="math" if source == prepare.GSM else "chat")
                    for source in (prepare.GSM, prepare.CHAT, prepare.CODE) for i in range(300)]
        self.human = [dict(task_id=f"HumanEval/{i}", prompt=f'def problem_{i}(x):\n    """Task {i}."""\n    ',
                           canonical_solution="SECRET_SOLUTION", test="SECRET_TEST") for i in range(164)]
        self.mbpp = [dict(task_id=i, text=f"Write function number {i}.",
                          test_list=[f"assert problem_{i}({j}) == {j}" for j in range(3)],
                          code="SECRET_SOLUTION", challenge_test_list=["SECRET_TEST"])
                     for i in range(11, 511)]

    def build(self, **kwargs):
        return prepare.build_panel(self.old, self.human, self.mbpp, self.tokenizer, "/fake/ouro", **kwargs)

    def test_exact_counts_source_splits_and_deterministic_mbpp_membership(self):
        first, second = self.build(), self.build()
        self.assertEqual(first, second)
        self.assertEqual(len(first), 1064)
        self.assertEqual(len({row["id"] for row in first}), 1064)
        self.assertFalse(any(row["source"] == prepare.CODE for row in first))
        actual = [row["source_id"] for row in first if row["source"] == prepare.MBPP]
        expected = [str(row["task_id"]) for row in sorted(
            self.mbpp, key=lambda row: prepare.rank(0, "evaluation_v1:mbpp", str(row["task_id"]))
        )[:300]]
        self.assertEqual(actual, expected)
        self.assertEqual(len(set(actual)), 300)
        self.assertTrue(all(row["split"] == "eval" for row in first))

    def test_humaneval_preserves_trailing_whitespace_without_chat_or_answers(self):
        rows = [row for row in self.build() if row["source"] == prepare.HUMAN]
        for source, row in zip(self.human, rows):
            self.assertEqual(row["prompt"], source["prompt"])
            self.assertTrue(row["prompt"].endswith("\n    "))
            self.assertEqual(row["input_ids"], self.tokenizer.encode(source["prompt"], add_special_tokens=False))
            self.assertEqual(row["benchmark_docstring"], source["prompt"].split('"""')[1])
            self.assertNotIn("SECRET", json.dumps(row))

    def test_legacy_prompt_and_input_ids_are_unchanged(self):
        old_by_id = {row["id"]: row for row in self.old}
        legacy = [row for row in self.build() if row["prompt_format"] == "native_chat"]
        self.assertEqual(len(legacy), 600)
        for row in legacy:
            old = old_by_id[row["id"]]
            self.assertEqual(row["prompt"], old["prompt"])
            self.assertEqual(row["input_ids"], self.tokenizer.apply_chat_template(
                [{"role": "user", "content": old["prompt"]}], tokenize=True, add_generation_prompt=True))

    def test_mbpp_uses_public_tests_and_no_reference_solution(self):
        panel = [row for row in self.build() if row["source"] == prepare.MBPP]
        for row in panel:
            self.assertEqual(row["prompt"].count("assert "), 3)
            self.assertTrue(row["prompt"].endswith("\n[BEGIN]\n"))
            self.assertNotIn("SECRET", json.dumps(row))
            self.assertEqual(row["prompt_format"], "raw_completion")

    def test_frozen_hashes_use_exact_utf8_and_compact_json(self):
        row = self.build()[0]
        self.assertEqual(row["prompt_sha256"], hashlib.sha256(row["prompt"].encode()).hexdigest())
        expected = hashlib.sha256(json.dumps(row["input_ids"], separators=(",", ":")).encode()).hexdigest()
        self.assertEqual(row["input_ids_sha256"], expected)

    def test_rejects_missing_humaneval_sanitized_mbpp_and_truncation(self):
        for human, mbpp in ((self.human[:-1], self.mbpp), (self.human, self.mbpp[:257])):
            with self.assertRaises(ValueError):
                prepare.build_panel(self.old, human, mbpp, self.tokenizer, "/fake/ouro")
        with self.assertRaisesRegex(ValueError, "untruncated limit"):
            self.build(max_prompt_tokens=10)
        with self.assertRaisesRegex(ValueError, "seed 0"):
            self.build(seed=1)

    def test_overlap_keeps_actual_training_separate_from_reserves(self):
        panel = [row for row in self.build() if row["source"] == prepare.MBPP][:2]
        train = [dict(id="code-train", source=prepare.CODE, prompt="  " + panel[0]["benchmark_text"].upper() + "  ")]
        reserve = [dict(id="code-reserve", source=prepare.CODE, prompt=panel[1]["benchmark_text"])]
        report = prepare.overlap_audit(panel, train, reserve, self.tokenizer)
        self.assertEqual(report["comparisons"]["actual_training"]["matched_eval_requests"], 1)
        self.assertEqual(report["comparisons"]["reserves"]["matched_eval_requests"], 1)
        match = report["comparisons"]["actual_training"]["matches"][0]
        self.assertEqual(match["field"], "benchmark_text")
        self.assertEqual(match["train_matches"][0]["id"], "code-train")
        self.assertIn("no semantic decontamination", report["scope"])

    def test_serialized_and_token_exact_matches_are_reported(self):
        row = self.build()[0]
        train = [dict(id="existing", source=prepare.GSM, prompt=row["prompt"])]
        report = prepare.overlap_audit([row], train, [], self.tokenizer)
        fields = {match["field"] for match in report["comparisons"]["actual_training"]["matches"]}
        self.assertEqual(fields, {"prompt", "benchmark_text", "serialized_prompt", "input_ids"})


if __name__ == "__main__":
    unittest.main()
