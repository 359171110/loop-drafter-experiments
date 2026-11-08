"""CPU tests for ARC membership, no-answer templates and overlap isolation."""

import json
from pathlib import Path
import tempfile
import unittest

from . import prepare_reasoning_evaluation as prepare
from .test_prepare_v1_evaluation import FakeTokenizer


class QuestionOnlyRow(dict):
    def __getitem__(self, key):
        if key == "answerKey":
            raise AssertionError("answerKey must never be read")
        return super().__getitem__(key)


class PrepareReasoningTests(unittest.TestCase):
    def setUp(self):
        self.tokenizer = FakeTokenizer()
        self.rows = [QuestionOnlyRow(id=f"science-{i}", question=f"What does species {i} need?",
                                    choices={"label": ["1", "2", "3"], "text": ["water", "iron", "stone"]},
                                    answerKey="SECRET") for i in range(8)]

    def build(self, training=(), **kwargs):
        return prepare.build_panel(self.rows, training, self.tokenizer, "/fake/ouro",
                                   count=3, expected_candidates=8, **kwargs)

    def test_original_labels_variable_options_and_no_answer_access(self):
        expected = "Question: What does species 0 need?\nChoices:\n1. water\n2. iron\n3. stone\n" + prepare.INSTRUCTION
        self.assertEqual(prepare.arc_prompt(self.rows[0]), expected)
        five = QuestionOnlyRow(question="Five?", choices={"label": list("ABCDE"), "text": list("abcde")})
        self.assertIn("E. e\n", prepare.arc_prompt(five))
        panel, candidates, _ = self.build()
        self.assertNotIn("SECRET", json.dumps(panel + candidates))

    def test_membership_is_ranked_seed_zero_and_all_candidates_retained(self):
        panel, candidates, audit = self.build()
        expected = sorted(self.rows, key=lambda r: prepare.rank(0, prepare.PURPOSE, r["id"]))[:3]
        self.assertEqual([r["source_id"] for r in panel], [r["id"] for r in expected])
        self.assertEqual({r["source_id"] for r in candidates}, {r["id"] for r in self.rows})
        self.assertEqual(audit["selection_counts"], {"eligible": 5, "selected": 3})
        self.assertEqual(self.build()[0], panel)
        self.assertTrue(all(r["source"] == prepare.SOURCE and r["source_split"] == "test" for r in panel))
        self.assertTrue(all(r["group_id"] == "allenai/ai2_arc|" + r["source_id"] for r in panel))

    def test_exact_parent_overlap_is_excluded_before_ranking(self):
        first = min(self.rows, key=lambda r: prepare.rank(0, prepare.PURPOSE, r["id"]))
        training = [dict(id="old", source="chat", prompt="  " + first["question"].upper() + "  ")]
        panel, candidates, audit = self.build(training)
        self.assertNotIn(first["id"], {r["source_id"] for r in panel})
        excluded = next(r for r in candidates if r["source_id"] == first["id"])
        self.assertEqual(excluded["selection_status"], "excluded_parent_overlap")
        self.assertEqual(excluded["old_parent_training_matches"][0]["id"], "old")
        self.assertEqual(audit["selection_counts"]["excluded_parent_overlap"], 1)

    def test_near_overlap_rule_and_provenance(self):
        text = " ".join(f"word{i}" for i in range(120))
        index = prepare.OverlapIndex([dict(id="near", source="train", prompt=text)])
        found = index.find(text + " extra")
        self.assertEqual(found[0]["match"], "fivegram_jaccard")
        self.assertGreaterEqual(found[0]["similarity"], .90)
        self.assertEqual(found[0]["id"], "near")
        self.assertEqual(index.find("unrelated words here"), [])

    def test_frozen_ids_hashes_and_token_limit(self):
        row = self.build()[0][0]
        self.assertEqual(row["input_ids"], prepare.encode_prompt(self.tokenizer, row["prompt"], "native_chat"))
        self.assertEqual(row["input_ids_sha256"], prepare.ids_hash(row["input_ids"]))
        self.assertEqual(row["prompt_sha256"], prepare.text_hash(row["prompt"]))
        with self.assertRaisesRegex(ValueError, "untruncated"):
            self.build(max_prompt_tokens=2)
        with self.assertRaisesRegex(ValueError, "seed 0"):
            self.build(seed=1)

    def test_incomplete_candidate_set_and_exhausted_pool_fail(self):
        with self.assertRaisesRegex(ValueError, "unique source IDs"):
            prepare.build_panel(self.rows[:-1], [], self.tokenizer, "/fake", expected_candidates=8)
        training = [dict(id=str(i), source="train", prompt=r["question"]) for i, r in enumerate(self.rows)]
        with self.assertRaisesRegex(ValueError, "eligible requests"):
            self.build(training)

    def test_expanded_audit_covers_full_test_and_group_ids(self):
        panel, candidates, _ = self.build()
        unselected = next(r for r in candidates if r["id"] not in {p["id"] for p in panel})
        training = [dict(id="new-a", source="new", split="train", prompt=panel[0]["prompt"]),
                    dict(id="new-b", source="new", split="train", prompt="Different wording", group_id=unselected["group_id"])]
        report = prepare.audit_expanded_training(panel, candidates, training)
        self.assertEqual(report["matched_selected_requests"], 1)
        self.assertEqual(report["matched_full_test_requests"], 2)
        self.assertEqual(len(report["matches"]), 2)

    def test_parent_requests_resolve_actual_manifest_replacements(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            original = dict(id="original", source="code", prompt="unused")
            replacement = dict(id="replacement", source="code", prompt="used")
            for name, rows in (("train.jsonl", [original]), ("reserves.jsonl", [replacement]),
                               ("manifest.jsonl", [dict(id="replacement", source="code", split="train")])):
                (root / name).write_text("".join(json.dumps(r) + "\n" for r in rows))
            self.assertEqual(prepare.actual_parent_training(root, root / "manifest.jsonl", expected=1), [replacement])

    def test_runtime_guard_checks_versions_path_and_frozen_files(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            token_file = root / "tokenizer.json"
            token_file.write_text("tokenizer content")
            core = root / "sources.json"
            core.write_text(json.dumps({"tokenizer": {"tokenizer_path": str(root),
                             "transformers_version": "5.9.0", "tokenizers_version": "0.22.2",
                             "files": [prepare.file_identity(token_file)]}}))
            versions = {"transformers": "5.9.0", "tokenizers": "0.22.2"}
            prepare.validate_tokenizer_runtime(core, root, installed=versions)
            with self.assertRaisesRegex(ValueError, "runtime mismatch"):
                prepare.validate_tokenizer_runtime(core, root, installed={**versions, "transformers": "4.57.1"})
            with self.assertRaisesRegex(ValueError, "path"):
                prepare.validate_tokenizer_runtime(core, root / "other", installed=versions)
            token_file.write_text("different tokenizer")
            with self.assertRaisesRegex(ValueError, "file differs"):
                prepare.validate_tokenizer_runtime(core, root, installed=versions)

    def test_reencoding_changes_only_tokenizer_dependent_fields(self):
        panel, candidates, _ = self.build()

        class OtherTokenizer(FakeTokenizer):
            def encode(self, text, add_special_tokens):
                return [42] + super().encode(text, add_special_tokens)

        token_fields = {"input_ids", "input_ids_sha256", "prompt_tokens", "tokenizer_path"}
        for original in (panel, candidates):
            repaired = prepare.reencode_records(original, OtherTokenizer(), "/fake/ouro")
            self.assertEqual([{k: v for k, v in r.items() if k not in token_fields} for r in repaired],
                             [{k: v for k, v in r.items() if k not in token_fields} for r in original])
            self.assertTrue(all(a["input_ids"] != b["input_ids"] for a, b in zip(repaired, original)))
            self.assertTrue(all(a["input_ids_sha256"] == prepare.ids_hash(a["input_ids"]) for a in repaired))

    def test_effective_audit_uses_a_new_path_and_preserves_previous_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sources.json").write_text(json.dumps({"expanded_training_audit": {"status": "pending"}}))
            report = dict(status="passed", training_file={"path": "/train.jsonl", "sha256": "original"},
                          training_rows=16384, matched_selected_requests=0, matched_full_test_requests=0)
            first = prepare.save_expanded_audit(root, report)
            original_bytes = first.read_bytes()
            effective = {**report, "training_file": {"path": "/effective.jsonl", "sha256": "replacement"}}
            second = prepare.save_expanded_audit(root, effective, root / "overlap_report_effective_16k.json")
            self.assertNotEqual(first, second)
            self.assertEqual(first.read_bytes(), original_bytes)
            self.assertEqual(json.loads(second.read_text())["training_file"]["sha256"], "replacement")
            sources = json.loads((root / "sources.json").read_text())
            self.assertEqual(sources["expanded_training_audit"]["report"]["path"], str(second))
            self.assertEqual(sources["expanded_training_audit_history"][0]["training_file"]["sha256"], "original")
            with self.assertRaises(FileExistsError):
                prepare.save_expanded_audit(root, effective)
            self.assertEqual(first.read_bytes(), original_bytes)

    def test_reserve_audit_stays_separate_from_actual_train_count(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / "sources.json").write_text("{}")
            report = dict(status="passed", training_file={"path": "/train.jsonl", "sha256": "original"},
                          training_rows=16384, matched_selected_requests=0, matched_full_test_requests=0,
                          reserves=dict(training_rows=1152, matched_selected_requests=0, matched_full_test_requests=0,
                                        file={"path": "/reserves.jsonl", "sha256": "reserve"}))
            prepare.save_expanded_audit(root, report)
            stored = json.loads((root / "sources.json").read_text())["expanded_training_audit"]
            self.assertEqual(stored["training_rows"], 16384)
            self.assertEqual(stored["reserves"]["training_rows"], 1152)


if __name__ == "__main__":
    unittest.main()
