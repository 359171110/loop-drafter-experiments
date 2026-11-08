"""Check removal of MBPP without altering retained inputs, ordering or whitespace."""

import json
import unittest

from . import prepare_v2_evaluation as prepare


class PrepareV2Tests(unittest.TestCase):
    def setUp(self):
        self.records = []
        for source, count in prepare.V1_COUNTS.items():
            for i in range(count):
                prompt = f"def task_{i}():\n    " if source == prepare.HUMAN else f"{source} request {i}"
                ids = [i + 1, i + 2]
                self.records.append(dict(id=f"{source}:{i}", source=source, source_id=str(i), split="eval",
                                         prompt=prompt, input_ids=ids, prompt_sha256=prepare.text_hash(prompt),
                                         input_ids_sha256=prepare.ids_hash(ids), extra_field={"preserve": True}))
        # Exercise arbitrary interleaving and noncanonical JSON whitespace.
        self.records = self.records[::2] + self.records[1::2]
        self.lines = [(json.dumps(r, separators=(", ", " : ")) + "\n").encode() for r in self.records]
        self.payload = b"".join(self.lines)

    def test_retains_exact_records_lines_and_relative_order(self):
        payload, kept, excluded = prepare.select_core_bytes(self.payload)
        expected = [r for r in self.records if r["source"] != prepare.MBPP]
        self.assertEqual(kept, expected)
        self.assertEqual(payload, b"".join(line for line, row in zip(self.lines, self.records) if row["source"] != prepare.MBPP))
        self.assertEqual(len(kept), 764)
        self.assertEqual(len(excluded), 300)
        self.assertFalse(any(r["source"] == prepare.MBPP for r in kept))
        self.assertTrue(all(r["prompt"].endswith("\n    ") for r in kept if r["source"] == prepare.HUMAN))

    def test_rejects_missing_records_and_duplicate_evaluation_ids(self):
        with self.assertRaisesRegex(ValueError, "complete 1064"):
            prepare.select_core_bytes(b"".join(self.lines[:-1]))
        records = [dict(r) for r in self.records]
        records[1]["id"] = records[0]["id"]
        with self.assertRaisesRegex(ValueError, "distinct evaluation IDs"):
            prepare.select_core_bytes(b"".join((json.dumps(r) + "\n").encode() for r in records))

    def test_rejects_changed_frozen_token_ids(self):
        records = [dict(r) for r in self.records]
        row = next(r for r in records if r["source"] != prepare.MBPP)
        row["input_ids"] = [999]
        with self.assertRaisesRegex(ValueError, "stored hashes"):
            prepare.select_core_bytes(b"".join((json.dumps(r) + "\n").encode() for r in records))


if __name__ == "__main__":
    unittest.main()
