"""CPU-only checks for prompt split isolation and benchmark exemptions."""

import json
from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from .prepare_protocol_data import (
    CHAT, CODE, GSM, PromptIndex, group_id, make_record, matching_benchmark,
    native_token_count, select_panels, stream_json_array, verify_panels,
)


class Tokenizer:
    chat_template = "native"

    def apply_chat_template(self, messages, **kwargs):
        assert kwargs == dict(tokenize=True, add_generation_prompt=True, return_dict=False)
        return [1] + messages[0]["content"].split() + [2, 3]


class ProtocolDataTest(unittest.TestCase):
    def test_original_conversation_and_split_namespaces(self):
        self.assertEqual(group_id(CHAT, "train", "abc_0"), group_id(CHAT, "conversation_pool", "abc_3"))
        self.assertNotEqual(group_id(GSM, "train", "12"), group_id(GSM, "test", "12"))

    def test_native_template_overhead_is_counted(self):
        self.assertEqual(native_token_count(Tokenizer(), "one two three"), 6)

    def test_near_duplicates_not_only_exact_strings(self):
        index = PromptIndex(.90)
        base = " ".join(f"item{i}" for i in range(100))
        index.add(base)
        self.assertTrue(index.matches(base.upper().replace(" ", "  ")))
        self.assertTrue(index.matches(base + " ending"))
        self.assertFalse(index.matches("entirely separate problem about a red circle"))

    def test_gsm_official_eval_self_match_only_exemption(self):
        row = make_record(GSM, "test", 1, "one official math question")
        index = PromptIndex(.85)
        index.add(row["prompt"])
        self.assertIsNone(matching_benchmark(row, "eval", {"gsm8k_test": index}))
        self.assertEqual(matching_benchmark(row, "train", {"gsm8k_test": index}), "gsm8k_test")
        self.assertEqual(matching_benchmark(row, "eval", {"math500": index}), "math500")

    def test_eval_first_old_pilot_and_reserves_are_disjoint(self):
        # Same original group must remain excluded even if its prompt changed.
        old = make_record(CODE, "train", 0, "old heldout original question")
        pool = [make_record(CODE, "train", i, f"candidate {i} unique task") for i in range(20)]
        pool[0]["prompt"] = "changed heldout text with the same source id"
        pools = {(CODE, "train"): pool}
        args = dict(train_counts={CODE: 5}, eval_per_source=3, reserve_per_source=2, max_tokens=512)
        panels, _, audit = select_panels(pools, Tokenizer(), [old], {}, **args)
        verify_panels(panels, {CODE: 5}, 3, 2)
        selected = [r for rows in panels.values() for r in rows]
        self.assertNotIn(old["group_id"], {r["group_id"] for r in selected})
        self.assertEqual(audit["selected_old_group_overlap"], 0)
        self.assertTrue(all(r["split"] == "train" and r["panel_role"] == "reserve" for r in panels["reserve"]))
        repeated, _, _ = select_panels(pools, Tokenizer(), [old], {}, **args)
        self.assertEqual(panels, repeated)

    def test_streaming_json_handles_large_record_and_limit(self):
        rows = [{"id": "a", "text": "中" * 300000}, {"id": "b"}]
        with TemporaryDirectory() as folder:
            path = Path(folder) / "rows.json"
            path.write_text(json.dumps(rows, ensure_ascii=False))
            self.assertEqual(list(stream_json_array(path)), list(enumerate(rows)))
            self.assertEqual(list(stream_json_array(path, 1)), [(0, rows[0])])


if __name__ == "__main__":
    unittest.main()
