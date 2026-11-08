"""Copy the frozen v1 core minus MBPP, without tokenization or new sampling."""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path

from .prepare_v1_evaluation import CHAT, COUNTS as V1_COUNTS, GSM, HUMAN, MBPP, ROOT, file_identity, ids_hash, text_hash


COUNTS = {GSM: 300, HUMAN: 164, CHAT: 300}


def select_core_bytes(payload):
    """Preserve each retained JSONL line byte for byte and in original order."""
    original, kept, kept_lines, excluded = [], [], [], []
    for line in payload.splitlines(keepends=True):
        if not line.strip():
            continue
        row = json.loads(line)
        original.append(row)
        if row["source"] == MBPP:
            excluded.append(row["id"])
        elif row["source"] in COUNTS:
            kept.append(row)
            kept_lines.append(line)
        else:
            raise ValueError("unexpected source in frozen v1 panel")
    if Counter(r["source"] for r in original) != Counter(V1_COUNTS):
        raise ValueError("origin must be the complete 1064-request v1 core")
    if len({r["id"] for r in original}) != len(original) or any(r["split"] != "eval" for r in original):
        raise ValueError("origin requires distinct evaluation IDs")
    for row in kept:
        if row["prompt_sha256"] != text_hash(row["prompt"]) or row["input_ids_sha256"] != ids_hash(row["input_ids"]):
            raise ValueError("frozen prompt/input IDs failed their stored hashes")
    if Counter(r["source"] for r in kept) != Counter(COUNTS):
        raise ValueError("v2 requires GSM8K 300, HumanEval 164, ShareGPT 300")
    return b"".join(kept_lines), kept, excluded


def freeze(origin, out, reasoning):
    if out.exists() and any(out.iterdir()):
        raise FileExistsError("evaluation_v2 requires a fresh directory; historical panels are never overwritten")
    origin_sources_path = origin / "sources.json"
    origin_prompts_path = origin / "prompts.jsonl"
    source = json.loads(origin_sources_path.read_text())
    identity = file_identity(origin_prompts_path)
    if identity["sha256"] != source["prompts_file"]["sha256"]:
        raise ValueError("v1 prompt file differs from its frozen sources.json")
    payload, records, excluded = select_core_bytes(origin_prompts_path.read_bytes())
    reasoning_sources = json.loads((reasoning / "sources.json").read_text())
    reasoning_identity = file_identity(reasoning / "prompts.jsonl")
    if reasoning_identity["sha256"] != reasoning_sources["prompts_file"]["sha256"]:
        raise ValueError("independent reasoning panel differs from its frozen sources.json")
    summary = {}
    for name in COUNTS:
        rows = [r for r in records if r["source"] == name]
        lengths = [len(r["input_ids"]) for r in rows]
        summary[name] = {"rows": len(rows), "min_prompt_tokens": min(lengths), "max_prompt_tokens": max(lengths),
                         "source_ids": [r["source_id"] for r in rows]}
    metadata = {"schema": "loop_drafter_evaluation_v2", "created_at": datetime.now(timezone.utc).isoformat(),
                "seed": source["seed"], "expected_counts": COUNTS, "total_requests": len(records), "summary": summary,
                "selection": {"rule": "Drop MBPP records from evaluation_v1; preserve all other records and their relative order",
                              "new_sampling": False, "reencoded": False, "retained_jsonl_lines_byte_identical": True,
                              "retained_ids": [r["id"] for r in records],
                              "excluded_source": MBPP, "excluded_count": len(excluded), "excluded_ids": excluded},
                "origin": {"schema": source["schema"], "prompts_file": identity,
                           "sources_file": file_identity(origin_sources_path)},
                "tokenizer": source["tokenizer"],
                "templates": {k: v for k, v in source["templates"].items() if k != "mbpp"},
                "generation": {**source["generation"], "stop": "native EOS only; no task-specific stopping"},
                "checkpoint_selection": "fixed final epoch; no evaluation-based checkpoint selection",
                "metric_scope": source["metric_scope"],
                "aggregation": "764-request core only; do not mix with the historical 1064-request core or ARC aggregates",
                "separate_reasoning_panel": {"prompts_file": reasoning_identity,
                                             "sources_file": file_identity(reasoning / "sources.json"),
                                             "expected_counts": reasoning_sources["expected_counts"],
                                             "included_in_core": False},
                "preparation": "Copy existing frozen records; no tokenizer/model loading, generation, or network access"}
    out.mkdir(parents=True, exist_ok=True)
    with (out / "prompts.jsonl").open("xb") as stream:
        stream.write(payload)
    metadata["prompts_file"] = file_identity(out / "prompts.jsonl")
    with (out / "sources.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, ensure_ascii=False, indent=2)
        stream.write("\n")
    if file_identity(origin_prompts_path) != identity:
        raise RuntimeError("origin panel changed during preparation")
    return metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--origin", type=Path, default=ROOT / "evaluation_v1")
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation_v2")
    parser.add_argument("--reasoning", type=Path, default=ROOT / "evaluation_reasoning_v1")
    args = parser.parse_args()
    metadata = freeze(args.origin, args.out, args.reasoning)
    print(json.dumps({"out": str(args.out), "rows": metadata["total_requests"],
                      "source_counts": metadata["expected_counts"], "excluded_mbpp": metadata["selection"]["excluded_count"],
                      "prompts_sha256": metadata["prompts_file"]["sha256"], "retained_jsonl_lines_byte_identical": True}))


if __name__ == "__main__":
    main()
