"""Freeze the local 1064-request v1 evaluation panel without running a model.

Keep the existing 300 GSM8K and 300 ShareGPT requests and their input tokens.
Add all 164 HumanEval completion prompts and a seed-0 sample of 300 MBPP full
test tasks. This is a speed/token-identity panel, not a task-accuracy harness.
"""

from __future__ import annotations

import argparse
import ast
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
from importlib.metadata import version
import json
from pathlib import Path

from .prepare_protocol_data import CHAT, CODE, GSM, arrow_rows, normalize, rank, read_jsonl


HUMAN = "openai/openai_humaneval"
MBPP = "google-research-datasets/mbpp"
ROOT = Path("/experiments/loop_drafter")
MODEL = Path("/models/Ouro-1.4B")
HUMAN_PATH = Path("/datasets/huggingface/openai_humaneval/openai_humaneval/0.0.0/7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544/openai_humaneval-test.arrow")
MBPP_PATH = Path("/datasets/huggingface/mbpp/full/0.0.0/4b9416760167ee3c/mbpp-test.arrow")
COUNTS = {GSM: 300, HUMAN: 164, MBPP: 300, CHAT: 300}
TEMPLATES = {
    "gsm8k": {"prompt_format": "native_chat", "description": "Existing question and native chat-template tokenization unchanged; no added demonstrations."},
    "sharegpt": {"prompt_format": "native_chat", "description": "Existing first-user request and native chat-template tokenization unchanged."},
    "humaneval": {"prompt_format": "raw_completion", "description": "Official prompt verbatim, preserving trailing whitespace; no chat wrapper, extra BOS, solution, or hidden tests."},
    "mbpp": {"prompt_format": "raw_completion", "description": "Zero-shot adaptation of the official task/tests/[BEGIN] completion format; all three public test_list assertions, no demonstrations or reference code."},
}


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def ids_hash(ids):
    return text_hash(json.dumps(ids, separators=(",", ":")))


def file_identity(path):
    path = Path(path).resolve()
    hasher = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(block)
    return {"path": str(path), "size_bytes": path.stat().st_size, "sha256": hasher.hexdigest()}


def encode_prompt(tokenizer, prompt, prompt_format):
    if prompt_format == "raw_completion":
        return list(tokenizer.encode(prompt, add_special_tokens=False))
    if prompt_format != "native_chat":
        raise ValueError(f"unsupported prompt_format: {prompt_format}")
    if getattr(tokenizer, "chat_template", None):
        return list(tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        ))
    return list(tokenizer.encode(prompt, add_special_tokens=True))


def mbpp_prompt(row):
    if len(row["test_list"]) != 3:
        raise ValueError("MBPP full tasks must supply three public tests")
    return ("You are an expert Python programmer, and here is your task: " + row["text"]
            + "\nYour code should pass these tests:\n\n" + "\n".join(row["test_list"])
            + "\n[BEGIN]\n")


def human_docstring(prompt):
    """Additional exact-text comparison; no attempt at semantic decontamination."""
    try:
        tree = ast.parse(prompt)
    except SyntaxError:
        return None
    function = next((node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))), None)
    return ast.get_docstring(function, clean=False) if function is not None else None


def build_panel(old_eval, human_rows, mbpp_rows, tokenizer, tokenizer_path, *, seed=0,
                max_prompt_tokens=2048):
    if seed != 0:
        raise ValueError("v1 freezes seed 0")
    if {r["task_id"] for r in human_rows} != {f"HumanEval/{i}" for i in range(164)} or len(human_rows) != 164:
        raise ValueError("HumanEval must contain the complete 164 unique official task IDs")
    if {r["task_id"] for r in mbpp_rows} != set(range(11, 511)) or len(mbpp_rows) != 500:
        raise ValueError("MBPP must be the full test split, task IDs 11 through 510")
    records = []
    for source, panel in ((GSM, "gsm8k"), (CHAT, "sharegpt")):
        rows = [r for r in old_eval if r["source"] == source]
        if len(rows) != 300 or any(r["split"] != "eval" for r in rows):
            raise ValueError(f"existing {source} panel must contain 300 eval requests")
        for row in rows:
            records.append({**row, "panel": panel, "prompt_format": "native_chat",
                            "benchmark_text": row["prompt"], "template_id": panel + "_legacy_native_chat_v1"})
    for row in sorted(human_rows, key=lambda r: int(r["task_id"].split("/")[-1])):
        task_id = row["task_id"]
        records.append({"id": "openai__openai_humaneval:test:" + task_id,
                        "source": HUMAN, "source_id": task_id, "source_split": "test", "split": "eval",
                        "group_id": HUMAN + "|test|" + task_id, "domain": "code", "panel": "humaneval",
                        "prompt": row["prompt"], "benchmark_text": row["prompt"],
                        "benchmark_docstring": human_docstring(row["prompt"]),
                        "prompt_format": "raw_completion", "template_id": "humaneval_verbatim_completion_v1"})
    selected = sorted(mbpp_rows, key=lambda r: rank(seed, "evaluation_v1:mbpp", str(r["task_id"])))[:300]
    for row in selected:
        task_id = str(row["task_id"])
        records.append({"id": "google-research-datasets__mbpp:test:" + task_id,
                        "source": MBPP, "source_id": task_id, "source_split": "test", "split": "eval",
                        "group_id": MBPP + "|test|" + task_id, "domain": "code", "panel": "mbpp",
                        "prompt": mbpp_prompt(row), "benchmark_text": row["text"],
                        "prompt_format": "raw_completion", "template_id": "mbpp_zero_shot_three_public_tests_v1"})
    if len({r["id"] for r in records}) != len(records) or Counter(r["source"] for r in records) != Counter(COUNTS):
        raise ValueError("invalid v1 panel IDs or source counts")
    for row in records:
        ids = encode_prompt(tokenizer, row["prompt"], row["prompt_format"])
        if not 1 <= len(ids) <= max_prompt_tokens:
            raise ValueError(f"prompt needs {len(ids)} tokens, exceeding the untruncated limit: {row['id']}")
        row.update(input_ids=ids, prompt_tokens=len(ids), tokenizer_path=str(Path(tokenizer_path).resolve()),
                   prompt_sha256=text_hash(row["prompt"]), input_ids_sha256=ids_hash(ids), selection_seed=seed)
    return records


def overlap_audit(panel, training_rows, reserve_rows, tokenizer):
    """Report normalized exact matches only, keeping actual train and reserves apart."""
    report = {"normalization": "Unicode NFKC, casefold, whitespace collapse",
              "scope": "Exact text/token comparisons only; no semantic decontamination or pretraining-overlap claim.",
              "training_rows": len(training_rows), "reserve_rows": len(reserve_rows), "comparisons": {}}
    for name, rows in (("actual_training", training_rows), ("reserves", reserve_rows)):
        raw_index, formatted_index, token_index = defaultdict(list), defaultdict(list), defaultdict(list)
        for row in rows:
            identity = {"id": row["id"], "source": row["source"]}
            raw_index[normalize(row["prompt"])].append(identity)
            if getattr(tokenizer, "chat_template", None):
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": row["prompt"]}], tokenize=False, add_generation_prompt=True,
                )
            else:
                formatted = row["prompt"]
            formatted_index[normalize(formatted)].append(identity)
            token_index[ids_hash(encode_prompt(tokenizer, row["prompt"], "native_chat"))].append(identity)
        matches = []
        for row in panel:
            for field in ("prompt", "benchmark_text", "benchmark_docstring"):
                text = row.get(field)
                if text and normalize(text) in raw_index:
                    matches.append({"eval_id": row["id"], "source": row["source"], "field": field,
                                    "train_matches": raw_index[normalize(text)]})
            formatted = row["prompt"]
            if row["prompt_format"] == "native_chat" and getattr(tokenizer, "chat_template", None):
                formatted = tokenizer.apply_chat_template(
                    [{"role": "user", "content": formatted}], tokenize=False, add_generation_prompt=True,
                )
            for field, values in (("serialized_prompt", formatted_index.get(normalize(formatted), [])),
                                  ("input_ids", token_index.get(row["input_ids_sha256"], []))):
                if values:
                    matches.append({"eval_id": row["id"], "source": row["source"], "field": field,
                                    "train_matches": values})
        report["comparisons"][name] = {"matched_eval_requests": len({r["eval_id"] for r in matches}),
                                       "matches": matches}
    index, within = defaultdict(list), []
    for row in panel:
        key = normalize(row["prompt"])
        if key in index:
            within.append({"id": row["id"], "matching_ids": index[key].copy()})
        index[key].append(row["id"])
    report["within_panel_normalized_prompt_duplicates"] = within
    return report


def source_metadata(path, source, dataset_config, url):
    path = Path(path)
    info_path = path.parent / "dataset_info.json"
    info = json.loads(info_path.read_text()) if info_path.exists() else {}
    return {"source": source, "dataset_config": dataset_config, "source_split": "test", "official_url": url,
            "file": file_identity(path), "local_cache_fingerprint": path.parent.name,
            "dataset_version": info.get("version"), "hf_revision": None,
            "revision_note": "Local cached artifact is identified by file digest; cache fingerprint is not claimed as a Hub commit."}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--old-prompts", type=Path, default=ROOT / "prompts")
    parser.add_argument("--training-manifest", type=Path, default=ROOT / "capture_fp32/manifest.jsonl")
    parser.add_argument("--humaneval", type=Path, default=HUMAN_PATH)
    parser.add_argument("--mbpp", type=Path, default=MBPP_PATH)
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--out", type=Path, default=ROOT / "evaluation_v1")
    parser.add_argument("--seed", type=int, choices=[0], default=0)
    parser.add_argument("--max-prompt-tokens", type=int, default=2048)
    args = parser.parse_args()
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("evaluation freeze requires a fresh directory; existing artifacts are never overwritten")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    old_eval = read_jsonl(args.old_prompts / "eval.jsonl")
    train = read_jsonl(args.old_prompts / "train.jsonl")
    reserves = read_jsonl(args.old_prompts / "reserves.jsonl")
    manifest = [row for row in read_jsonl(args.training_manifest) if row["split"] == "train"]
    if len(manifest) != 8192 or len({r["id"] for r in manifest}) != 8192:
        raise ValueError("actual training manifest must contain 8192 distinct trajectories")
    lookup = {r["id"]: r for r in train + reserves}
    actual_train = []
    for entry in manifest:
        row = lookup.get(entry["id"])
        if row is None or row["source"] != entry["source"]:
            raise ValueError("training manifest ID/source is absent from prepared train/reserves")
        actual_train.append(row)
    human, mbpp = arrow_rows(args.humaneval), arrow_rows(args.mbpp)
    panel = build_panel(old_eval, human, mbpp, tokenizer, args.model,
                        seed=args.seed, max_prompt_tokens=args.max_prompt_tokens)
    # Validate legacy inputs independently against the existing capture adapter.
    from .capture import prompt_ids
    legacy_lookup = {r["id"]: r for r in old_eval}
    legacy_checks = [row["input_ids"] == prompt_ids(tokenizer, legacy_lookup[row["id"]], 512)[0].tolist()
                     for row in panel if row["prompt_format"] == "native_chat"]
    if len(legacy_checks) != 600 or not all(legacy_checks):
        raise ValueError("legacy GSM8K/ShareGPT input IDs changed")
    audit = overlap_audit(panel, actual_train, reserves, tokenizer)
    source_stats = {}
    for source in COUNTS:
        rows = [r for r in panel if r["source"] == source]
        lengths = [r["prompt_tokens"] for r in rows]
        source_stats[source] = {"rows": len(rows), "min_prompt_tokens": min(lengths), "max_prompt_tokens": max(lengths),
                                "above_512": sum(length > 512 for length in lengths),
                                "source_ids": [r["source_id"] for r in rows]}
    sources = {"schema": "loop_drafter_evaluation_v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "seed": args.seed, "expected_counts": COUNTS, "summary": source_stats,
               "templates": TEMPLATES, "mbpp_selection": "First 300 task IDs sorted by BLAKE2b(seed|evaluation_v1:mbpp|task_id), 16-byte digest",
               "generation": {"max_prompt_tokens": args.max_prompt_tokens, "max_new_tokens": 256, "decoding": "greedy"},
               "legacy_input_ids_unchanged": {"checked": 600, "passed": True},
               "legacy_codealpaca": {"rows": 300, "path": str((args.old_prompts / "eval.jsonl").resolve()),
                                      "included_in_v1": False, "note": "Existing artifact retained; do not mix its results into v1 aggregates."},
               "tokenizer": {"tokenizer_path": str(args.model.resolve()), "has_chat_template": bool(tokenizer.chat_template),
                             "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
                             "files": [file_identity(args.model / name) for name in
                                       ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
                                       if (args.model / name).exists()]},
               "inputs": [file_identity(args.old_prompts / name) for name in
                          ("train.jsonl", "eval.jsonl", "reserves.jsonl", "sources.json")]
                         + [file_identity(args.training_manifest)],
               "new_sources": [source_metadata(args.humaneval, HUMAN, "openai_humaneval", "https://huggingface.co/datasets/openai/openai_humaneval"),
                               source_metadata(args.mbpp, MBPP, "full", "https://github.com/google-research/google-research/blob/master/mbpp/README.md")],
               "network_downloads": 0, "metric_scope": "Speed and token-identity diagnostics; no answer accuracy scoring."}
    args.out.mkdir(parents=True, exist_ok=True)
    prompt_path = args.out / "prompts.jsonl"
    with prompt_path.open("x", encoding="utf-8") as stream:
        for row in panel:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    sources["prompts_file"] = file_identity(prompt_path)
    for name, payload in (("sources.json", sources), ("overlap_report.json", audit)):
        with (args.out / name).open("x", encoding="utf-8") as stream:
            json.dump(payload, stream, ensure_ascii=False, indent=2)
            stream.write("\n")
    print(json.dumps({"out": str(args.out), "rows": len(panel), "source_counts": dict(Counter(r["source"] for r in panel)),
                      "max_prompt_tokens": max(r["prompt_tokens"] for r in panel),
                      "exact_matches_actual_train": audit["comparisons"]["actual_training"]["matched_eval_requests"],
                      "exact_matches_reserves": audit["comparisons"]["reserves"]["matched_eval_requests"]}))


if __name__ == "__main__":
    main()
