"""Freeze a separate ARC-Challenge explanation-generation speed panel on CPU.

The existing 1064-request evaluation_v1 panel is never modified or pooled here.
Selection uses source IDs, prompt lengths and overlap checks, never model output.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import sys

from .prepare_protocol_data import PromptIndex, fivegrams, normalize, rank, read_jsonl
from .prepare_v1_evaluation import MODEL, ROOT, encode_prompt, file_identity, ids_hash, source_metadata, text_hash


SOURCE = "allenai/ai2_arc:ARC-Challenge"
ARC_PATH = Path("/datasets/huggingface/arc/ARC-Challenge/0.0.0/55eda3b29fc4b7ed/arc-test.arrow")
OUT = ROOT / "evaluation_reasoning_v1"
PURPOSE = "reasoning_eval_v1"
INSTRUCTION = "Explain your reasoning briefly, then give the final option label."
TEMPLATE_ID = "arc_explanation_native_chat_v1"
CORE_SOURCES = ROOT / "evaluation_v1/sources.json"


def validate_tokenizer_runtime(core_sources, model, *, installed=None):
    """Do not freeze IDs with a tokenizer implementation unlike the core suite."""
    core = json.loads(Path(core_sources).read_text())["tokenizer"]
    installed = installed or {name: version(name) for name in ("transformers", "tokenizers")}
    for name in ("transformers", "tokenizers"):
        expected = core[name + "_version"]
        if installed[name] != expected:
            raise ValueError(f"tokenizer runtime mismatch: {name}={installed[name]}, core requires {expected}; "
                             "use python")
    if Path(model).resolve() != Path(core["tokenizer_path"]).resolve():
        raise ValueError("tokenizer path must match frozen core sources.json")
    for identity in core["files"]:
        if file_identity(identity["path"])["sha256"] != identity["sha256"]:
            raise ValueError("tokenizer file differs from frozen core: " + identity["path"])


def tokenizer_metadata(tokenizer, model):
    return {"tokenizer_path": str(Path(model).resolve()), "has_chat_template": bool(tokenizer.chat_template),
            "transformers_version": version("transformers"), "tokenizers_version": version("tokenizers"),
            "class": type(tokenizer).__name__, "module": type(tokenizer).__module__,
            "is_fast": bool(tokenizer.is_fast), "python_executable": sys.executable,
            "files": [file_identity(Path(model) / name) for name in
                      ("tokenizer.json", "tokenizer_config.json", "special_tokens_map.json", "vocab.json", "merges.txt")
                      if (Path(model) / name).exists()]}


def reencode_records(records, tokenizer, model):
    """Retain the chosen IDs, ordering, prompts and overlap decisions exactly."""
    updated = []
    for row in records:
        if row["prompt_sha256"] != text_hash(row["prompt"]):
            raise ValueError("saved prompt changed before tokenizer repair")
        ids = encode_prompt(tokenizer, row["prompt"], row["prompt_format"])
        if not ids or len(ids) > 2048:
            raise ValueError("repaired input would exceed untruncated token limit")
        updated.append({**row, "input_ids": ids, "input_ids_sha256": ids_hash(ids),
                        "prompt_tokens": len(ids), "tokenizer_path": str(Path(model).resolve())})
    return updated


def arc_test_rows(path):
    import pyarrow as pa
    with pa.memory_map(str(path), "r") as stream:
        return pa.ipc.open_stream(stream).read_all().select(["id", "question", "choices"]).to_pylist()


def arc_prompt(row):
    """Use only question/choices; labels are not necessarily A-D or four in number."""
    labels, texts = row["choices"]["label"], row["choices"]["text"]
    if len(labels) != len(texts) or not labels or len(set(labels)) != len(labels):
        raise ValueError("ARC choices require distinct original labels and matching texts")
    return ("Question: " + row["question"] + "\nChoices:\n"
            + "\n".join(f"{label}. {text}" for label, text in zip(labels, texts))
            + "\n" + INSTRUCTION)


class OverlapIndex(PromptIndex):
    """The existing 5-gram/Jaccard rule, with matching request provenance."""

    def __init__(self, rows, threshold=.90):
        super().__init__(threshold=threshold)
        self.metadata = []
        self.exact_indices = defaultdict(list)
        for row in rows:
            seen = set()
            for field in ("prompt", "benchmark_text", "question"):
                text = row.get(field)
                if not isinstance(text, str) or not normalize(text) or normalize(text) in seen:
                    continue
                seen.add(normalize(text))
                self.exact_indices[normalize(text)].append(len(self.metadata))
                self.metadata.append({"id": row["id"], "source": row["source"], "field": field})
                self.add(text)

    def find(self, text):
        normalized = normalize(text)
        exact = set(self.exact_indices.get(normalized, ()))
        matches = [{**self.metadata[i], "match": "normalized_exact", "similarity": 1.0}
                   for i in sorted(exact)]
        grams = fivegrams(text)
        if len(grams) >= 8:
            shared = Counter(i for gram in grams for i in self.inverted.get(gram, ()))
            for i, count in sorted(shared.items()):
                if i in exact or count < 8:
                    continue
                similarity = count / (len(grams) + len(self.grams[i]) - count)
                if similarity >= self.threshold:
                    matches.append({**self.metadata[i], "match": "fivegram_jaccard", "similarity": similarity})
        return matches


def overlap_matches(row, index):
    matches = []
    for field in ("benchmark_text", "prompt"):
        for match in index.find(row[field]):
            matches.append({"eval_field": field, **match})
    return matches


def actual_parent_training(prompts_dir, manifest_path, expected=8192):
    train = read_jsonl(prompts_dir / "train.jsonl")
    reserves = read_jsonl(prompts_dir / "reserves.jsonl")
    lookup = {r["id"]: r for r in train + reserves}
    if len(lookup) != len(train) + len(reserves):
        raise ValueError("parent train/reserve request IDs must be unique")
    manifest = [r for r in read_jsonl(manifest_path) if r["split"] == "train"]
    if len(manifest) != expected or len({r["id"] for r in manifest}) != expected:
        raise ValueError(f"actual parent manifest must contain {expected} unique trajectories")
    selected = []
    for entry in manifest:
        row = lookup.get(entry["id"])
        if row is None or row["source"] != entry["source"]:
            raise ValueError("parent manifest ID/source is absent from prepared train/reserves")
        selected.append(row)
    return selected


def build_panel(rows, parent_training, tokenizer, tokenizer_path, *, count=300,
                expected_candidates=1172, seed=0, max_prompt_tokens=2048):
    if seed != 0:
        raise ValueError("reasoning v1 freezes seed 0")
    if len(rows) != expected_candidates or len({r["id"] for r in rows}) != expected_candidates:
        raise ValueError(f"ARC-Challenge test must contain {expected_candidates} unique source IDs")
    index = OverlapIndex(parent_training)
    candidates = []
    for original in rows:
        source_id = str(original["id"])
        prompt = arc_prompt(original)
        ids = encode_prompt(tokenizer, prompt, "native_chat")
        if not ids or len(ids) > max_prompt_tokens:
            raise ValueError(f"untruncated ARC input exceeds token limit: {source_id} ({len(ids)})")
        record = dict(id=f"allenai__ai2_arc:ARC-Challenge:test:{source_id}", source=SOURCE,
                      source_id=source_id, source_split="test", dataset_config="ARC-Challenge",
                      group_id=f"allenai/ai2_arc|{source_id}", split="eval", domain="reasoning",
                      panel="arc_challenge", prompt=prompt, benchmark_text=original["question"],
                      prompt_format="native_chat", template_id=TEMPLATE_ID, input_ids=ids,
                      prompt_tokens=len(ids), prompt_words=len(prompt.split()),
                      tokenizer_path=str(Path(tokenizer_path).resolve()), prompt_sha256=text_hash(prompt),
                      input_ids_sha256=ids_hash(ids), selection_seed=seed)
        matches = overlap_matches(record, index)
        candidates.append({**record, "old_parent_training_matches": matches,
                           "selection_status": "excluded_parent_overlap" if matches else "eligible"})
    ordered = sorted(candidates, key=lambda r: (rank(seed, PURPOSE, r["source_id"]), r["source_id"]))
    selected, seen_prompts = [], set()
    for row in ordered:
        if row["selection_status"] != "eligible":
            continue
        normalized = normalize(row["prompt"])
        if normalized in seen_prompts:
            row["selection_status"] = "excluded_duplicate_prompt"
        elif len(selected) < count:
            row["selection_status"] = "selected"
            seen_prompts.add(normalized)
            selected.append({key: value for key, value in row.items()
                             if key not in ("selection_status", "old_parent_training_matches")})
    if len(selected) != count:
        raise ValueError(f"only {len(selected)} independent eligible requests remain; need {count}")
    audit = {"scope": "Normalized exact and lexical 5-gram overlap only; no semantic or pretraining decontamination claim.",
             "normalization": "Unicode NFKC, casefold, whitespace collapse",
             "near_rule": "5-gram Jaccard >= 0.90 with at least 8 shared grams",
             "old_actual_parent_training_rows": len(parent_training), "test_candidates": len(candidates),
             "selection_counts": dict(Counter(r["selection_status"] for r in candidates)),
             "excluded_parent_overlaps": [{"source_id": r["source_id"], "matches": r["old_parent_training_matches"]}
                                          for r in candidates if r["old_parent_training_matches"]],
             "selected_parent_overlap_requests": 0,
             "expanded_training_audit": "pending; must audit the actual frozen 16K requests separately"}
    return selected, sorted(candidates, key=lambda r: r["source_id"]), audit


def audit_expanded_training(panel, candidates, training_rows):
    if len({r["id"] for r in training_rows}) != len(training_rows) or any(r["split"] != "train" for r in training_rows):
        raise ValueError("expanded train requests must have unique IDs and split=train")
    index = OverlapIndex(training_rows)
    selected_ids = {r["id"] for r in panel}
    matches = []
    by_group = defaultdict(list)
    for row in training_rows:
        if row.get("group_id"):
            by_group[row["group_id"]].append({"id": row["id"], "source": row["source"]})
    for row in candidates:
        text_matches = overlap_matches(row, index)
        group_matches = by_group.get(row["group_id"], [])
        if text_matches or group_matches:
            matches.append({"source_id": row["source_id"], "selected": row["id"] in selected_ids,
                            "text_matches": text_matches, "group_matches": group_matches})
    return {"training_rows": len(training_rows), "test_candidates": len(candidates),
            "selected_requests": len(panel), "matched_selected_requests": sum(r["selected"] for r in matches),
            "matched_full_test_requests": len(matches), "matches": matches,
            "scope": "Frozen train-request list; exact/group and 5-gram Jaccard >=0.90 only. No semantic decontamination claim.",
            "capture_replacements": "If capture replaces a request with a reserve, rerun against the effective request list."}


def write_json(path, value):
    with Path(path).open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2)
        stream.write("\n")


def save_expanded_audit(out, report, output=None):
    """Every effective training list gets its own report; never replace evidence."""
    report_path = Path(output) if output is not None else out / "overlap_report_16k.json"
    write_json(report_path, report)
    sources_path = out / "sources.json"
    sources = json.loads(sources_path.read_text())
    previous = sources.get("expanded_training_audit", {})
    if "report" in previous:
        sources.setdefault("expanded_training_audit_history", []).append(previous)
    sources["expanded_training_audit"] = {"status": report["status"], "report": file_identity(report_path),
                                         "training_file": report["training_file"],
                                         "training_rows": report["training_rows"],
                                         "matched_selected_requests": report["matched_selected_requests"],
                                         "matched_full_test_requests": report["matched_full_test_requests"]}
    if "reserves" in report:
        sources["expanded_training_audit"]["reserves"] = {
            key: report["reserves"][key] for key in
            ("training_rows", "matched_selected_requests", "matched_full_test_requests", "file")}
    temporary = sources_path.with_suffix(".json.tmp")
    write_json(temporary, sources)
    temporary.replace(sources_path)
    return report_path


def repair_tokenizer(out, tokenizer, model, core_sources):
    sources_path = out / "sources.json"
    sources = json.loads(sources_path.read_text())
    previous = {"tokenizer": sources["tokenizer"], "prompts_file": sources["prompts_file"],
                "candidates_file": sources["candidates_file"]}
    updated = {}
    for name, identity_key in (("prompts.jsonl", "prompts_file"), ("candidates.jsonl", "candidates_file")):
        path = out / name
        if file_identity(path)["sha256"] != sources[identity_key]["sha256"]:
            raise ValueError("saved data differs from sources.json before repair: " + name)
        updated[name] = reencode_records(read_jsonl(path), tokenizer, model)
    panel, candidates = updated["prompts.jsonl"], updated["candidates.jsonl"]
    if [r["source_id"] for r in panel] != sources["selection"]["selected_source_ids"]:
        raise ValueError("selected source IDs differ from the frozen selection")
    if [r["source_id"] for r in candidates] != sources["selection"]["candidate_source_ids"]:
        raise ValueError("candidate source IDs differ from the frozen selection")
    sources["tokenizer"] = tokenizer_metadata(tokenizer, model)
    sources["core_tokenizer_reference"] = file_identity(core_sources)
    sources["tokenizer_repair"] = {"at": datetime.now(timezone.utc).isoformat(),
                                   "reason": "Align tokenizer runtime with frozen core and actual model execution",
                                   "previous": previous, "source_ids_and_prompts_preserved": True}
    lengths = [r["prompt_tokens"] for r in panel]
    sources["summary"].update(min_prompt_tokens=min(lengths), max_prompt_tokens=max(lengths),
                              mean_prompt_tokens=sum(lengths) / len(lengths),
                              candidate_max_prompt_tokens=max(r["prompt_tokens"] for r in candidates))
    for name, rows in updated.items():
        path = out / name
        temporary = path.with_suffix(path.suffix + ".repair.tmp")
        with temporary.open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
        temporary.replace(path)
    sources["prompts_file"] = file_identity(out / "prompts.jsonl")
    sources["candidates_file"] = file_identity(out / "candidates.jsonl")
    temporary = sources_path.with_suffix(".json.repair.tmp")
    write_json(temporary, sources)
    temporary.replace(sources_path)
    print(json.dumps({"out": str(out), "tokenizer_repaired": True, "source_ids_and_prompts_preserved": True,
                      **sources["summary"], "tokenizer": {k: v for k, v in sources["tokenizer"].items() if k != "files"}}))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--arc", type=Path, default=ARC_PATH)
    parser.add_argument("--old-prompts", type=Path, default=ROOT / "prompts")
    parser.add_argument("--training-manifest", type=Path, default=ROOT / "capture_fp32/manifest.jsonl")
    parser.add_argument("--model", type=Path, default=MODEL)
    parser.add_argument("--core-sources", type=Path, default=CORE_SOURCES)
    parser.add_argument("--out", type=Path, default=OUT)
    parser.add_argument("--repair-tokenizer", action="store_true", help="Re-encode the unused frozen panel without changing source IDs or prompts")
    parser.add_argument("--audit-train", type=Path, help="Audit an already-frozen 16K train.jsonl without changing panel membership")
    parser.add_argument("--audit-reserves", type=Path, help="Audit reserves separately from actual train requests")
    parser.add_argument("--audit-output", type=Path, help="New report path, e.g. overlap_report_effective_16k.json; existing files are never overwritten")
    args = parser.parse_args()
    if args.repair_tokenizer and args.audit_train:
        parser.error("tokenizer repair and expanded training audit must run separately")
    if (args.audit_reserves or args.audit_output) and not args.audit_train:
        parser.error("--audit-reserves and --audit-output require --audit-train")
    validate_tokenizer_runtime(args.core_sources, args.model)
    if args.audit_train:
        panel, candidates = read_jsonl(args.out / "prompts.jsonl"), read_jsonl(args.out / "candidates.jsonl")
        training = read_jsonl(args.audit_train)
        if len(training) != 16384:
            raise ValueError("expanded frozen training list must contain exactly 16384 requests")
        report = audit_expanded_training(panel, candidates, training)
        if args.audit_reserves:
            reserves = read_jsonl(args.audit_reserves)
            report["reserves"] = audit_expanded_training(panel, candidates, reserves)
            report["reserves"]["file"] = file_identity(args.audit_reserves)
            report["reserves"]["scope"] = "Reserve candidates only, not added to the actual train count. Exact/group and 5-gram Jaccard >=0.90."
        matches = report["matched_full_test_requests"] + report.get("reserves", {}).get("matched_full_test_requests", 0)
        report.update(created_at=datetime.now(timezone.utc).isoformat(),
                      status="passed" if not matches else "overlap_detected",
                      training_file=file_identity(args.audit_train), prompts_file=file_identity(args.out / "prompts.jsonl"))
        report_path = save_expanded_audit(args.out, report, args.audit_output)
        summary = {key: value for key, value in report.items() if key not in ("matches", "reserves")}
        if "reserves" in report:
            summary["reserves"] = {key: value for key, value in report["reserves"].items() if key != "matches"}
        print(json.dumps({"report_path": str(report_path), **summary}))
        if matches:
            raise SystemExit(1)
        return
    if not args.repair_tokenizer and args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError("reasoning evaluation freeze requires a fresh directory")
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), trust_remote_code=True, local_files_only=True)
    if args.repair_tokenizer:
        repair_tokenizer(args.out, tokenizer, args.model, args.core_sources)
        return
    training = actual_parent_training(args.old_prompts, args.training_manifest)
    panel, candidates, audit = build_panel(arc_test_rows(args.arc), training, tokenizer, args.model)
    lengths = [r["prompt_tokens"] for r in panel]
    sources = {"schema": "loop_drafter_reasoning_evaluation_v1", "created_at": datetime.now(timezone.utc).isoformat(),
               "seed": 0, "expected_counts": {SOURCE: 300}, "panel": "arc_challenge",
               "selection": {"purpose": PURPOSE, "rank": "BLAKE2b(seed|purpose|source_id), 16-byte digest",
                             "candidate_count": 1172, "candidate_source_ids": [r["source_id"] for r in candidates],
                             "selected_source_ids": [r["source_id"] for r in panel],
                             "before_ranking": "exclude overlap with actual old 8192 parent trajectories",
                             "model_output_used": False},
               "template": {"id": TEMPLATE_ID, "prompt_format": "native_chat", "instruction": INSTRUCTION,
                            "format": "Question: {question}\\nChoices:\\n{original label}. {choice text} [each choice]\\n" + INSTRUCTION,
                            "answer_fields_read": False, "original_choice_labels_preserved": True},
               "summary": {"rows": len(panel), "min_prompt_tokens": min(lengths), "max_prompt_tokens": max(lengths),
                           "mean_prompt_tokens": sum(lengths) / len(lengths),
                           "candidate_max_prompt_tokens": max(r["prompt_tokens"] for r in candidates)},
               "generation": {"max_prompt_tokens": 2048, "max_new_tokens": 256, "decoding": "greedy",
                              "stop": "native EOS only; no option-letter, newline, or task-specific stop"},
               "metric_scope": "Explanation-generation speed and token-identity diagnostics; no answer accuracy scoring.",
               "aggregation": "Report this 300-request panel separately; never pool it into the historical 1064-request core aggregate.",
               "checkpoint_selection": "fixed final epoch; no test-based checkpoint or sample selection",
               "expanded_training_audit": {"status": "pending", "expected_report": "overlap_report_16k.json",
                                           "command_option": "--audit-train /experiments/loop_drafter/data_v2/prompts/train.jsonl"},
               "tokenizer": tokenizer_metadata(tokenizer, args.model),
               "core_tokenizer_reference": file_identity(args.core_sources),
               "inputs": [file_identity(args.old_prompts / name) for name in ("train.jsonl", "reserves.jsonl")]
                         + [file_identity(args.training_manifest)],
               "source": source_metadata(args.arc, SOURCE, "ARC-Challenge", "https://huggingface.co/datasets/allenai/ai2_arc"),
               "network_downloads": 0}
    args.out.mkdir(parents=True, exist_ok=True)
    for name, rows in (("prompts.jsonl", panel), ("candidates.jsonl", candidates)):
        with (args.out / name).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    sources["prompts_file"] = file_identity(args.out / "prompts.jsonl")
    sources["candidates_file"] = file_identity(args.out / "candidates.jsonl")
    write_json(args.out / "sources.json", sources)
    write_json(args.out / "overlap_report.json", audit)
    print(json.dumps({"out": str(args.out), **sources["summary"], "selection_counts": audit["selection_counts"]}))


if __name__ == "__main__":
    main()
