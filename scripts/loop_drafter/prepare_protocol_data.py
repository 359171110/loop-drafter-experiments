"""Prepare the local 8192-train / 900-eval protocol without model generation.

Read source locations from the existing pilot sources.json. Only the native
tokenizer is loaded; no model, CUDA operation, network download, or answer text
is needed. Evaluation groups are selected before training and reserve groups.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime
import hashlib
import json
from pathlib import Path
import re
import unicodedata
from zoneinfo import ZoneInfo


GSM = "openai/gsm8k"
CODE = "sahil2801/CodeAlpaca-20k"
CHAT = "Aeala/ShareGPT_Vicuna_unfiltered"
TRAIN_COUNTS = {GSM: 3072, CODE: 3072, CHAT: 2048}
DOMAINS = {GSM: "math", CODE: "code", CHAT: "chat"}
BLOCKED = re.compile(
    r"<image>|<video>|!\[|data:image|<think>|/think|"
    r"https?://\S+\.(?:png|jpg|jpeg)|"
    r"\b(?:gsm8k|humaneval|mbpp|mmlu|hellaswag|winogrande)\b|"
    r"(?:attached|following|this)\s+(?:image|photo|screenshot)", re.I,
)


def read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def normalize(text):
    return " ".join(unicodedata.normalize("NFKC", text).casefold().split())


def fivegrams(text):
    tokens = re.findall(r"\w+|[^\w\s]", normalize(text))
    return {tuple(tokens[i:i + 5]) for i in range(max(0, len(tokens) - 4))}


class PromptIndex:
    """Exact and high-overlap prompt matching with an inverted 5-gram index."""

    def __init__(self, threshold=.90):
        self.threshold = threshold
        self.exact = set()
        self.grams = []
        self.inverted = defaultdict(set)

    def add(self, text):
        self.exact.add(normalize(text))
        grams = fivegrams(text)
        index = len(self.grams)
        self.grams.append(grams)
        for gram in grams:
            self.inverted[gram].add(index)

    def matches(self, text):
        if normalize(text) in self.exact:
            return True
        grams = fivegrams(text)
        if len(grams) < 8:
            return False
        shared = Counter(i for gram in grams for i in self.inverted.get(gram, ()))
        return any(count >= 8 and count / (len(grams) + len(self.grams[i]) - count) >= self.threshold
                   for i, count in shared.items())


def rank(seed, purpose, value):
    return hashlib.blake2b(f"{seed}|{purpose}|{value}".encode(), digest_size=16).digest()


def group_id(source, source_split, source_id):
    # All segments of a ShareGPT conversation share one group. GSM train/test
    # row numbers are different namespaces, including when importing pilot IDs.
    original = re.sub(r"_\d+$", "", str(source_id)) if source == CHAT else str(source_id)
    namespace = "conversation" if source == CHAT else source_split
    return f"{source}|{namespace}|{original}"


def make_record(source, source_split, source_id, prompt):
    source_id = str(source_id)
    return dict(id=f"{source.replace('/', '__')}:{source_split}:{source_id}", source=source,
                source_split=source_split, source_id=source_id,
                group_id=group_id(source, source_split, source_id), domain=DOMAINS[source],
                prompt=str(prompt).strip())


def native_token_count(tokenizer, prompt):
    # Keep this identical to capture.prompt_ids, without importing torch.
    if getattr(tokenizer, "chat_template", None):
        ids = tokenizer.apply_chat_template(
            [{"role": "user", "content": prompt}], tokenize=True,
            add_generation_prompt=True, return_dict=False,
        )
    else:
        ids = tokenizer.encode(prompt, add_special_tokens=True)
    return len(ids)


def stream_json_array(path, limit=0):
    decoder = json.JSONDecoder()
    buffer, position, count, eof = "", 0, 0, False
    with Path(path).open(encoding="utf-8") as stream:
        while not limit or count < limit:
            while position >= len(buffer) and not eof:
                chunk = stream.read(262144)
                eof, buffer, position = not chunk, chunk, 0
            while position < len(buffer) and buffer[position] in " \t\r\n[,":
                position += 1
            if position < len(buffer) and buffer[position] == "]":
                return
            try:
                row, end = decoder.raw_decode(buffer, position)
            except json.JSONDecodeError:
                if eof:
                    raise ValueError(f"incomplete JSON array: {path}")
                chunk = stream.read(262144)
                eof, buffer, position = not chunk, buffer[position:] + chunk, 0
                continue
            yield count, row
            count, position = count + 1, end
            if position > 262144:
                buffer, position = buffer[position:], 0


def arrow_rows(path):
    import pyarrow as pa
    with pa.memory_map(str(path), "r") as stream:
        return pa.ipc.open_stream(stream).read_all().to_pylist()


def load_candidates(pilot_sources, sharegpt_limit=0):
    paths = {row["source"]: Path(row["path"]) for row in pilot_sources["sources"]}
    pools = defaultdict(list)
    counts = Counter()
    rejected = Counter()

    def add(row):
        key = f"{row['source']}:{row['source_split']}"
        counts[key] += 1
        prompt = row["prompt"]
        if not 40 <= len(prompt) <= 20000:
            rejected[key + ":character_length"] += 1
        elif BLOCKED.search(prompt):
            rejected[key + ":media_thinking_or_benchmark_marker"] += 1
        elif sum(ord(c) > 127 for c in prompt) / len(prompt) > .2:
            rejected[key + ":non_english"] += 1
        else:
            pools[(row["source"], row["source_split"])].append(row)

    gsm_test = next(Path(row["path"]) for row in pilot_sources["benchmark_exclusion_sources"]
                    if row["source"] == "gsm8k_test")
    for split, path in (("train", paths[GSM]), ("test", gsm_test)):
        for index, row in enumerate(arrow_rows(path)):
            add(make_record(GSM, split, index, row["question"]))
    for index, row in enumerate(arrow_rows(paths[CODE])):
        prompt = (row.get("instruction") or "").strip()
        context = (row.get("input") or "").strip()
        if context and context.lower() not in ("<noinput>", "<no input>", "none", "n/a"):
            prompt += "\n\nInput:\n" + context
        add(make_record(CODE, "train", index, prompt))
    for index, row in stream_json_array(paths[CHAT], sharegpt_limit):
        counts[CHAT + ":scanned_json_records"] += 1
        source_id = str(row.get("id", index))
        if not source_id.endswith("_0"):
            rejected[CHAT + ":continuation_segment"] += 1
            continue
        turns = row.get("conversations", [])
        if not turns or turns[0].get("from") not in ("human", "user"):
            rejected[CHAT + ":no_initial_user"] += 1
            continue
        add(make_record(CHAT, "conversation_pool", source_id, turns[0].get("value", "")))
    return pools, counts, rejected


def benchmark_indexes(pilot_sources):
    indexes, metadata = {}, []
    for spec in pilot_sources["benchmark_exclusion_sources"]:
        index = PromptIndex(.85)
        rows = arrow_rows(spec["path"])
        for row in rows:
            if row.get(spec["column"]):
                index.add(str(row[spec["column"]]))
        indexes[spec["source"]] = index
        metadata.append({**spec, "rows": len(rows)})
    return indexes, metadata


def matching_benchmark(row, split, indexes):
    for name, index in indexes.items():
        if row["source"] == GSM and row["source_split"] == "test" and split == "eval" and name == "gsm8k_test":
            continue
        if index.matches(row["prompt"]):
            return name
    return None


def select_panels(pools, tokenizer, old_eval, benchmark, *, train_counts=None,
                  eval_per_source=300, reserve_per_source=128, seed=0, max_tokens=512):
    train_counts = TRAIN_COUNTS if train_counts is None else train_counts
    old_groups = {group_id(r["source"], r["source_split"], r["source_id"]) for r in old_eval}
    old_index = PromptIndex(.90)
    for row in old_eval:
        old_index.add(row["prompt"])
    selected_index, selected_groups = PromptIndex(.90), set()
    rejected, old_seen = Counter(), set()
    token_counts = {}
    panels = {"train": [], "eval": [], "reserve": []}

    for split in ("eval", "train", "reserve"):
        for source in train_counts:
            source_split = "test" if source == GSM and split == "eval" else ("conversation_pool" if source == CHAT else "train")
            quota = train_counts[source] if split == "train" else (eval_per_source if split == "eval" else reserve_per_source)
            pool = sorted(pools[(source, source_split)], key=lambda r: rank(seed, split, r["id"]))
            kept = 0
            for row in pool:
                if kept == quota:
                    break
                reason = None
                if row["group_id"] in old_groups:
                    old_seen.add(row["group_id"])
                    reason = "old_pilot_eval_group"
                elif old_index.matches(row["prompt"]):
                    reason = "old_pilot_eval_exact_or_near_prompt"
                elif row["group_id"] in selected_groups:
                    reason = "already_selected_group"
                elif selected_index.matches(row["prompt"]):
                    reason = "selected_exact_or_near_prompt"
                else:
                    match = matching_benchmark(row, split, benchmark)
                    if match:
                        reason = "benchmark_" + match
                if reason is not None:
                    rejected[f"{split}:{source}:{reason}"] += 1
                    continue
                if row["id"] not in token_counts:
                    token_counts[row["id"]] = native_token_count(tokenizer, row["prompt"])
                count = token_counts[row["id"]]
                if not 1 <= count <= max_tokens:
                    rejected[f"{split}:{source}:native_token_length"] += 1
                    continue
                panels[split].append({**row, "split": "train" if split == "reserve" else split,
                                      "panel_role": split, "prompt_tokens": count})
                selected_groups.add(row["group_id"])
                selected_index.add(row["prompt"])
                kept += 1
            if kept != quota:
                raise ValueError(f"Insufficient eligible {source} {split}: {kept}/{quota}")
            print(json.dumps({"selected": split, "source": source, "rows": kept}), flush=True)
        panels[split].sort(key=lambda r: rank(seed, "order", r["id"]))

    audit = dict(old_pilot_eval_rows=len(old_eval), old_pilot_eval_groups=len(old_groups),
                 old_groups_encountered_during_selection=len(old_seen),
                 policy="All old pilot eval groups and exact/near prompts excluded from train, eval, and reserves",
                 excluded_records=[{k: row[k] for k in ("id", "source", "source_split", "source_id")}
                                   for row in old_eval],
                 selected_old_group_overlap=len(selected_groups & old_groups))
    return panels, rejected, audit


def verify_panels(panels, train_counts, eval_per_source, reserve_per_source):
    ids, groups, prompts = set(), set(), set()
    for split, rows in panels.items():
        expected = train_counts if split == "train" else {s: eval_per_source if split == "eval" else reserve_per_source for s in train_counts}
        if Counter(row["source"] for row in rows) != Counter(expected):
            raise ValueError(f"Incorrect source quotas: {split}")
        for row in rows:
            key = normalize(row["prompt"])
            if row["id"] in ids or row["group_id"] in groups or key in prompts:
                raise ValueError("selected panels share an id, group, or normalized prompt")
            ids.add(row["id"])
            groups.add(row["group_id"])
            prompts.add(key)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot-sources", type=Path, required=True)
    parser.add_argument("--pilot-eval", type=Path)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--max-prompt-tokens", type=int, default=512)
    parser.add_argument("--eval-per-source", type=int, default=300)
    parser.add_argument("--reserve-per-source", type=int, default=128)
    parser.add_argument("--sharegpt-max-records", type=int, default=0, help="0 streams the whole local file")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    if min(args.max_prompt_tokens, args.eval_per_source, args.reserve_per_source) < 1 or args.sharegpt_max_records < 0:
        raise ValueError("invalid token/sample limits")
    names = {"train": "train.jsonl", "eval": "eval.jsonl", "reserve": "reserves.jsonl"}
    if any((args.out / name).exists() for name in (*names.values(), "sources.json")):
        raise FileExistsError("Use a fresh output directory; existing data is never overwritten")
    pilot = json.loads(args.pilot_sources.read_text())
    old_eval_path = args.pilot_eval or args.pilot_sources.parent / "pilot_eval.jsonl"
    old_eval = read_jsonl(old_eval_path)
    print(json.dumps({"stage": "reading_local_sources"}), flush=True)
    pools, counts, preliminary_rejections = load_candidates(pilot, args.sharegpt_max_records)
    print(json.dumps({"stage": "native_tokenizer_selection", "source_rows": dict(counts)}), flush=True)
    benchmark, benchmark_metadata = benchmark_indexes(pilot)
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True)
    panels, rejected, old_audit = select_panels(
        pools, tokenizer, old_eval, benchmark, eval_per_source=args.eval_per_source,
        reserve_per_source=args.reserve_per_source, seed=args.seed, max_tokens=args.max_prompt_tokens,
    )
    verify_panels(panels, TRAIN_COUNTS, args.eval_per_source, args.reserve_per_source)
    summary = {split: dict(rows=len(rows), sources=dict(Counter(r["source"] for r in rows)),
                          domains=dict(Counter(r["domain"] for r in rows)),
                          prompt_tokens=dict(min=min(r["prompt_tokens"] for r in rows),
                                             max=max(r["prompt_tokens"] for r in rows),
                                             mean=sum(r["prompt_tokens"] for r in rows) / len(rows)))
               for split, rows in panels.items()}
    metadata = dict(
        schema="loop_drafter_protocol_prompts_v1",
        created_at=datetime.now(ZoneInfo("Asia/Shanghai")).isoformat(),
        seed=args.seed, train_counts=TRAIN_COUNTS, summary=summary,
        inputs=dict(pilot_sources=str(args.pilot_sources.resolve()), old_pilot_eval=str(old_eval_path.resolve()),
                    tokenizer=str(args.model.resolve())),
        source_locations=[{key: row[key] for key in ("source", "path", "prompt_extraction", "license_evidence")
                           if key in row} for row in pilot["sources"]],
        source_rows_scanned=dict(counts),
        preliminary_rejections=dict(preliminary_rejections), selection_rejections=dict(rejected),
        old_pilot_exclusion=old_audit, benchmark_exclusion_sources=benchmark_metadata,
        tokenizer=dict(class_name=type(tokenizer).__name__, native_chat_template=bool(getattr(tokenizer, "chat_template", None)),
                       max_prompt_tokens=args.max_prompt_tokens, matches="capture.prompt_ids"),
        sampling_policy=dict(
            evaluation="GSM8K official test; CodeAlpaca/ShareGPT held-out source groups selected first, 300 per source by default",
            old_pilot="All old pilot32 eval groups and exact/near prompts remain excluded; no separate dev or test panel",
            order="Single seed, stable BLAKE2b ordering by source split and record id",
            isolation="No selected id/group/normalized prompt shared across train/eval/reserves; selected 5-gram Jaccard <0.90",
            benchmark="Exact or 5-gram Jaccard >=0.85 with >=8 shared grams excluded; official GSM8K eval is exempt only from GSM8K-test self-matching",
            sharegpt="First segment (_0), first user turn only; grouped by original conversation id",
            sharegpt_max_records=args.sharegpt_max_records,
            reserve="128 per source by default; split=train, panel_role=reserve; replace failed/short (<2 generated tokens) training captures within the same source",
            valid_capture="8192 is the exact main prompt count. Effective training trajectories require capture-time validation and same-source reserve promotion",
            generated_tokens="The downstream target capture is configured separately for FP32 and at most 256 generated tokens",
            residual_contamination="Exact/high-overlap filtering cannot guarantee removal of semantic paraphrases",
        ),
        execution=dict(network_downloads=False, gpu_used=False, model_generation=False, answer_text_written=False),
    )
    args.out.mkdir(parents=True, exist_ok=True)
    for split, rows in panels.items():
        with (args.out / names[split]).open("x", encoding="utf-8") as stream:
            for row in rows:
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    with (args.out / "sources.json").open("x", encoding="utf-8") as stream:
        json.dump(metadata, stream, indent=2, ensure_ascii=False)
        stream.write("\n")
    print(json.dumps({"out": str(args.out), "summary": summary,
                      "old_pilot_eval_excluded": old_audit["old_pilot_eval_rows"],
                      "selected_old_group_overlap": old_audit["selected_old_group_overlap"]}), flush=True)


if __name__ == "__main__":
    main()
