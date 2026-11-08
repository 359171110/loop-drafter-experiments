"""Freeze the 16K local prompt recipe and a CPU-only capture plan.

Legacy prompts are retained from the actual capture manifest. Reference answers,
APPS solutions/input_output, and multiple-choice gold labels are never serialized.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from datetime import datetime, timezone
from importlib.metadata import version
import json
from pathlib import Path
import re
import sys

from .prepare_protocol_data import CHAT, CODE, GSM, arrow_rows, fivegrams, normalize, rank, read_jsonl
from .prepare_v1_evaluation import (HUMAN_PATH, MBPP_PATH, MODEL, ROOT, encode_prompt,
                                   file_identity, human_docstring, ids_hash, mbpp_prompt, text_hash)


CACHE = Path('/datasets/huggingface')
MATH = 'hendrydong/hendrycks_math'
APPS = 'codeparrot/apps'
ARC_C = 'allenai/ai2_arc:ARC-Challenge'
ARC_E = 'allenai/ai2_arc:ARC-Easy'
HELLA = 'Rowan/hellaswag'
RACE = 'ehovy/race'
MATH_PATH = Path('/datasets/raw/hendrycks_math/train/data-00000-of-00001.arrow')
MATH_TEST = MATH_PATH.parent.parent / 'test/data-00000-of-00001.arrow'
GSM_TEST = Path('/datasets/raw/gsm8k/test/data-00000-of-00001.arrow')
ARC_BASE = CACHE / 'arc'
ARC_REVISION = '0.0.0/55eda3b29fc4b7ed'
HELLA_PATH = CACHE / 'hellaswag/default/0.0.0/1b5e44df7eeeac25/hellaswag-train.arrow'
RACE_PATH = CACHE / 'ehovy___race/all/0.0.0/2fec9fd81f1dc971569a9b729c43f2f0e6436637/race-train.arrow'
NEW_COUNTS = {MATH: 2048, APPS: 2048, ARC_C: 512, ARC_E: 512, HELLA: 1024, RACE: 2048}
OLD_COUNTS = {GSM: 3072, CODE: 3072, CHAT: 2048}
LEGACY_RESERVE_POLICY = ('Old three source reserve pools are backup only and are not automatically activated in v2. '
                         'All original 8192 prompt IDs are retained; any legacy recapture with fewer than two '
                         'generated tokens or another anomaly stops for investigation.')
CAPTURE_RESERVE_POLICY = ('Only a newly added v2 prompt that successfully generates fewer than two tokens may '
                          'be replaced by a reserve from the same one of the six new sources. OOM, infrastructure '
                          'failures, and other execution errors require retry or stop, never prompt replacement. '
                          'Use EOS/512, FP32 target and BF16 states; no generation performed here.')
MC_INSTRUCTION = 'Explain your reasoning briefly, then give the final option label.'
DIAGRAM = re.compile(r'\[asy\]|\\begin\{(?:asy|tikzpicture)\}|!\[|<image>|\bdiagram\b|'
                     r'\b(?:diagram|figure|graph|picture)\s+(?:below|above|shown)|'
                     r'\b(?:shown|illustrated)\s+in\s+the\s+(?:diagram|figure)', re.I)


class OverlapIndex:
    """Normalized exact/5-gram Jaccard matching, with source identities."""

    def __init__(self, threshold=.90):
        self.threshold = threshold
        self.exact = defaultdict(list)
        self.grams = []
        self.identities = []
        self.inverted = defaultdict(set)

    def add(self, text, identity):
        text = normalize(text)
        if not text:
            return
        index = len(self.grams)
        grams = fivegrams(text)
        self.exact[text].append(index)
        self.grams.append(grams)
        self.identities.append(identity)
        for gram in grams:
            self.inverted[gram].add(index)

    def matches(self, text):
        normalized = normalize(text)
        if normalized in self.exact:
            return [dict(self.identities[i], match='exact', jaccard=1.)
                    for i in self.exact[normalized]]
        grams = fivegrams(normalized)
        if not grams:
            return []
        shared = Counter(i for gram in grams for i in self.inverted.get(gram, ()))
        results = []
        for i, count in shared.items():
            union = len(grams) + len(self.grams[i]) - count
            if count / union >= self.threshold:
                results.append(dict(self.identities[i], match='fivegram_jaccard', jaccard=count / union))
        return sorted(results, key=lambda r: (r['id'], r.get('field', '')))


def variants(row):
    seen = set()
    for field in ('benchmark_text', 'prompt', 'benchmark_docstring'):
        text = row.get(field)
        if text and normalize(text) not in seen:
            seen.add(normalize(text))
            yield field, text


def add_record(index, row):
    for field, text in variants(row):
        index.add(text, {'id': row['id'], 'source': row['source'], 'field': field})


def matching_record(index, row):
    found, seen = [], set()
    for field, text in variants(row):
        for match in index.matches(text):
            key = (match['id'], match.get('field'), field)
            if key not in seen:
                found.append(dict(match, candidate_field=field))
                seen.add(key)
    return found


def mc_prompt(question, labels, choices):
    if not labels or len(labels) != len(choices) or len(set(labels)) != len(labels):
        raise ValueError('multiple-choice labels must be nonempty, unique and aligned')
    return ('Question: ' + question + '\nChoices:\n'
            + '\n'.join(f'{label}. {text}' for label, text in zip(labels, choices))
            + '\n' + MC_INSTRUCTION)


def arc_prompt(row):
    return mc_prompt(row['question'], row['choices']['label'], row['choices']['text'])


def apps_prompt(row):
    # Preserve the supplied skeleton's trailing indentation. No solutions or
    # input_output are consulted, including when the skeleton is absent.
    return ('### Question:\n' + row['question'].strip()
            + '\n\n### Python solution:\n' + (row.get('starter_code') or ''))


def record(source, source_id, prompt, domain, *, group=None, benchmark_text=None, **metadata):
    source_id = str(source_id)
    return dict(id=f"{source.replace('/', '__')}:train:{source_id}", source=source,
                source_id=source_id, source_split='train', split='train', panel_role='train',
                domain=domain, group_id=group or f'{source}|train|{source_id}', prompt=prompt,
                benchmark_text=prompt if benchmark_text is None else benchmark_text,
                prompt_format='raw_completion' if source == APPS else 'native_chat',
                origin='new_v2', **metadata)


def load_new_candidates(apps_path):
    pools, rejected, stats, sources = {}, Counter(), {}, []
    math_rows = []
    for i, row in enumerate(arrow_rows(MATH_PATH)):
        if row['level'] not in ('1', '2', '3', '4', '5'):
            rejected['math:unknown_level'] += 1
        elif DIAGRAM.search(row['problem']):
            rejected['math:diagram'] += 1
        else:
            math_rows.append(record(MATH, i, row['problem'], 'math',
                                    group=MATH + '|problem|' + text_hash(normalize(row['problem'])),
                                    math_type=row['type'], math_level=row['level'], template_id='math_problem_native_v2'))
    pools[MATH] = math_rows
    sources.append(dict(source=MATH, split='train', file=file_identity(MATH_PATH),
                        upstream='hendrydong/hendrycks_math',
                        revision='02682dae781ed3461d9753fd5398ea7894d67ecc',
                        note='Local mirror, Arrow checked equal to pinned snapshot parquet; solution is boxed answer only and is discarded.'))
    apps_rows, apps_stats = [], Counter()
    with Path(apps_path).open(encoding='utf-8') as stream:
        for i, line in enumerate(stream):
            if not line.strip():
                continue
            row = json.loads(line)
            sid = row.get('problem_id', row.get('id', i))
            starter = bool((row.get('starter_code') or '').strip())
            apps_stats['rows'] += 1
            apps_stats['with_starter' if starter else 'without_starter'] += 1
            apps_stats['difficulty:' + str(row.get('difficulty', 'unknown'))] += 1
            apps_rows.append(record(APPS, sid, apps_prompt(row), 'code', benchmark_text=row['question'],
                                    starter_code_present=starter, difficulty=row.get('difficulty'),
                                    template_id='apps_question_python_solution_completion_v2'))
    if len(apps_rows) != 5000 or len({r['id'] for r in apps_rows}) != 5000:
        raise ValueError('APPS must contain the complete official 5000 distinct training problems')
    pools[APPS], stats[APPS] = apps_rows, dict(apps_stats)
    sources.append(dict(source=APPS, split='train', file=file_identity(apps_path)))
    for source, subset in ((ARC_C, 'ARC-Challenge'), (ARC_E, 'ARC-Easy')):
        path = ARC_BASE / subset / ARC_REVISION / 'arc-train.arrow'
        pools[source] = [record(source, r['id'], arc_prompt(r), 'reasoning',
                                group='allenai/ai2_arc|' + str(r['id']), benchmark_text=r['question'],
                                option_labels=r['choices']['label'], option_count=len(r['choices']['label']),
                                template_id='arc_brief_reasoning_original_label_v1') for r in arrow_rows(path)]
        sources.append(dict(source=source, split='train', file=file_identity(path)))
    pools[HELLA] = [record(HELLA, r['ind'], mc_prompt(r['ctx'] + '\nWhich continuation is most plausible?',
                                                  ['A', 'B', 'C', 'D'], r['endings']),
                           'reasoning', group=HELLA + '|context|' + text_hash(normalize(r['ctx'])),
                           benchmark_text=r['ctx'], template_id='hellaswag_brief_reasoning_label_v2')
                    for r in arrow_rows(HELLA_PATH)]
    pools[RACE] = [record(RACE, r['example_id'] + ':' + str(i), mc_prompt(r['article'] + '\n\n' + r['question'],
                                                        ['A', 'B', 'C', 'D'], r['options']),
                          'reasoning', group=RACE + '|article|' + text_hash(normalize(r['article'])),
                          benchmark_text=r['article'] + '\n\n' + r['question'],
                          template_id='race_brief_reasoning_label_v2') for i, r in enumerate(arrow_rows(RACE_PATH))]
    for source, path in ((HELLA, HELLA_PATH), (RACE, RACE_PATH)):
        sources.append(dict(source=source, split='train', file=file_identity(path)))
    stats.update({source: dict(stats.get(source, {}), candidate_rows=len(rows)) for source, rows in pools.items()})
    return pools, rejected, stats, sources


def benchmark_records(old_prompts):
    rows, paths = [], []
    def add(source, sid, text, **extra):
        rows.append(dict(id=f'{source}:test:{sid}', source=source, prompt=text, **extra))
    for i, r in enumerate(arrow_rows(GSM_TEST)):
        add(GSM, i, r['question'])
    paths.append(GSM_TEST)
    for r in arrow_rows(HUMAN_PATH):
        add('openai/openai_humaneval', r['task_id'], r['prompt'], benchmark_docstring=human_docstring(r['prompt']))
    paths.append(HUMAN_PATH)
    for r in arrow_rows(MBPP_PATH):
        add('google-research-datasets/mbpp', r['task_id'], mbpp_prompt(r), benchmark_text=r['text'])
    paths.append(MBPP_PATH)
    for source, subset in ((ARC_C, 'ARC-Challenge'), (ARC_E, 'ARC-Easy')):
        path = ARC_BASE / subset / ARC_REVISION / 'arc-test.arrow'
        for r in arrow_rows(path):
            add(source, r['id'], arc_prompt(r), benchmark_text=r['question'], group_id='allenai/ai2_arc|' + str(r['id']))
        paths.append(path)
    for i, r in enumerate(arrow_rows(MATH_TEST)):
        add(MATH, i, r['problem'])
    paths.append(MATH_TEST)
    for path in (Path(old_prompts) / 'eval.jsonl', ROOT.parent / 'loop_drafter_pilot/prompts/pilot_eval.jsonl'):
        if path.exists():
            rows.extend(read_jsonl(path))
            paths.append(path)
    return rows, paths


def freeze_ids(row, tokenizer, model, maximum):
    result = dict(row)
    ids = encode_prompt(tokenizer, row['prompt'], row.get('prompt_format', 'native_chat'))
    if not 1 <= len(ids) <= maximum:
        return None
    result.update(prompt_format=row.get('prompt_format', 'native_chat'), input_ids=ids,
                  prompt_tokens=len(ids), tokenizer_path=str(Path(model).resolve()),
                  prompt_sha256=text_hash(row['prompt']), input_ids_sha256=ids_hash(ids), selection_seed=0)
    return result


def select_records(old_rows, pools, benchmark, tokenizer, model, *, new_counts=None, maximum=2048):
    new_counts = NEW_COUNTS if new_counts is None else new_counts
    selected, selected_groups, seen_ids = [], set(), set()
    selected_index, benchmark_index = OverlapIndex(), OverlapIndex()
    benchmark_groups = {r['group_id'] for r in benchmark if r.get('group_id')}
    for row in benchmark:
        add_record(benchmark_index, row)
    audit = dict(threshold=.90, normalization='NFKC, casefold, collapsed whitespace',
                 scope='Normalized exact and token 5-gram Jaccard only; not semantic decontamination.',
                 legacy_benchmark_matches=[], legacy_internal_matches=[], rejected_examples=[])
    for row in old_rows:
        if row['id'] in seen_ids or row['group_id'] in selected_groups:
            raise ValueError('actual legacy data repeats an ID/group; report before changing its provenance')
        frozen = freeze_ids(dict(row, origin='retained_v1', prompt_format='native_chat'), tokenizer, model, maximum)
        if frozen is None:
            raise ValueError('retained legacy prompt no longer fits input budget')
        for name, index in (('legacy_benchmark_matches', benchmark_index), ('legacy_internal_matches', selected_index)):
            matches = matching_record(index, frozen)
            if matches:
                audit[name].append(dict(id=row['id'], source=row['source'], matches=matches))
        selected.append(frozen)
        seen_ids.add(row['id'])
        selected_groups.add(row['group_id'])
        add_record(selected_index, frozen)
    rejected = Counter()
    selected_strata = Counter()
    for source, quota in new_counts.items():
        grouped = defaultdict(list)
        for row in pools[source]:
            if source == MATH:
                key = (row['math_type'], row['math_level'])
            elif source == APPS:
                key = (str(row['difficulty']), str(row['starter_code_present']))
            else:
                key = ('all',)
            grouped[key].append(row)
        queues = {key: deque(sorted(rs, key=lambda r: rank(0, 'data_v2:' + source + '|' + '|'.join(key), r['id'])))
                  for key, rs in grouped.items()}
        cycle = deque(sorted(queues))
        kept = 0
        while cycle and kept < quota:
            key = cycle.popleft()
            queue = queues[key]
            while queue:
                row = queue.popleft()
                reason, matches = None, []
                if row['id'] in seen_ids or row['group_id'] in selected_groups:
                    reason = 'selected_id_or_group'
                elif row['group_id'] in benchmark_groups:
                    reason = 'benchmark_group'
                else:
                    matches = matching_record(benchmark_index, row)
                    if matches:
                        reason = 'benchmark_exact_or_near'
                    else:
                        matches = matching_record(selected_index, row)
                        if matches:
                            reason = 'selected_exact_or_near'
                frozen = None if reason else freeze_ids(row, tokenizer, model, maximum)
                if reason is None and frozen is None:
                    reason = 'prompt_token_budget'
                if reason:
                    rejected[source + '|' + '|'.join(key) + ':' + reason] += 1
                    if len(audit['rejected_examples']) < 200:
                        audit['rejected_examples'].append(dict(id=row['id'], reason=reason, matches=matches[:5]))
                    continue
                selected.append(frozen)
                selected_groups.add(row['group_id'])
                seen_ids.add(row['id'])
                add_record(selected_index, frozen)
                kept += 1
                selected_strata[source + '|' + '|'.join(key)] += 1
                break
            if queue:
                cycle.append(key)
        if kept != quota:
            raise ValueError(f'Insufficient eligible pool for {source}: {kept}/{quota}; no source substitution allowed')
        print(json.dumps(dict(selected_source=source, rows=kept)), flush=True)
    # A single stable order is frozen for every model arm.
    selected.sort(key=lambda r: rank(0, 'data_v2:order', r['id']))
    if len(selected) != len(old_rows) + sum(new_counts.values()):
        raise ValueError('final source allocation is incomplete')
    audit.update(rejections=dict(rejected), selected_strata=dict(selected_strata))
    return selected, audit


def capture_plan(records, legacy_audit):
    if legacy_audit.get('status') != 'passed':
        raise ValueError('legacy capture audit must pass before reuse is planned')
    old = {r['id']: r for r in legacy_audit['records']}
    retained = {r['id'] for r in records if r['origin'] == 'retained_v1'}
    if retained != set(old):
        raise ValueError('legacy capture audit does not cover exactly the retained prompts')
    entries = []
    for row in records:
        item = dict(id=row['id'], source=row['source'], domain=row['domain'],
                    input_ids_sha256=row['input_ids_sha256'], prompt_tokens=row['prompt_tokens'])
        if row['id'] in old:
            prior = old[row['id']]
            if (not prior.get('actual_prompt_ids_match_frozen_prompt')
                    or prior.get('prompt_input_ids_sha256') != row['input_ids_sha256']):
                raise ValueError('retained prompt IDs differ from the actual legacy target inputs')
            action = prior['reuse_decision']
            if action not in ('reuse_complete', 'recapture_to_512'):
                raise ValueError('unexpected legacy trajectory reuse decision')
            if ((action == 'reuse_complete' and prior['terminal'] != 'native_eos')
                    or (action == 'recapture_to_512' and
                        (prior['terminal'] != 'cap256_non_eos' or prior['generated_length'] != 256))):
                raise ValueError('legacy reuse decision disagrees with the audited stop reason')
            item.update(action=action, prior_path=prior['path'],
                        prior_generated_length=prior['generated_length'], prior_terminal=prior['terminal'])
        else:
            item['action'] = 'capture_new_to_512'
        entries.append(item)
    return dict(schema='loop_drafter_capture_plan_v2', execution='plan_only_no_generation',
                max_prompt_tokens=2048, max_new_tokens=512, target_dtype='float32',
                feature_dtype='bfloat16', target_rounds=4, attention_backend='sdpa', decoding='greedy',
                policy='Reuse EOS-complete v1 records; fully regenerate every v1 non-EOS capped record from its original prompt to EOS/512; generate all new prompts under the same EOS/512 rule.',
                counts=dict(Counter(r['action'] for r in entries)), records=entries)


def select_reserves(training, pools, benchmark, tokenizer, model, *, limit=128, maximum=2048):
    """Select bounded same-source replacements without changing frozen training."""
    selected_index, benchmark_index = OverlapIndex(), OverlapIndex()
    used_ids = {r['id'] for r in training}
    groups = {r['group_id'] for r in training}
    benchmark_groups = {r['group_id'] for r in benchmark if r.get('group_id')}
    for row in training:
        add_record(selected_index, row)
    for row in benchmark:
        add_record(benchmark_index, row)
    records, rejected, counts = [], Counter(), {}
    for source, pool in pools.items():
        kept = 0
        for row in sorted(pool, key=lambda r: rank(0, 'data_v2:reserves:' + source, r['id'])):
            if kept == limit:
                break
            if row['id'] in used_ids or row['group_id'] in groups:
                reason = 'training_or_reserve_id_group'
            elif row['group_id'] in benchmark_groups or matching_record(benchmark_index, row):
                reason = 'benchmark_group_exact_or_near'
            elif matching_record(selected_index, row):
                reason = 'training_or_reserve_exact_or_near'
            else:
                reason = None
            frozen = None if reason else freeze_ids(row, tokenizer, model, maximum)
            if reason is None and frozen is None:
                reason = 'prompt_token_budget'
            if reason:
                rejected[source + ':' + reason] += 1
                continue
            frozen.update(origin='reserve_v2', split='train', panel_role='reserve')
            records.append(frozen)
            used_ids.add(row['id'])
            groups.add(row['group_id'])
            add_record(selected_index, frozen)
            kept += 1
        counts[source] = kept
        print(json.dumps(dict(reserve_source=source, rows=kept, requested=limit)), flush=True)
    records.sort(key=lambda r: rank(0, 'data_v2:reserve_order', r['id']))
    return records, dict(requested_per_source=limit, actual_counts=counts,
                         shortages={source: limit - n for source, n in counts.items() if n < limit},
                         candidate_counts={source: len(rs) for source, rs in pools.items()}, rejections=dict(rejected))


def load_frozen_runtime(args):
    reference = json.loads(args.runtime_reference.read_text())['tokenizer']
    if args.model.resolve() != Path(reference['tokenizer_path']).resolve():
        raise ValueError('tokenizer checkpoint differs from the frozen evaluation panel')
    for package in ('transformers', 'tokenizers'):
        if version(package) != reference[package + '_version']:
            raise ValueError(f'{package} differs from the frozen target tokenizer runtime')
    for identity in reference['files']:
        if file_identity(identity['path']) != identity:
            raise ValueError('target tokenizer files differ from the frozen evaluation panel')
    from transformers import AutoTokenizer
    return AutoTokenizer.from_pretrained(str(args.model), local_files_only=True, trust_remote_code=True)


def prepare_reserve_artifact(args):
    path, metadata_path = args.out / 'reserves.jsonl', args.out / 'reserves_sources.json'
    if path.exists() or metadata_path.exists():
        raise FileExistsError('existing frozen reserves must not be overwritten')
    training_path = args.out / 'train.jsonl'
    training_identity = file_identity(training_path)
    training = read_jsonl(training_path)
    if Counter(r['source'] for r in training) != Counter(OLD_COUNTS | NEW_COUNTS):
        raise ValueError('reserves require the complete frozen 16384-request training panel')
    tokenizer = load_frozen_runtime(args)
    new_pools, _, _, _ = load_new_candidates(args.apps)
    old_reserves = read_jsonl(args.old_prompts / 'reserves.jsonl')
    pools = {source: [r for r in old_reserves if r['source'] == source] for source in OLD_COUNTS}
    pools.update(new_pools)
    benchmarks, _ = benchmark_records(args.old_prompts)
    reserves, report = select_reserves(training, pools, benchmarks, tokenizer, args.model,
                                       limit=args.reserves_per_source)
    if file_identity(training_path) != training_identity:
        raise ValueError('frozen training manifest changed while selecting reserves')
    report.update(schema='loop_drafter_training_reserves_v2', created_at=datetime.now(timezone.utc).isoformat(),
                  training=training_identity, seed=0, max_prompt_tokens=2048, max_new_tokens=512,
                  legacy_policy=LEGACY_RESERVE_POLICY,
                  overlap_policy='All frozen training rows, complete benchmark exclusion records, and previously selected reserves; group/exact/5gram Jaccard>=.90.',
                  capture_policy=CAPTURE_RESERVE_POLICY,
                  tokenizer=dict(class_name=type(tokenizer).__name__, module=type(tokenizer).__module__,
                                 is_fast=tokenizer.is_fast, transformers_version=version('transformers'),
                                 tokenizers_version=version('tokenizers'), python_executable=sys.executable))
    path.write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in reserves))
    metadata_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps(dict(reserve_rows=len(reserves), counts=report['actual_counts'], shortages=report['shortages'])), flush=True)
    return report


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--apps', type=Path, required=True)
    parser.add_argument('--apps-source', type=Path)
    parser.add_argument('--old-prompts', type=Path, default=ROOT / 'prompts')
    parser.add_argument('--old-manifest', type=Path, default=ROOT / 'capture_fp32/manifest.jsonl')
    parser.add_argument('--legacy-audit', type=Path, default=ROOT / 'data_v2/legacy_capture_audit.json')
    parser.add_argument('--model', type=Path, default=MODEL)
    parser.add_argument('--runtime-reference', type=Path, default=ROOT / 'evaluation_v1/sources.json')
    parser.add_argument('--out', type=Path, default=ROOT / 'data_v2/prompts')
    parser.add_argument('--reserves-only', action='store_true')
    parser.add_argument('--reserves-per-source', type=int, default=128)
    args = parser.parse_args(argv)
    if not 1 <= args.reserves_per_source <= 128:
        raise ValueError('reserve count must be between one and 128 per source')
    if args.reserves_only:
        return prepare_reserve_artifact(args)
    if args.out.exists() and any(args.out.iterdir()):
        raise FileExistsError('data v2 freeze requires an empty destination; existing freezes are immutable')
    legacy_audit = json.loads(args.legacy_audit.read_text())
    manifest = read_jsonl(args.old_manifest)
    lookup = {r['id']: r for name in ('train.jsonl', 'reserves.jsonl')
              for r in read_jsonl(args.old_prompts / name)}
    if len(manifest) != 8192 or Counter(r['source'] for r in manifest) != Counter(OLD_COUNTS):
        raise ValueError('expected the actual 8192-request v1 capture manifest')
    old = []
    for item in manifest:
        row = lookup[item['id']]
        if row['source'] != item['source'] or row['split'] != 'train':
            raise ValueError('legacy actual capture differs from its prepared prompt provenance')
        old.append(row)
    tokenizer = load_frozen_runtime(args)
    pools, preliminary_rejections, pool_stats, sources = load_new_candidates(args.apps)
    benchmark, benchmark_paths = benchmark_records(args.old_prompts)
    records, overlaps = select_records(old, pools, benchmark, tokenizer, args.model)
    plan = capture_plan(records, legacy_audit)
    counts = Counter(r['source'] for r in records)
    if counts != Counter(OLD_COUNTS | NEW_COUNTS):
        raise ValueError('final source counts differ from the frozen recipe')
    summaries = {}
    for source in counts:
        rs = [r for r in records if r['source'] == source]
        lengths = sorted(r['prompt_tokens'] for r in rs)
        summaries[source] = dict(rows=len(rs), min_prompt_tokens=lengths[0], max_prompt_tokens=lengths[-1],
                                 mean_prompt_tokens=sum(lengths) / len(lengths), above512=sum(n > 512 for n in lengths))
        if source == APPS:
            summaries[source].update(with_starter=sum(r['starter_code_present'] for r in rs),
                                     difficulty=dict(Counter(r['difficulty'] for r in rs)))
    metadata = dict(schema='loop_drafter_training_prompts_v2', created_at=datetime.now(timezone.utc).isoformat(),
                    seed=0, rows=len(records), source_counts=dict(counts),
                    domain_counts=dict(Counter(r['domain'] for r in records)), summaries=summaries,
                    max_prompt_tokens=2048, max_new_tokens=512, sources=sources,
                    apps_source=json.loads(args.apps_source.read_text()) if args.apps_source else None,
                    legacy_manifest=file_identity(args.old_manifest), legacy_audit=file_identity(args.legacy_audit),
                    benchmark_files=[file_identity(p) for p in benchmark_paths],
                    preliminary_rejections=dict(preliminary_rejections), candidate_pools=pool_stats,
                    stratum_policy='Deterministic round-robin over MATH (type,level) and APPS (difficulty,starter presence); exhausted strata yield to remaining strata; only source quotas are fixed.',
                    tokenizer=dict(path=str(args.model.resolve()), class_name=type(tokenizer).__name__,
                                   module=type(tokenizer).__module__, is_fast=tokenizer.is_fast,
                                   python_executable=sys.executable, transformers_version=version('transformers'),
                                   tokenizers_version=version('tokenizers'), runtime_reference=str(args.runtime_reference.resolve()),
                                   files=[file_identity(args.model / name) for name in
                                          ('tokenizer.json', 'tokenizer_config.json', 'special_tokens_map.json', 'vocab.json', 'merges.txt')
                                          if (args.model / name).exists()]),
                    templates=dict(arc=MC_INSTRUCTION, apps='### Question:\n{question}\n\n### Python solution:\n{starter_code_verbatim}',
                                   general='Native chat; preserve original ARC option labels; request a brief explanation then the final option label; no gold labels.'),
                    selection_policy='Retain actual v1 prompts, then deterministic source/type/level-stratified selection with exact/group/5gram>=.90 exclusion against prior selected rows and complete test sets. No semantic-contamination claim.',
                    execution=dict(gpu_used=False, model_generation=False, network_downloads=False))
    args.out.mkdir(parents=True, exist_ok=True)
    for name, content in (('sources.json', metadata), ('overlap_report.json', overlaps), ('capture_plan.json', plan)):
        (args.out / name).write_text(json.dumps(content, ensure_ascii=False, indent=2) + '\n')
    (args.out / 'train.jsonl').write_text(''.join(json.dumps(r, ensure_ascii=False) + '\n' for r in records))
    print(json.dumps(dict(out=str(args.out), rows=len(records), domain_counts=metadata['domain_counts'],
                          capture_actions=plan['counts'], legacy_benchmark_matches=len(overlaps['legacy_benchmark_matches']))), flush=True)
    return metadata


if __name__ == '__main__':
    main()
