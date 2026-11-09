"""Materialize the frozen v2 dataset with independent GPU workers and atomic publication."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from contextlib import ExitStack
import fcntl
import gc
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback

import torch
from .capture_protocol import capture_batch, _atomic_json, _atomic_torch, _append, _journal, _lock
from .capture_parallel import _write_jsonl
from .auto_train_when_ready import gpu_inventory, eligible_gpus, write_status
from .evaluate_loop_memory import encode_prompt
from .loop_memory_draft import LoopMemoryConfig
from .train import validate_record


def read_rows(path):
    return [json.loads(line) for line in Path(path).open() if line.strip()]


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def batch_limit(length):
    if not 1 <= length <= 2048:
        raise ValueError('frozen prompt exceeds registered 2048 token limit')
    return 4 if length + 512 <= 1024 else 2 if length + 512 <= 1792 else 1


def frozen_inputs(root):
    folder = root / 'data_v2/prompts'
    return {str(p): digest(p) for p in [folder / name for name in
            ('train.jsonl', 'reserves.jsonl', 'capture_plan.json', 'sources.json',
             'overlap_report.json', 'reserves_sources.json')]}


def prepare(root, protocol, gpus):
    recipe = json.loads(protocol.read_text())
    if (recipe['capture']['max_new_tokens'] != 512 or recipe['capture']['max_prompt_tokens'] != 2048
            or recipe['capture']['target_dtype'] != 'float32' or recipe['capture']['target_rounds'] != 4):
        raise ValueError('unsupported capture protocol')
    out = root / 'data_v2/capture'
    out.mkdir(parents=True, exist_ok=True)
    from importlib.metadata import version
    metadata = json.loads((root / 'data_v2/prompts/sources.json').read_text())['tokenizer']
    if Path(metadata['path']).resolve() != Path(recipe['capture']['target']).resolve():
        raise ValueError('model differs from frozen tokenizer')
    for package in ('transformers', 'tokenizers'):
        if version(package) != metadata[package + '_version']:
            raise ValueError(f'frozen tokenizer runtime differs: {package}')
    for identity in metadata['files']:
        if digest(identity['path']) != identity['sha256']:
            raise ValueError('frozen tokenizer file changed')
    identity = frozen_inputs(root) | {str(protocol): digest(protocol)}
    plan_path = out / 'plan.json'
    if plan_path.exists():
        plan = json.loads(plan_path.read_text())
        if plan['inputs'] != identity or plan['gpus'] != gpus:
            raise ValueError('capture plan changed; resume with original protocol and GPU assignment')
        return plan
    rows = read_rows(root / 'data_v2/prompts/train.jsonl')
    reserves = read_rows(root / 'data_v2/prompts/reserves.jsonl')
    actions = json.loads((root / 'data_v2/prompts/capture_plan.json').read_text())['records']
    if len(rows) != recipe['training_data']['effective_requests'] or len(actions) != len(rows):
        raise ValueError('incomplete frozen prompt/action set')
    if len({r['id'] for r in rows + reserves}) != len(rows) + len(reserves):
        raise ValueError('duplicate prompt/reserve IDs')
    workers = [[] for _ in gpus]
    loads = [0] * len(gpus)
    reuse, pending = [], []
    for slot, (row, action) in enumerate(zip(rows, actions)):
        if (row['id'] != action['id'] or row['input_ids_sha256'] != action['input_ids_sha256']
                or row['split'] != 'train' or row['prompt_tokens'] != len(row['input_ids'])):
            raise ValueError('frozen capture plan disagrees with prompt inputs')
        item = dict(slot=slot, record=row, action=action)
        (reuse if action['action'] == 'reuse_complete' else pending).append(item)
    # Assign complete equal-length batches, balancing estimated token work.
    buckets = defaultdict(list)
    for item in pending:
        buckets[len(item['record']['input_ids'])].append(item)
    for length in sorted(buckets, reverse=True):
        size = batch_limit(length)
        for start in range(0, len(buckets[length]), size):
            group = buckets[length][start:start + size]
            worker = min(range(len(gpus)), key=lambda i: loads[i])
            workers[worker].extend(group)
            loads[worker] += len(group) * (length + 512)
    reserve_groups = [[] for _ in gpus]
    source_reserves = defaultdict(list)
    for row in reserves:
        source_reserves[row['source']].append(row)
    for source, candidates in source_reserves.items():
        active = [i for i, items in enumerate(workers) if any(
            x['record']['source'] == source and x['action']['action'] == 'capture_new_to_512' for x in items)]
        for i, row in enumerate(candidates):
            if active:
                reserve_groups[active[i % len(active)]].append(row)
    spec = []
    for i, (gpu, items) in enumerate(zip(gpus, workers)):
        folder = out / f'worker_{i}'
        folder.mkdir(exist_ok=True)
        _write_jsonl(folder / 'tasks.jsonl', items)
        _write_jsonl(folder / 'reserves.jsonl', reserve_groups[i])
        spec.append(dict(index=i, gpu=gpu, tasks=len(items),
                         task_sha256=digest(folder / 'tasks.jsonl'),
                         reserve_sha256=digest(folder / 'reserves.jsonl')))
    _write_jsonl(out / 'reuse.jsonl', reuse)
    model = Path(recipe['capture']['target'])
    model_files = {str(p): digest(p) for p in model.iterdir()
                   if p.suffix in ('.json', '.py', '.txt')}
    plan = dict(schema='loop_drafter_capture_v2', inputs=identity, model=str(model),
                model_files=model_files, root=str(root), protocol=str(protocol), gpus=gpus,
                workers=spec, reuse=len(reuse), pending=len(pending), expected=len(rows),
                reuse_sha256=digest(out / 'reuse.jsonl'), source_counts=dict(Counter(r['source'] for r in rows)))
    _atomic_json(plan_path, plan)
    return plan


def check_plan(plan):
    for path, value in (plan['inputs'] | plan['model_files']).items():
        if digest(path) != value:
            raise ValueError(f'frozen input changed: {path}')


def validate_trajectory(value, prompt, eos, reused=False):
    validate_record(value, LoopMemoryConfig())
    tokens = value['token_ids']
    length = len(prompt['input_ids'])
    if (value['id'] != prompt['id'] or value['source'] != prompt['source']
            or value['split'] != 'train' or value['prompt_length'] != length
            or tokens[:length].tolist() != prompt['input_ids']
            or value['round_hidden'].dtype != torch.bfloat16):
        raise ValueError('trajectory identity or frozen inputs mismatch')
    generated = tokens[length:].tolist()
    if not 2 <= len(generated) <= 512 or any(t in eos for t in generated[:-1]):
        raise ValueError('invalid generated trajectory length or EOS position')
    if generated[-1] not in eos and (reused or len(generated) != 512):
        raise ValueError('trajectory neither EOS complete nor at output cap')


def capture_retry(target, ids, eos):
    """OOM retries retain precisely the same prompts and generation budget."""
    try:
        return capture_batch(target, ids, 512, eos)
    except torch.cuda.OutOfMemoryError:
        if len(ids) == 1:
            raise
    gc.collect()
    torch.cuda.empty_cache()
    middle = len(ids) // 2
    return capture_retry(target, ids[:middle], eos) + capture_retry(target, ids[middle:], eos)


def worker(root, index, backend='reference'):
    global capture_batch
    if backend == 'graph':
        from .capture_fast import capture_batch_graph
        capture_batch = capture_batch_graph
    elif backend == 'buffered':
        from .capture_fast import capture_batch_fast
        capture_batch = capture_batch_fast
    elif backend != 'reference':
        raise ValueError('unknown capture backend')
    out = root / 'data_v2/capture'
    plan = json.loads((out / 'plan.json').read_text())
    check_plan(plan)
    folder = out / f'worker_{index}'
    spec = plan['workers'][index]
    if digest(folder / 'tasks.jsonl') != spec['task_sha256'] or digest(folder / 'reserves.jsonl') != spec['reserve_sha256']:
        raise ValueError('worker shard changed')
    with _lock(folder):
        (folder / 'trajectories').mkdir(exist_ok=True)
        torch.set_num_threads(4)
        torch.cuda.set_per_process_memory_fraction(.80)
        from .target import OuroTarget
        target = OuroTarget.from_pretrained(plan['model'], device='cuda:0', dtype=torch.float32, attention_backend='sdpa')
        eos_value = target.tokenizer.eos_token_id
        eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
        tasks = read_rows(folder / 'tasks.jsonl')
        reserves = read_rows(folder / 'reserves.jsonl')
        candidates = {r['id']: r for r in reserves + [t['record'] for t in tasks]}
        # Validate even completed frozen prompts on resume, without retokenizing.
        for row in candidates.values():
            encode_prompt(target.tokenizer, row, 2048)
        done = _journal(folder / 'manifest.jsonl')
        completed = {r['slot'] for r in done}
        if len(completed) != len(done):
            raise ValueError('duplicate completed slots')
        rejected = _journal(folder / 'rejected.jsonl')
        used = {r['id'] for r in done + rejected}
        started = time.monotonic()
        initial_done = len(done)
        generation_seconds = write_seconds = 0.
        def progress(state='capturing'):
            write_status(folder / 'status.json', state=state, done=len(done), total=len(tasks),
                         gpu=spec['gpu'], backend=backend, initial_done=initial_done,
                         session_completed=len(done)-initial_done, generation_seconds=generation_seconds,
                         write_seconds=write_seconds, elapsed_seconds=time.monotonic()-started)
        def save(item, prompt, tokens, hidden, captured_backend=None):
            nonlocal write_seconds
            write_started = time.monotonic()
            original = item['record']
            path = folder / 'trajectories' / f"{item['slot']:06d}.pt"
            row = dict(id=prompt['id'], source=prompt['source'], domain=prompt['domain'], split='train',
                       slot=item['slot'], original_id=original['id'], path=str(path),
                       prompt_length=len(prompt['input_ids']), generated_length=len(tokens)-len(prompt['input_ids']),
                       terminal='native_eos' if int(tokens[-1]) in eos else 'cap512_non_eos',
                       input_ids_sha256=prompt['input_ids_sha256'], capture_backend=captured_backend or backend)
            payload = row | dict(token_ids=tokens, round_hidden=hidden)
            validate_trajectory(payload, prompt, eos)
            _atomic_torch(path, payload)
            _append(folder / 'manifest.jsonl', row)
            done.append(row); completed.add(item['slot']); used.add(prompt['id'])
            write_seconds += time.monotonic() - write_started
        # Recovery of a completed tensor whose journal append was interrupted.
        for item in tasks:
            path = folder / 'trajectories' / f"{item['slot']:06d}.pt"
            if item['slot'] not in completed and path.exists():
                value = torch.load(path, map_location='cpu', weights_only=True)
                if value['original_id'] != item['record']['id'] or value['id'] not in candidates:
                    raise ValueError('orphan trajectory identity changed')
                prompt = candidates[value['id']]
                save(item, prompt, value['token_ids'], value['round_hidden'], value.get('capture_backend', 'reference'))
        pending = [t for t in tasks if t['slot'] not in completed]
        # Validate each length tier at its largest prompt before full production.
        probes = {}
        for item in pending:
            length = len(item['record']['input_ids'])
            tier = batch_limit(length)
            if tier not in probes or length > len(probes[tier]['record']['input_ids']):
                probes[tier] = item
        measurements = []
        for size, item in sorted(probes.items()):
            ids = encode_prompt(target.tokenizer, item['record'], 2048).repeat(size, 1).cuda()
            torch.cuda.reset_peak_memory_stats()
            captured = capture_retry(target, ids, eos)
            measurements.append(dict(batch=size, prompt_tokens=ids.shape[1],
                                     peak_allocated=torch.cuda.max_memory_allocated(),
                                     peak_reserved=torch.cuda.max_memory_reserved()))
            del captured, ids
            gc.collect(); torch.cuda.empty_cache()
            _atomic_json(folder / 'memory_preflight.json', measurements)
            progress('preflight')
        buckets = defaultdict(list)
        for item in pending:
            buckets[len(item['record']['input_ids'])].append(item)
        for length in sorted(buckets):
            size = batch_limit(length)
            for offset in range(0, len(buckets[length]), size):
                group = buckets[length][offset:offset+size]
                ids = torch.cat([encode_prompt(target.tokenizer, x['record'], 2048) for x in group]).cuda()
                batch_started = time.monotonic()
                outputs = capture_retry(target, ids, eos)
                generation_seconds += time.monotonic() - batch_started
                for item, (tokens, hidden) in zip(group, outputs):
                    prompt = item['record']
                    while len(tokens) - len(prompt['input_ids']) < 2:
                        if item['action']['action'] != 'capture_new_to_512':
                            raise ValueError('legacy trajectory too short; replacement forbidden')
                        if prompt['id'] not in used:
                            _append(folder / 'rejected.jsonl', dict(id=prompt['id'], source=prompt['source'],
                                    slot=item['slot'], reason='fewer_than_two_generated_tokens'))
                        used.add(prompt['id'])
                        prompt = next((r for r in reserves if r['source'] == prompt['source'] and r['id'] not in used), None)
                        if prompt is None:
                            raise ValueError('same-source reserve shard exhausted')
                        tokens, hidden = capture_retry(target, encode_prompt(target.tokenizer, prompt, 2048).cuda(), eos)[0]
                    save(item, prompt, tokens, hidden)
                progress()
        progress('complete')


def finalize(root, plan):
    check_plan(plan)
    out = root / 'data_v2/capture'
    if digest(out / 'reuse.jsonl') != plan['reuse_sha256']:
        raise ValueError('reuse list changed')
    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(plan['model'], trust_remote_code=True, local_files_only=True)
    eos_value = tokenizer.eos_token_id
    eos = set(eos_value if isinstance(eos_value, list) else [eos_value])
    rows = []
    prompts = read_rows(root / 'data_v2/prompts/train.jsonl')
    candidates = {r['id']: r for r in prompts + read_rows(root / 'data_v2/prompts/reserves.jsonl')}
    for item in read_rows(out / 'reuse.jsonl'):
        path = Path(item['action']['prior_path'])
        old_config = json.loads((path.parent.parent / 'capture_config.json').read_text())
        if any(old_config.get(k) != v for k,v in dict(model=plan['model'], target_dtype='float32', target_rounds=4, attention_backend='sdpa').items()):
            raise ValueError('legacy target capture configuration differs')
        value = torch.load(path, map_location='cpu', weights_only=True)
        validate_trajectory(value, item['record'], eos, reused=True)
        row = {k:value[k] for k in ('id','source','split','prompt_length','generated_length')}
        rows.append(row | dict(path=str(path), slot=item['slot'], original_id=value['id'],
                               domain=item['record']['domain'], terminal='native_eos', reused=True))
    for spec in plan['workers']:
        folder = out / f"worker_{spec['index']}"
        status = json.loads((folder / 'status.json').read_text())
        if status['state'] != 'complete' or status['done'] != spec['tasks']:
            raise ValueError('worker incomplete')
        rows.extend(read_rows(folder / 'manifest.jsonl'))
    rows.sort(key=lambda r:r['slot'])
    if [r['slot'] for r in rows] != list(range(plan['expected'])) or len({r['id'] for r in rows}) != plan['expected']:
        raise ValueError('missing or duplicate effective slots/IDs')
    if dict(Counter(r['source'] for r in rows)) != plan['source_counts']:
        raise ValueError('effective source quotas differ')
    effective, entries, totals = [], [], defaultdict(Counter)
    for row in rows:
        prompt = candidates[row['id']]
        original = prompts[row['slot']]
        if row['original_id'] != original['id'] or prompt['source'] != original['source']:
            raise ValueError('effective request changed source/slot')
        if prompt['id'] != original['id'] and original.get('origin') == 'retained_v1':
            raise ValueError('legacy replacement forbidden')
        path = Path(row['path'])
        value = torch.load(path, map_location='cpu', weights_only=True)
        validate_trajectory(value, prompt, eos, reused=row.get('reused',False))
        if value['generated_length'] != row['generated_length']:
            raise ValueError('manifest length differs from tensor')
        stat = path.stat(); n = len(value['token_ids']); p = value['prompt_length']
        entries.append(dict(path=str(path), id=row['id'], source=row['source'], prompt_length=p,
                            token_count=n, available_anchors=n-p-1, file_size=stat.st_size, mtime_ns=stat.st_mtime_ns))
        effective.append(prompt)
        totals[prompt['domain']].update(requests=1, generated_tokens=n-p, available_anchors=n-p-1)
    _write_jsonl(out / 'effective_prompts.jsonl', effective)
    _atomic_json(out / 'effective_overlap_audit.json', dict(
        effective_prompts_sha256=digest(out / 'effective_prompts.jsonl'),
        provenance=plan['inputs'], replacements=[dict(original=r['original_id'], effective=r['id']) for r in rows if r['original_id'] != r['id']],
        basis='Effective prompts are a subset of the frozen jointly deduplicated primary/reserve pools. Original exclusion scope is retained; no semantic decontamination claim.',
        source_counts=dict(Counter(r['source'] for r in rows)), domains=totals))
    _atomic_json(out / 'config.json', dict(config=LoopMemoryConfig().dict(), capture_plan=str(out/'plan.json')))
    _atomic_json(out / 'data_index.json', entries)
    # The canonical manifest is the publication barrier observed by the training supervisor.
    _write_jsonl(out / 'manifest.jsonl', rows)
    _atomic_json(out / 'summary.json', dict(complete=True, records=len(rows), reused=plan['reuse'], domains=totals))


def supervise(root, protocol, gpus, backend='reference'):
    out = root / 'data_v2/capture'
    out.mkdir(parents=True, exist_ok=True)
    children = []
    with _lock(out), ExitStack() as stack:
        try:
            plan = prepare(root, protocol, gpus)
            if (out / 'summary.json').exists():
                print('Capture already complete', flush=True); return
            leases = root / 'gpu_leases'; leases.mkdir(exist_ok=True)
            for gpu in gpus:
                handle = stack.enter_context((leases/f'gpu{gpu}.lock').open('a'))
                fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
            if not set(gpus).issubset(eligible_gpus(gpu_inventory())):
                raise RuntimeError('selected GPU now occupied; no capture launched')
            for spec in plan['workers']:
                folder = out / f"worker_{spec['index']}"
                log = stack.enter_context((folder/'worker.log').open('a'))
                env = dict(os.environ, CUDA_VISIBLE_DEVICES=spec['gpu'], OMP_NUM_THREADS='4', MKL_NUM_THREADS='4')
                child = subprocess.Popen([sys.executable, '-u', '-m', 'scripts.loop_drafter.capture_v2',
                        '--phase','worker','--root',str(root),'--worker',str(spec['index']),'--backend',backend], env=env,
                        cwd=Path(__file__).resolve().parents[2], stdout=log, stderr=subprocess.STDOUT)
                children.append(child)
            _append(out/'execution_sessions.jsonl', dict(started_unix=time.time(), backend=backend,
                    gpus=gpus, worker_pids=[c.pid for c in children]))
            while any(c.poll() is None for c in children):
                failed = [c for c in children if c.poll() not in (None,0)]
                if failed:
                    raise RuntimeError(f'capture worker failed: {[c.pid for c in failed]}; see worker.log')
                write_status(out/'status.json', state='capturing', gpus=gpus, worker_pids=[c.pid for c in children],
                             pending=plan['pending'], reused=plan['reuse'], backend=backend)
                time.sleep(10)
            if any(c.returncode for c in children):
                raise RuntimeError('capture worker failed')
            write_status(out/'status.json', state='validating_and_publishing', gpus=gpus)
            finalize(root, plan)
            write_status(out/'status.json', state='complete', records=plan['expected'], gpus=gpus)
        except BaseException as error:
            for child in children:
                if child.poll() is None:
                    child.terminate()
            for child in children:
                try: child.wait(timeout=30)
                except subprocess.TimeoutExpired: child.kill(); child.wait()
            write_status(out/'status.json', state='failed', error=repr(error))
            raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', required=True, type=Path)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('data_protocol_v2.json'))
    parser.add_argument('--phase', choices=['run','worker'], default='run')
    parser.add_argument('--gpus', default='0,2,3,4,5,6,7')
    parser.add_argument('--worker', type=int)
    parser.add_argument('--backend', choices=['reference','buffered','graph'], default='reference')
    args = parser.parse_args()
    if args.phase == 'worker': worker(args.root.resolve(), args.worker, args.backend)
    else:
        gpus = args.gpus.split(',')
        if not gpus or len(set(gpus)) != len(gpus) or any(not g.isdigit() for g in gpus):
            raise ValueError('GPU indices must be unique integers')
        supervise(args.root.resolve(), args.protocol.resolve(), gpus, args.backend)


if __name__ == '__main__':
    main()
