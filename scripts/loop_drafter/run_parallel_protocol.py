"""Resume the fixed training protocol with stage-specific GPU parallelism."""

import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import json
import os
from pathlib import Path
import random
import signal
import subprocess
import sys
import time

from .capture import read_jsonl
from .evaluate_protocol import aggregate, atomic_json


def now():
    return datetime.now(timezone.utc).isoformat()


def evaluation_panels(prompts, workers, per_source, seed=0):
    counts = Counter(row['source'] for row in prompts)
    if not counts or any(count != per_source for count in counts.values()):
        raise ValueError(f'unexpected evaluation panel counts: {dict(counts)}')
    if per_source % workers:
        raise ValueError('evaluation prompts per source must divide evenly across GPUs')
    if len({row['id'] for row in prompts}) != len(prompts):
        raise ValueError('duplicate evaluation prompt IDs')
    panels = [[] for _ in range(workers)]
    rng = random.Random(seed)
    for source in sorted(counts):
        rows = [row for row in prompts if row['source'] == source]
        rng.shuffle(rows)
        for index, row in enumerate(rows):
            panels[index % workers].append(row)
    return panels


def merge_evaluation(root, prompts, workers):
    """Pair methods within each GPU before combining request-time denominators."""
    root = Path(root)
    records, hardware, by_gpu = [], [], []
    methods = {'ar', 'final', 'multiloop', 'trajectory'}
    expected = {(row['id'], method): row['source'] for row in prompts for method in methods}
    seen = set()
    for index, gpu in enumerate(workers):
        shard = root / 'shards' / f'worker{index}'
        summary = json.loads((shard / 'summary.json').read_text())
        rows = read_jsonl(shard / 'requests.jsonl')
        panel = read_jsonl(root / 'panels' / f'worker{index}.jsonl')
        shard_expected = {(row['id'], method) for row in panel for method in methods}
        if {(row['id'], row['method']) for row in rows} != shard_expected:
            raise ValueError(f'incomplete evaluation shard {index}')
        if summary['hardware']['cuda_visible_devices'] != gpu:
            raise ValueError(f'evaluation shard {index} moved to another GPU')
        for row in rows:
            key = row['id'], row['method']
            if key in seen or expected.get(key) != row['source']:
                raise ValueError(f'duplicate or invalid evaluation row: {key}')
            seen.add(key)
            records.append(row | {'worker_index': index, 'physical_gpu': gpu})
        hardware.append(summary['hardware'])
        by_gpu.append({'physical_gpu': gpu, 'results': aggregate(rows)})
    if seen != set(expected):
        raise ValueError('combined evaluation does not cover the full panel')
    temporary = root / 'requests.jsonl.tmp'
    temporary.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in records))
    temporary.replace(root / 'requests.jsonl')
    result = {'source_counts': dict(Counter(row['source'] for row in prompts)),
              'results': aggregate(records), 'results_by_gpu': by_gpu, 'hardware': hardware,
              'metric_notes': {'speedup': 'paired AR summed latency divided by drafter summed latency; all methods for one prompt run on the same GPU',
                               'throughput': 'sum tokens / sum request times, not aggregate multi-GPU serving throughput',
                               'output_comparison': 'token identity diagnostic only; no answer correctness scoring',
                               'timing_scope': 'shared_gpu_interleaved_diagnostic'}}
    atomic_json(root / 'summary.json', result)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--model', required=True)
    parser.add_argument('--root', required=True)
    parser.add_argument('--gpus', default='4,5,6,7')
    parser.add_argument('--run-name', default='run_seed0')
    parser.add_argument('--expected-train-requests', type=int, default=8192)
    parser.add_argument('--eval-per-source', type=int, default=300)
    parser.add_argument('--baseline-epochs', type=int, default=4)
    parser.add_argument('--branch-epochs', type=int, default=2)
    parser.add_argument('--requests-per-update', type=int, default=8)
    parser.add_argument('--max-anchors-per-request', type=int, default=64)
    parser.add_argument('--microbatch', type=int, default=16)
    parser.add_argument('--capture-batch-size', type=int, default=4)
    parser.add_argument('--max-new-tokens', type=int, default=256)
    parser.add_argument('--memory-fraction', type=float, default=.18)
    args = parser.parse_args()
    gpus = args.gpus.split(',')
    if len(gpus) != 4 or len(set(gpus)) != len(gpus) or any(not gpu.isdigit() for gpu in gpus):
        raise ValueError('provide four distinct physical GPU indices')
    if args.requests_per_update < len(gpus) or args.eval_per_source % len(gpus):
        raise ValueError('global request batch must cover all GPUs; evaluation panel must divide evenly')
    root = Path(args.root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    with (root / 'protocol.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        run = root / args.run_name
        run.mkdir(parents=True, exist_ok=True)
        repo = Path(__file__).resolve().parents[2]
        # The scientific protocol stays identical to run_protocol. GPU placement
        # is stored separately, so migration does not create another experiment.
        config = {key: value for key, value in vars(args).items() if key != 'gpus'}
        config.update(seed=0, target_dtype='float32', source_layout='canonical',
                      loss='uniform_ce', checkpoint_selection='fixed_final_epoch',
                      development_panel=None, additional_seeds=[])
        protocol_path = run / 'protocol.json'
        if protocol_path.exists() and json.loads(protocol_path.read_text()) != config:
            raise ValueError('training protocol differs from the existing run')
        atomic_json(protocol_path, config)
        execution = {'gpus': gpus, 'baseline_world_size': len(gpus),
                     'branch_gpus': dict(zip(('final', 'multiloop', 'trajectory'), gpus)),
                     'evaluation': 'source-balanced prompt shards; all methods on the same GPU'}
        execution_path = run / 'execution_parallel.json'
        if execution_path.exists() and json.loads(execution_path.read_text()) != execution:
            raise ValueError('parallel GPU assignment changed on resume')
        atomic_json(execution_path, execution)
        status_path = run / 'status.json'
        previous = json.loads(status_path.read_text()) if status_path.exists() else {}
        completed = list(previous.get('completed_stages', []))
        jobs = dict(previous.get('jobs', {}))
        status = {'state': 'running', 'started_at': previous.get('started_at', now()),
                  'resumed_at': now(), 'pid': os.getpid(), 'gpus': gpus,
                  'config': config, 'execution': execution, 'jobs': jobs,
                  'completed_stages': completed}

        def save():
            status['updated_at'] = now()
            atomic_json(status_path, status)

        def interrupt(signum, frame):
            raise KeyboardInterrupt(f'parallel launcher received signal {signum}')

        signal.signal(signal.SIGTERM, interrupt)

        def job(name, module, options, gpu_ids, artifact, distributed=False):
            command = [sys.executable, '-m']
            if distributed:
                command += ['torch.distributed.run', '--standalone', f'--nproc_per_node={len(gpu_ids)}', '-m']
            command += [f'scripts.loop_drafter.{module}', *map(str, options)]
            return dict(name=name, command=command, gpus=','.join(gpu_ids), artifact=str(artifact))

        def pool(stage, specs, progress=None):
            status['stage'] = stage
            save()
            active = []
            with ExitStack() as stack:
                try:
                    for spec in specs:
                        name = spec['name']
                        if jobs.get(name, {}).get('state') == 'complete':
                            if not Path(spec['artifact']).is_file():
                                raise FileNotFoundError(spec['artifact'])
                            continue
                        log_path = run / f'{name}.log'
                        log = stack.enter_context(log_path.open('a'))
                        log.write('\n' + json.dumps({'started_at': now(), **spec}) + '\n')
                        log.flush()
                        env = dict(os.environ, CUDA_VISIBLE_DEVICES=spec['gpus'], OMP_NUM_THREADS='8', PYTHONUNBUFFERED='1')
                        child = subprocess.Popen(spec['command'], cwd=repo, env=env,
                                                 stdout=log, stderr=subprocess.STDOUT,
                                                 start_new_session=True, pass_fds=(lock.fileno(),))
                        jobs[name] = dict(spec, state='running', pid=child.pid, log=str(log_path), started_at=now())
                        active.append((spec, child))
                        save()
                    last_progress = 0.
                    while active:
                        for spec, child in list(active):
                            result = child.poll()
                            if result is None:
                                continue
                            if result or not Path(spec['artifact']).is_file():
                                jobs[spec['name']].update(state='failed', returncode=result or 1)
                                raise RuntimeError(f"{spec['name']} failed; see {jobs[spec['name']]['log']}")
                            jobs[spec['name']].update(state='complete', finished_at=now())
                            active.remove((spec, child))
                            save()
                        if progress and time.monotonic() - last_progress > 10:
                            progress()
                            last_progress = time.monotonic()
                        if active:
                            time.sleep(1)
                except BaseException as error:
                    # Terminate only children launched by this invocation.
                    for spec, child in active:
                        if child.poll() is None:
                            os.killpg(child.pid, signal.SIGTERM)
                            try:
                                child.wait(timeout=15)
                            except subprocess.TimeoutExpired:
                                os.killpg(child.pid, signal.SIGKILL)
                                child.wait()
                            jobs[spec['name']].update(state='interrupted', returncode=child.returncode)
                    status.update(state='failed', error=repr(error), failed_at=now())
                    save()
                    raise
            if progress:
                progress()

        def finish(stage):
            if stage not in completed:
                completed.append(stage)
            save()

        capture = root / 'capture_fp32'
        artifacts = {'capture': capture / 'capture_summary.json',
                     'baseline': run / 'baseline/model.pt',
                     'speed_evaluation': run / 'evaluation/summary.json'}
        artifacts.update({f'branch_{mode}': run / f'branch_{mode}/model.pt'
                          for mode in ('final', 'multiloop', 'trajectory')})
        for stage in completed:
            if stage in artifacts and not artifacts[stage].is_file():
                raise FileNotFoundError(f'completed stage {stage} is missing {artifacts[stage]}')
        common = ['--device', 'cuda:0', '--memory-fraction', args.memory_fraction, '--num-threads', 8]
        if 'capture' not in completed:
            plan_path = root / 'capture_parallel/plan.json'
            pool('capture_prepare', [job('capture_prepare', 'capture_parallel',
                 ['--phase', 'prepare', '--root', root, '--workers', len(gpus), '--model', args.model,
                  '--max-new-tokens', args.max_new_tokens], [], plan_path)])
            plan = json.loads(plan_path.read_text())

            def capture_progress():
                counts = Counter(plan['existing_counts'])
                worker_progress = []
                for worker in plan['workers']:
                    path = Path(worker['out']) / 'progress.json'
                    data = json.loads(path.read_text()) if path.exists() else {}
                    counts.update(data.get('source_counts', {}))
                    worker_progress.append({'index': worker['index'], 'gpu': gpus[worker['index']], **data})
                atomic_json(capture / 'progress.json', {'done': sum(counts.values()), 'total': sum(plan['quotas'].values()),
                            'source_counts': dict(counts), 'quotas': plan['quotas'], 'workers': worker_progress,
                            'updated_at': now(), 'execution': 'parallel'})

            specs = [job(f"capture_worker{worker['index']}", 'capture_protocol',
                         ['--model', args.model, '--prompts', worker['prompts'], '--reserves', worker['reserves'],
                          '--out', worker['out'], '--batch-size', args.capture_batch_size,
                          '--max-new-tokens', args.max_new_tokens] + common,
                         [gpus[worker['index']]], Path(worker['out']) / 'capture_summary.json')
                     for worker in plan['workers'] if sum(worker['quotas'].values())]
            pool('capture_parallel', specs, capture_progress)
            pool('capture_merge', [job('capture_merge', 'capture_parallel',
                 ['--phase', 'merge', '--root', root, '--workers', len(gpus)], [], capture / 'capture_summary.json')])
            finish('capture')

        train_common = ['--data', capture / 'manifest.jsonl', '--frozen-io', capture / 'frozen_io.pt',
                        '--expected-requests', args.expected_train_requests, '--seed', 0,
                        '--requests-per-update', args.requests_per_update,
                        '--max-anchors-per-request', args.max_anchors_per_request,
                        '--batch-size', args.microbatch] + common

        def training(name, role, mode, epochs, gpu_ids, parent=None):
            out = run / name
            options = ['--out', out, '--role', role, '--feature-mode', mode, '--epochs', epochs] + train_common
            if parent:
                options += ['--init-checkpoint', parent]
            if (out / 'latest.pt').exists():
                options += ['--resume']
            return job(name, 'train_protocol', options, gpu_ids, out / 'model.pt', len(gpu_ids) > 1)

        baseline = run / 'baseline/model.pt'
        if 'baseline' not in completed:
            pool('baseline', [training('baseline', 'baseline', 'final', args.baseline_epochs, gpus)])
            finish('baseline')
        modes = ('final', 'multiloop', 'trajectory')
        branch_jobs = [training(f'branch_{mode}', 'branch', mode, args.branch_epochs, [gpus[index]], baseline)
                       for index, mode in enumerate(modes) if f'branch_{mode}' not in completed]
        pool('branches_parallel', branch_jobs)
        for mode in modes:
            finish(f'branch_{mode}')

        if 'speed_evaluation' not in completed:
            evaluation = run / 'evaluation'
            (evaluation / 'panels').mkdir(parents=True, exist_ok=True)
            prompts = read_jsonl(root / 'prompts/eval.jsonl')
            panels = evaluation_panels(prompts, len(gpus), args.eval_per_source)
            specs = []
            for index, panel in enumerate(panels):
                panel_path = evaluation / 'panels' / f'worker{index}.jsonl'
                if panel_path.exists() and read_jsonl(panel_path) != panel:
                    raise ValueError('evaluation panel changed since parallel run started')
                if not panel_path.exists():
                    panel_path.write_text(''.join(json.dumps(row, ensure_ascii=False) + '\n' for row in panel))
                out = evaluation / 'shards' / f'worker{index}'
                specs.append(job(f'evaluation_worker{index}', 'evaluate_protocol',
                    ['--model', args.model, '--prompts', panel_path, '--frozen-io', capture / 'frozen_io.pt',
                     '--checkpoints', *[run / f'branch_{mode}/model.pt' for mode in modes], '--out', out,
                     '--expected-per-source', args.eval_per_source // len(gpus),
                     '--max-new-tokens', args.max_new_tokens, '--seed', 0] + common,
                     [gpus[index]], out / 'summary.json'))
            pool('speed_evaluation_parallel', specs)
            merge_evaluation(evaluation, prompts, gpus)
            finish('speed_evaluation')
        status.update(state='complete', finished_at=now())
        save()


if __name__ == '__main__':
    main()
