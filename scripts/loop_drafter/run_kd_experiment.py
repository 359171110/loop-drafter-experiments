"""One scratch KD recipe, followed by paired AR / v1 CE / new KD evaluation."""
import argparse
from collections import Counter
from contextlib import ExitStack
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path
import shutil
import signal
import subprocess
import sys
import time

from .auto_train_when_ready import eligible_gpus, gpu_inventory
from .capture import read_jsonl
from .evaluate_protocol import aggregate, atomic_json
from .run_loop_memory_experiment import panels


def now():
    return datetime.now(timezone.utc).isoformat()


def freeze_json(path, value):
    if path.exists() and json.loads(path.read_text()) != value:
        raise ValueError(f'registered configuration changed: {path}')
    if not path.exists():
        atomic_json(path, value)


def prepare(args):
    run = args.run.resolve(); run.mkdir(parents=True, exist_ok=True)
    baseline = args.baseline_run.resolve()
    prior = json.loads((baseline / 'recipe.json').read_text())
    objective = json.loads(Path(__file__).with_name('kd_objective_v1.json').read_text())
    capture = Path(prior['training_data']['trajectory_manifest']).parent
    entries = json.loads((capture / 'data_index.json').read_text())
    if len(entries) != 16384 or not json.loads((capture / 'summary.json').read_text()).get('complete'):
        raise ValueError('the registered 16K capture must be complete')
    coverage = dict(available_anchors=sum(e['available_anchors'] for e in entries),
                    first_position_exposures=sum(4 * min(128, e['available_anchors']) for e in entries),
                    unique_first_positions=sum(min(512, e['available_anchors']) for e in entries),
                    former_first_position_exposures=sum(4 * min(64, e['available_anchors']) for e in entries))
    recipe = dict(version='loop_memory_v1_kd_first2_cycle128_seed0', architecture=prior['architecture_version'],
                  objective=objective, training=dict(initialization='from_scratch', parent_checkpoint=None,
                  seed=0, epochs=4, updates=8192, requests_per_update=8, max_anchors_per_request=128,
                  microbatch=64, memory_mode='trajectory', optimizer=prior['training']),
                  training_data=prior['training_data'], capture=prior['capture'], evaluation=prior['evaluation'],
                  baseline_checkpoint=str(baseline / 'trajectory/model.pt'), frozen_io=prior['training']['frozen_io'], coverage=coverage,
                  comparison='complete training recipes; increased anchor budget is part of the new version')
    # The inherited optimizer record is filtered so it cannot describe stale arms or loss settings.
    recipe['training']['optimizer'] = {k: prior['training'][k] for k in
        ('peak_lr','floor_lr','warmup_fraction','lr_decay','adam_betas','weight_decay','grad_clip')}
    freeze_json(run / 'recipe.json', recipe)
    freeze_json(run / 'objective.json', objective)
    repo = Path(__file__).resolve().parents[2]
    files = list((repo/'scripts/loop_drafter').glob('*.py'))
    files += [repo/'scripts/fest_hybrid/model.py', repo/'scripts/fest_hybrid_prefix_tree/model.py']
    identity = {str(p.relative_to(repo)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(files)
                if not p.name.startswith('test_')}
    freeze_json(run/'source_identity.json', identity)
    for name in identity:
        dst=run/'source_snapshot'/name;dst.parent.mkdir(parents=True,exist_ok=True)
        if not dst.exists():shutil.copy2(repo/name,dst)
    freeze_json(run/'versions.json', {'v1_ce':str(baseline/'trajectory/model.pt'), 'v1_kd':str(run/'trajectory/model.pt')})
    return recipe, repo


def training_command(args, recipe):
    t=recipe['training'];data=recipe['training_data'];run=args.run.resolve()
    command=[sys.executable,'-m','torch.distributed.run','--standalone',f'--nproc_per_node={len(args.gpus)}',
             '-m','scripts.loop_drafter.train_loop_memory','--from-scratch','--data',data['trajectory_manifest'],
             '--frozen-io',recipe['frozen_io'],
             '--index-file',str(Path(data['trajectory_manifest']).with_name('data_index.json')),
             '--out',str(run/'trajectory'),'--memory-mode','trajectory','--objective-config',str(run/'objective.json'),
             '--epochs','4','--expected-requests','16384','--requests-per-update','8',
             '--max-anchors-per-request','128','--microbatch','64','--device','cuda',
             '--num-threads','4','--memory-fraction','.30','--save-every','100']
    if (run/'trajectory/latest.pt').exists():command.append('--resume')
    if args.stop_after_steps:command+=['--stop-after-steps',str(args.stop_after_steps)]
    return command


def monitor_evaluation(active, gpus, log):
    result=subprocess.run(['nvidia-smi','--query-compute-apps=gpu_uuid,pid','--format=csv,noheader,nounits'],
                          capture_output=True,text=True,check=True,timeout=20)
    uuids={g['uuid'] for g in gpu_inventory() if g['index'] in gpus}
    foreign=[]
    for line in result.stdout.splitlines():
        uuid,pid=(x.strip() for x in line.split(','))
        if uuid in uuids and int(pid) not in active:foreign.append(dict(uuid=uuid,pid=int(pid)))
    with log.open('a') as handle:handle.write(json.dumps(dict(time=now(),foreign_processes=foreign))+'\n')
    return not foreign


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True);p.add_argument('--baseline-run',type=Path,required=True)
    p.add_argument('--gpus',default='0,2,3,4');p.add_argument('--prepare-only',action='store_true')
    p.add_argument('--stop-after-steps',type=int,default=0)
    args=p.parse_args(argv);args.gpus=args.gpus.split(',')
    if len(args.gpus) not in (1,2,4) or len(set(args.gpus))!=len(args.gpus):raise ValueError('use 1, 2 or 4 distinct GPUs')
    recipe,repo=prepare(args);run=args.run.resolve();baseline=args.baseline_run.resolve()
    command=training_command(args,recipe)
    freeze_json(run/'execution.json',dict(gpus=args.gpus,python=sys.executable,stop_after_steps=args.stop_after_steps,
                                         global_requests_per_update=8,automatic_evaluation=not bool(args.stop_after_steps)))
    if args.prepare_only:
        print(json.dumps(dict(run=str(run),command=command,coverage=recipe['coverage']),indent=2));return
    children=[]
    status=dict(state='waiting_for_gpus',updated_at=now(),pid=os.getpid(),gpus=args.gpus)
    def save(**kwargs):
        status.update(kwargs,updated_at=now());atomic_json(run/'status.json',status)
    with ExitStack() as stack:
        own=stack.enter_context((run/'launcher.lock').open('a'));fcntl.flock(own,fcntl.LOCK_EX|fcntl.LOCK_NB)
        lease_dir=baseline.parent/'gpu_leases';lease_dir.mkdir(exist_ok=True)
        last=set()
        while True:
            idle=set(eligible_gpus(gpu_inventory()))
            if set(args.gpus).issubset(idle&last):break
            last=idle;save(state='waiting_for_gpus',eligible=sorted(idle));time.sleep(5)
        for gpu in args.gpus:
            lease=stack.enter_context((lease_dir/f'gpu{gpu}.lock').open('a'));fcntl.flock(lease,fcntl.LOCK_EX|fcntl.LOCK_NB)
        if not set(args.gpus).issubset(eligible_gpus(gpu_inventory())):raise RuntimeError('GPU became occupied before launch')
        def launch(command,name,gpus):
            stream=stack.enter_context((run/(name+'.log')).open('a'))
            env=dict(os.environ,CUDA_VISIBLE_DEVICES=','.join(gpus),OMP_NUM_THREADS='4',MKL_NUM_THREADS='4',PYTHONUNBUFFERED='1')
            child=subprocess.Popen(command,cwd=repo,env=env,stdout=stream,stderr=subprocess.STDOUT,start_new_session=True,
                                   pass_fds=(own.fileno(),))
            children.append(child);return child
        try:
            if not (run/'trajectory/model.pt').exists():
                child=launch(command,'training',args.gpus);save(state='training',training_pid=child.pid,training_started_at=now())
                code=child.wait()
                if code:raise RuntimeError(f'training failed ({code}); see training.log')
            if args.stop_after_steps:
                save(state='preflight_complete');return
            if not (run/'trajectory/model.pt').is_file():raise RuntimeError('final checkpoint missing')
            save(state='training_complete',training_completed_at=now())
            for name,definition in [('core',recipe['evaluation']['core']),('reasoning_extension',recipe['evaluation']['reasoning_extension'])]:
                root=run/'evaluation'/name;root.mkdir(parents=True,exist_ok=True)
                prompts=read_jsonl(definition['manifest']);shards=panels(prompts,len(args.gpus),0)
                active=[]
                for i,(gpu,panel) in enumerate(zip(args.gpus,shards)):
                    panel_path=root/f'panel{i}.jsonl';serialized=''.join(json.dumps(row,ensure_ascii=False)+'\n' for row in panel)
                    if panel_path.exists() and panel_path.read_text()!=serialized:raise ValueError('evaluation panel changed')
                    panel_path.write_text(serialized)
                    out=root/f'worker{i}'
                    if (out/'summary.json').exists():continue
                    c=[sys.executable,'-m','scripts.loop_drafter.evaluate_loop_memory',
                       '--model',recipe['capture']['target'],'--prompts',str(panel_path),
                       '--version-checkpoints',str(run/'versions.json'),
                       '--frozen-io',recipe['frozen_io'],
                       '--out',str(out),'--expected-counts',json.dumps(dict(Counter(x['source'] for x in panel))),
                       '--max-prompt-tokens','2048','--max-new-tokens','256','--num-threads','4','--memory-fraction','.30']
                    # Use the exact tokenizer provenance attached to the frozen v1 suite.
                    comparison=json.loads((baseline/'comparison_protocol.json').read_text())
                    sources=comparison['inputs'][name+'_sources']['path']
                    c+=['--tokenizer-manifest',sources]
                    active.append(launch(c,f'evaluation_{name}_{i}',[gpu]))
                save(state='evaluating',panel=name,evaluation_pids=[c.pid for c in active])
                clean=True;allowed={c.pid for c in children}
                while any(c.poll() is None for c in active):
                    clean=monitor_evaluation(allowed,args.gpus,root/'gpu_monitor.jsonl') and clean
                    time.sleep(5)
                if any(c.returncode for c in active):raise RuntimeError(f'evaluation failed: {name}')
                rows=[];hardware=[]
                for i,panel in enumerate(shards):
                    shard=read_jsonl(root/f'worker{i}/requests.jsonl')
                    expected={(x['id'],m) for x in panel for m in ('ar','v1_ce','v1_kd')}
                    keys=[(x['id'],x['method']) for x in shard]
                    if len(keys)!=len(expected) or set(keys)!=expected:raise RuntimeError('incomplete paired evaluation')
                    rows.extend(shard);hardware.append(json.loads((root/f'worker{i}/hardware.json').read_text()))
                journal=root/'gpu_monitor.jsonl'
                if journal.exists():clean=clean and all(not json.loads(l)['foreign_processes'] for l in journal.read_text().splitlines())
                atomic_json(root/'summary.json',dict(results=aggregate(rows),source_counts=dict(Counter(x['source'] for x in prompts)),
                            timing_scope='exclusive_gpu_interleaved' if clean else 'shared_gpu_diagnostic',hardware=hardware))
            save(state='complete',completed_at=now())
        except BaseException as error:
            for child in children:
                if child.poll() is None:
                    os.killpg(child.pid,signal.SIGTERM)
                    try:child.wait(timeout=15)
                    except subprocess.TimeoutExpired:os.killpg(child.pid,signal.SIGKILL);child.wait()
            save(state='failed',error=repr(error));raise

if __name__=='__main__':main()
