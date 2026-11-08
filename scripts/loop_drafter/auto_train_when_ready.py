"""Wait for the registered capture, then select idle GPUs and train from scratch.

This supervisor does not start capture, stop other jobs, or launch evaluation.
"""
import argparse
from contextlib import ExitStack
from datetime import datetime, timezone
import csv
import fcntl
import io
import json
import os
from pathlib import Path
import subprocess
import time


def now():
    return datetime.now(timezone.utc).isoformat()


def write_status(path, **values):
    tmp = path.with_suffix('.tmp')
    tmp.write_text(json.dumps(dict(updated_at=now(), pid=os.getpid(), **values), indent=2) + '\n')
    tmp.replace(path)


def capture_ready(recipe):
    manifest = Path(recipe['training_data']['trajectory_manifest'])
    required = [manifest, manifest.with_name('data_index.json'), manifest.with_name('config.json')]
    missing = [str(p) for p in required if not p.is_file()]
    if missing:
        return False, dict(missing_inputs=missing)
    try:
        rows = [json.loads(line) for line in manifest.open() if line.strip()]
        expected = recipe['training_data']['effective_requests']
        ready = len(rows) == expected and len({r['id'] for r in rows}) == expected
        ready = ready and all(r.get('split') == 'train' for r in rows)
        return ready, dict(observed_requests=len(rows), expected_requests=expected)
    except (OSError, ValueError, KeyError) as error:
        return False, dict(manifest_pending=str(error))


def gpu_inventory():
    def query(fields, kind):
        result = subprocess.run(['nvidia-smi', '--query-' + kind + '=' + fields,
                                 '--format=csv,noheader,nounits'], text=True, capture_output=True,
                                check=True, timeout=20)
        return list(csv.reader(io.StringIO(result.stdout), skipinitialspace=True))
    gpus = query('index,uuid,memory.total,memory.free,utilization.gpu', 'gpu')
    processes = query('gpu_uuid,pid', 'compute-apps')
    occupied = {row[0].strip() for row in processes if len(row) == 2}
    return [dict(index=row[0].strip(), uuid=row[1].strip(), total_mib=int(row[2]),
                 free_mib=int(row[3]), utilization=int(row[4]), has_compute_process=row[1].strip() in occupied)
            for row in gpus]


def eligible_gpus(inventory, minimum_free_mib=24576, max_utilization=20):
    return [g['index'] for g in sorted(inventory, key=lambda g: int(g['index']))
            if not g['has_compute_process'] and g['free_mib'] >= max(minimum_free_mib, g['total_mib'] * .18 + 4096)
            and g['utilization'] <= max_utilization]


def choose_gpus(eligible):
    count = next((n for n in (8, 4, 2, 1) if len(eligible) >= n), 0)
    return eligible[:count]


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--protocol', type=Path, default=Path(__file__).with_name('data_protocol_v2.json'))
    parser.add_argument('--root', type=Path, default=Path('/experiments/loop_drafter'))
    parser.add_argument('--poll-seconds', type=float, default=60)
    args = parser.parse_args(argv)
    if args.poll_seconds < 1:
        raise ValueError('poll interval must be positive')
    recipe_bytes = args.protocol.read_bytes()
    recipe = json.loads(recipe_bytes)
    if recipe['training']['initialization'] != 'from_scratch' or recipe['training']['parent_checkpoint'] is not None:
        raise ValueError('automatic launch requires the registered scratch protocol')
    run = args.root / recipe['run_name']
    run.mkdir(parents=True, exist_ok=True)
    state_file = run / 'auto_train_status.json'
    repo = Path(__file__).resolve().parents[2]
    base = [recipe['runtime']['python'], '-m', 'scripts.loop_drafter.run_loop_memory_experiment',
            '--protocol', str(args.protocol.resolve()), '--root', str(args.root.resolve())]
    policy = dict(minimum_free_mib=24576, max_utilization_percent=20,
                  no_existing_compute_process=True, consecutive_idle_observations=2,
                  poll_seconds=args.poll_seconds, train_only=True)
    with (run / 'auto_train.lock').open('a') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        validated = False
        previous_idle = set()
        try:
            while True:
                if args.protocol.read_bytes() != recipe_bytes:
                    raise ValueError('registered protocol changed while waiting; restart supervisor with the new protocol')
                ready, details = capture_ready(recipe)
                if not ready:
                    validated = False
                    previous_idle = set()
                    write_status(state_file, state='waiting_for_capture', policy=policy, **details)
                    time.sleep(args.poll_seconds)
                    continue
                if not validated:
                    write_status(state_file, state='validating_capture', policy=policy)
                    with (run / 'auto_train_prepare.log').open('a') as log:
                        subprocess.run(base + ['--gpus', '0'], cwd=repo, stdout=log, stderr=subprocess.STDOUT, check=True)
                    prepared = json.loads((run / 'preparation.json').read_text())
                    if prepared['state'] != 'prepared':
                        raise ValueError('complete capture did not pass launcher validation')
                    validated = True
                try:
                    inventory = gpu_inventory()
                except (subprocess.SubprocessError, OSError, ValueError) as error:
                    previous_idle = set()
                    write_status(state_file, state='waiting_for_gpu_probe', policy=policy, error=str(error))
                    time.sleep(args.poll_seconds)
                    continue
                idle = eligible_gpus(inventory)
                selected = choose_gpus([gpu for gpu in idle if gpu in previous_idle])
                previous_idle = set(idle)
                if not selected:
                    write_status(state_file, state='waiting_for_gpus', policy=policy, inventory=inventory,
                                 eligible_now=idle)
                    time.sleep(args.poll_seconds)
                    continue
                # Project-local leases plus a last device probe reduce competing
                # launches; no process on a GPU is ever terminated by this tool.
                with ExitStack() as stack:
                    lease_dir = args.root / 'gpu_leases'
                    lease_dir.mkdir(exist_ok=True)
                    try:
                        for gpu in selected:
                            lease = stack.enter_context((lease_dir / ('gpu' + gpu + '.lock')).open('a'))
                            fcntl.flock(lease, fcntl.LOCK_EX | fcntl.LOCK_NB)
                    except BlockingIOError:
                        previous_idle = set()
                        time.sleep(args.poll_seconds)
                        continue
                    if not set(selected).issubset(eligible_gpus(gpu_inventory())):
                        previous_idle = set()
                        continue
                    command = base + ['--gpus', ','.join(selected), '--execute', '--train-only']
                    write_status(state_file, state='training', policy=policy, selected_gpus=selected, command=command)
                    with (run / 'auto_train_launcher.log').open('a') as log:
                        child = subprocess.Popen(command, cwd=repo, stdout=log, stderr=subprocess.STDOUT)
                        write_status(state_file, state='training', policy=policy, selected_gpus=selected,
                                     launcher_pid=child.pid, command=command)
                        code = child.wait()
                    if code:
                        raise RuntimeError(f'training launcher failed with exit code {code}; see auto_train_launcher.log')
                    result = json.loads((run / 'status.json').read_text())
                    if result['state'] != 'training_complete':
                        raise RuntimeError('launcher exited without completing training')
                    write_status(state_file, state='training_complete', policy=policy, selected_gpus=selected)
                    return
        except Exception as error:
            write_status(state_file, state='failed', policy=policy, error=repr(error))
            raise


if __name__ == '__main__':
    main()
