import json
from pathlib import Path
from unittest.mock import patch

from . import auto_train_when_ready as watcher


def gpu(index, free=30000, utilization=0, occupied=False):
    return dict(index=str(index), uuid=f'GPU-{index}', total_mib=97871,
                free_mib=free, utilization=utilization, has_compute_process=occupied)


def test_gpu_filter_and_supported_group_sizes():
    inventory = [gpu(0), gpu(1, free=24000), gpu(2, utilization=21), gpu(3, occupied=True), gpu(4), gpu(5)]
    assert watcher.eligible_gpus(inventory) == ['0', '4', '5']
    assert watcher.choose_gpus(['0', '4', '5']) == ['0', '4']
    assert watcher.choose_gpus([]) == []
    assert len(watcher.choose_gpus(list(map(str, range(8))))) == 8


def test_capture_waits_for_index_and_complete_unique_manifest(tmp_path):
    path = tmp_path / 'manifest.jsonl'
    recipe = {'training_data': {'trajectory_manifest': str(path), 'effective_requests': 2}}
    assert not watcher.capture_ready(recipe)[0]
    path.write_text('{"id":"a","split":"train"}\n')
    (tmp_path / 'data_index.json').write_text('[]')
    (tmp_path / 'config.json').write_text('{}')
    assert not watcher.capture_ready(recipe)[0]
    path.write_text('{"id":"a","split":"train"}\n' * 2)
    assert not watcher.capture_ready(recipe)[0]
    path.write_text('{"id":"a","split":"train"}\n{"id":"b","split":"train"}\n')
    assert watcher.capture_ready(recipe)[0]


def test_supervisor_waits_for_capture_then_stable_gpu_and_launches_training_only(tmp_path):
    recipe = {'training': {'initialization': 'from_scratch', 'parent_checkpoint': None},
              'runtime': {'python': '/python'}, 'run_name': 'scratch'}
    protocol = tmp_path / 'protocol.json'
    protocol.write_text(json.dumps(recipe))
    run = tmp_path / 'scratch'
    commands = []
    states = []
    original_write = watcher.write_status
    def write(path, **values):
        states.append(values['state'])
        original_write(path, **values)
    def prepare(command, **kwargs):
        commands.append(command)
        (run / 'preparation.json').write_text('{"state":"prepared"}')
    class Child:
        pid = 42
        def __init__(self, command, **kwargs):
            commands.append(command)
            assert '--execute' in command and '--train-only' in command
            assert command[command.index('--gpus') + 1] == '2'
            (run / 'status.json').write_text('{"state":"training_complete"}')
        def wait(self):
            return 0
    with patch.object(watcher, 'capture_ready', side_effect=[(False, {}), (True, {}), (True, {})]), \
         patch.object(watcher, 'gpu_inventory', return_value=[gpu(2)]), \
         patch.object(watcher.time, 'sleep'), patch.object(watcher.subprocess, 'run', side_effect=prepare), \
         patch.object(watcher.subprocess, 'Popen', Child), patch.object(watcher, 'write_status', side_effect=write):
        watcher.main(['--protocol', str(protocol), '--root', str(tmp_path)])
    assert states[:3] == ['waiting_for_capture', 'validating_capture', 'waiting_for_gpus']
    assert states[-1] == 'training_complete'
    assert len(commands) == 2
    assert '--execute' not in commands[0]


def test_resume_preserves_registered_gpu_assignment(tmp_path):
    assert watcher.registered_gpus(tmp_path) is None
    (tmp_path / 'status.json').write_text('{"jobs": {"none": {"state": "interrupted"}}}')
    (tmp_path / 'execution.json').write_text('{"gpus": ["0", "2", "3", "4"]}')
    assert watcher.registered_gpus(tmp_path) == ['0', '2', '3', '4']
