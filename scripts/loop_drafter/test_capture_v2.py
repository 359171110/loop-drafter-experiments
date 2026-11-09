import unittest
from unittest.mock import patch
from types import SimpleNamespace
import torch
from . import capture_v2 as c

class CaptureV2Tests(unittest.TestCase):
    def test_length_tiers(self):
        self.assertEqual([c.batch_limit(n) for n in [1,512,513,1280,1281,2048]], [4,4,2,2,1,1])
        with self.assertRaises(ValueError): c.batch_limit(2049)

    def test_oom_preserves_rows_and_budget(self):
        seen=[]
        def capture(target, ids, budget, eos):
            self.assertEqual(budget,512)
            seen.append(ids.tolist())
            if len(ids)>1: raise torch.cuda.OutOfMemoryError('test')
            return [int(ids[0,0])]
        with patch.object(c,'capture_batch',capture), patch.object(c.torch.cuda,'empty_cache'):
            result=c.capture_retry(None,torch.tensor([[3],[7],[9],[11]]),{2})
        self.assertEqual(result,[3,7,9,11])
        self.assertEqual(len(seen),7)

    def test_singleton_oom_is_not_replaced(self):
        with patch.object(c,'capture_batch',side_effect=torch.cuda.OutOfMemoryError('test')):
            with self.assertRaises(torch.cuda.OutOfMemoryError):
                c.capture_retry(None,torch.tensor([[3]]),{2})

    def test_reuse_requires_frozen_prefix_and_eos(self):
        prompt=dict(id='x',source='s',input_ids=[3,4])
        record=dict(id='x',source='s',split='train',prompt_length=2,
                    token_ids=torch.tensor([3,4,5,2]),round_hidden=torch.zeros(3,4,8,dtype=torch.bfloat16))
        config=SimpleNamespace(rounds=4,target_hidden=8,vocab_size=20)
        with patch.object(c,'LoopMemoryConfig',return_value=config):
            c.validate_trajectory(record,prompt,{2},True)
            with self.assertRaises(ValueError): c.validate_trajectory(record,prompt|dict(input_ids=[3,6]),{2},True)
            with self.assertRaises(ValueError): c.validate_trajectory(record|dict(token_ids=torch.tensor([3,4,5,6])),prompt,{2},True)
            with self.assertRaises(ValueError): c.validate_trajectory(record|dict(token_ids=torch.tensor([3,4,2,2])),prompt,{2},True)

    def test_publication_builds_trainer_compatible_index(self):
        import json
        from pathlib import Path
        from tempfile import TemporaryDirectory
        from .train_history import load_training_index
        from .loop_memory_draft import LoopMemoryConfig
        with TemporaryDirectory() as tmp:
            root=Path(tmp); out=root/'data_v2/capture'; prompts=root/'data_v2/prompts'
            folder=out/'worker_0'; folder.mkdir(parents=True); prompts.mkdir()
            prompt=dict(id='x',source='s',domain='math',input_ids=[3,4],origin='new_v2')
            c._write_jsonl(prompts/'train.jsonl',[prompt]); c._write_jsonl(prompts/'reserves.jsonl',[])
            c._write_jsonl(out/'reuse.jsonl',[])
            path=folder/'x.pt'
            row=dict(id='x',original_id='x',source='s',domain='math',split='train',slot=0,
                     prompt_length=2,generated_length=2,path=str(path))
            torch.save(row|dict(token_ids=torch.tensor([3,4,5,2]),
                round_hidden=torch.zeros(3,4,LoopMemoryConfig().target_hidden,dtype=torch.bfloat16)),path)
            c._write_jsonl(folder/'manifest.jsonl',[row])
            c._atomic_json(folder/'status.json',dict(state='complete',done=1))
            plan=dict(inputs={},model_files={},model=tmp,reuse_sha256=c.digest(out/'reuse.jsonl'),
                      workers=[dict(index=0,tasks=1)],expected=1,source_counts={'s':1},reuse=0)
            with patch('transformers.AutoTokenizer.from_pretrained',return_value=SimpleNamespace(eos_token_id=2)):
                c.finalize(root,plan)
            entries=load_training_index(out/'manifest.jsonl',LoopMemoryConfig(),1,out/'data_index.json')
            self.assertEqual(entries[0]['available_anchors'],1)
            self.assertTrue(json.loads((out/'summary.json').read_text())['complete'])
