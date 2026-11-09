from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch
import torch
from . import evaluate_loop_memory as evaluation
from .loop_memory_draft import LoopMemoryConfig, LoopMemoryDraft
from .test_evaluate_loop_memory import protocol_payload
from .test_evaluate import FakeTarget, EOS
from .test_kd_training import objective


class VersionEvaluationTests(unittest.TestCase):
    def test_complete_recipes_rotate_and_pair_all_methods(self):
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root=Path(temp);target_path=str(root/'target');frozen=root/'io.pt'
            class Tokenizer:
                name_or_path=target_path
                eos_token_id=EOS
                def __len__(self):return 31
            target=FakeTarget(3);target.tokenizer=Tokenizer()
            config=LoopMemoryConfig(memory_mode='trajectory',target_hidden=4,vocab_size=31,width=8,mlp_width=16,
                    layers=1,init_rank=4,prefix_width=3,depth=3,state_dim=3,state_head_dim=4,memory_width=8,memory_heads=2,window=8)
            model=LoopMemoryDraft(config,torch.randn(31,4)).eval()
            old=protocol_payload('trajectory',scratch=True,frozen_io=frozen);old['config']=config.dict()
            new=deepcopy(old);new.update(protocol=evaluation.kd_training.PROTOCOL,objective=objective())
            new['settings']['max_anchors_per_request']=128
            mapping={}
            payloads={}
            for name,payload in [('v1_ce',old),('v1_kd',new)]:
                path=root/(name+'.pt');path.write_text('fixture');mapping[name]=str(path);payloads[str(path)]=payload
            versions=root/'versions.json';versions.write_text(json.dumps(mapping))
            prompts=root/'prompts.jsonl';prompts.write_text(''.join(json.dumps(dict(id=str(i),source='test',split='eval',
                    input_ids=[2,3,5],tokenizer_path=target_path))+'\n' for i in range(3)))
            with patch('scripts.loop_drafter.target.OuroTarget.from_pretrained',return_value=target),patch.object(evaluation,'load_loop_memory_checkpoint',side_effect=lambda path,**kw:(deepcopy(model),payloads[path])):
                result=evaluation.main(['--model',target_path,'--prompts',str(prompts),'--version-checkpoints',str(versions),
                    '--frozen-io',str(frozen),'--out',str(root/'out'),'--device','cpu','--max-new-tokens','9'])
            rows=[json.loads(l) for l in (root/'out/requests.jsonl').read_text().splitlines()]
            self.assertEqual(len(rows),9)
            self.assertEqual([r['method'] for r in rows],['ar','v1_ce','v1_kd','v1_ce','v1_kd','ar','v1_kd','ar','v1_ce'])
            self.assertEqual(set(result['hardware']['training_identity']),{'v1_ce','v1_kd'})
            for row in result['results']:
                if row['method']!='ar':self.assertIsNotNone(row['first_child_acceptance_rate'])
            ref=evaluation.checkpoint_training_identity(old,config.dict(),frozen)
            actual=evaluation.checkpoint_training_identity(new,config.dict(),frozen)
            evaluation.require_version_comparison_identity(ref,actual)
            with self.assertRaises(ValueError):evaluation.require_matching_training_identity(ref,actual)
            actual['data_index_sha256']='changed'
            with self.assertRaises(ValueError):evaluation.require_version_comparison_identity(ref,actual)
