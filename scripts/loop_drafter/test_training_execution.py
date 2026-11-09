from copy import deepcopy
from dataclasses import replace
from pathlib import Path
import tempfile
import unittest
import torch
from . import test_train_loop_memory as fixtures
from . import train_loop_memory as train
from .loop_memory_draft import LoopMemoryDraft,load_loop_memory_checkpoint
from .train_protocol import prepare_requests


class ExecutionTests(unittest.TestCase):
    def setUp(self):
        self.f=fixtures.LoopMemoryTrainingTests();self.f.setUp()
        self.plan=dict(schema='loop_training_execution_v1',microbatch=4,from_step=2)

    def test_switch_boundary(self):
        self.assertEqual(train.execution_microbatch(self.f.settings,self.plan,1),2)
        self.assertEqual(train.execution_microbatch(self.f.settings,self.plan,2),4)

    def test_microbatch_keeps_global_loss_and_gradient(self):
        with tempfile.TemporaryDirectory() as tmp:
            entries,io,_=self.f.fixture(Path(tmp))
            requests=prepare_requests(entries,[0,1,2,3],self.f.settings,0)
            model=LoopMemoryDraft(self.f.config,self.f.embedding)
            stats=train.accumulate_gradients(model,requests,2,torch.device('cpu'))
            grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            large=train.accumulate_gradients(model,requests,4,torch.device('cpu'))
            self.assertEqual(stats['valid_labels'],large['valid_labels'])
            self.assertEqual(stats['request_ids'],large['request_ids'])
            self.assertAlmostEqual(stats['loss'],large['loss'],places=5)
            for n,p in model.named_parameters():
                if n in grads:torch.testing.assert_close(p.grad,grads[n],rtol=2e-4,atol=1e-5)

    def test_schedule_survives_resume_and_rejects_changes(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp);entries,io,_=self.f.fixture(root)
            settings=replace(self.f.settings,role='baseline',epochs=2)
            base=LoopMemoryDraft(self.f.config,self.f.embedding)
            self.f.quiet(deepcopy(base),entries,settings,root/'full',io,from_scratch=True,execution_plan=self.plan)
            self.f.quiet(deepcopy(base),entries,settings,root/'split',io,from_scratch=True,execution_plan=self.plan,stop_after_steps=1)
            model,payload=load_loop_memory_checkpoint(root/'split/latest.pt')
            self.f.quiet(model,entries,settings,root/'split',io,from_scratch=True,resume_payload=payload)
            full=torch.load(root/'full/model.pt',weights_only=True);split=torch.load(root/'split/model.pt',weights_only=True)
            self.f.nested_equal(full['model'],split['model'])
            self.f.nested_equal(full['optimizer'],split['optimizer'])
            self.assertEqual(full['execution_plan'],self.plan)
            self.assertEqual([r['execution_microbatch'] for r in self.f.rows(root/'full')],[2,2,4,4])
            with self.assertRaisesRegex(ValueError,'schedule changed'):
                self.f.quiet(model,entries,settings,root/'split',io,from_scratch=True,resume_payload=payload,execution_plan=self.plan|dict(microbatch=8))
