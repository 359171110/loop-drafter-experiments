from contextlib import redirect_stdout
from copy import deepcopy
from dataclasses import replace
import io
import json
from pathlib import Path
import tempfile
import unittest

import torch
import torch.distributed as dist
import torch.multiprocessing as mp
from torch.nn import functional as F

from . import kd_training as kd
from . import test_train_loop_memory as fixtures
from . import train_loop_memory as train
from .loop_memory_draft import LoopMemoryDraft, load_loop_memory_checkpoint
from .train_protocol import shard_indices


def objective(depth=3):
    return dict(schema='loop_kd_objective_v1', loss='kd_only', temperature=1., topk=8,
                depth_weights=[2.] + [1.] * (depth - 1), anchor_sampling='seeded_cyclic_without_replacement',
                teacher='cached_bf16_final_round_fp32_head')


def worker(rank, rendezvous, job_path):
    torch.set_num_threads(1)
    dist.init_process_group('gloo', init_method='file://' + rendezvous, rank=rank, world_size=2)
    try:
        job = torch.load(job_path, weights_only=False)
        model = job['model']; settings = job['settings']
        requests = kd.prepare_requests(job['entries'], shard_indices(list(range(5)), rank, 2), settings, 0)
        stats = kd.synchronized_gradients(model, requests, 2, torch.device('cpu'), objective())
        if rank == 0:
            torch.save(dict(stats=stats, grads={n:p.grad for n,p in model.named_parameters()}), job['result'])
    finally:
        dist.destroy_process_group()


class KDTrainingTests(unittest.TestCase):
    def test_device_batch_matches_reference_at_boundaries(self):
        record = dict(token_ids=torch.arange(13), round_hidden=torch.randn(13, 4, 8), prompt_length=1)
        for roots in ([0, 1, 9, 10], [2, 6, 9]):
            expected = kd.make_loop_memory_batch(record, roots, 5, 6, 'cpu')
            actual = kd.device_batch(kd.stage_record(record, 'cpu'), roots, 5, 6)
            self.assertEqual(set(expected), set(actual))
            for key in expected:
                self.assertTrue(torch.equal(expected[key], actual[key]), key)

    def setUp(self):
        self.f = fixtures.LoopMemoryTrainingTests(); self.f.setUp()
        self.settings = replace(self.f.settings, role='baseline', epochs=2)

    def test_rotating_sampler_covers_real_budget_without_intra_epoch_duplicates(self):
        for n in [1, 63, 64, 65, 128, 129, 255, 256, 511]:
            batches = [kd.cyclic_roots(17, 17 + n + 1, 128, 0, 'sample', e) for e in range(4)]
            self.assertEqual(len(set().union(*map(set, batches))), n)
            self.assertTrue(all(len(set(b)) == len(b) == min(n, 128) for b in batches))
            self.assertEqual(batches[2], kd.cyclic_roots(17, 17 + n + 1, 128, 0, 'sample', 2))
        a=kd.cyclic_roots(3, 515, 64, 0, 'x', 0);b=kd.cyclic_roots(3, 515, 64, 0, 'x', 1)
        self.assertFalse(set(a)&set(b))

    def test_teacher_alignment_and_bucket_kl_match_dense_distribution(self):
        torch.manual_seed(0)
        record=dict(round_hidden=torch.randn(9,4,8));head=torch.randn(23,8);roots=[2,5]
        ids, logp=kd.teacher_buckets(record,roots,3,head,8)
        scores=F.linear(record['round_hidden'][torch.tensor(roots)[:,None]+1+torch.arange(3),-1],head)
        teacher=scores.softmax(-1);tp=teacher.gather(-1,ids)
        buckets=torch.cat((tp,1-tp.sum(-1,keepdim=True)),-1)
        torch.testing.assert_close(logp.exp(),buckets,rtol=2e-5,atol=2e-7)
        student=torch.randn_like(scores,requires_grad=True);sp=student.softmax(-1).gather(-1,ids)
        sb=torch.cat((sp,1-sp.sum(-1,keepdim=True)),-1)
        expected=F.kl_div(sb.log(),buckets,reduction='none').sum(-1)
        actual=kd.bucket_kl(student,ids,logp)
        torch.testing.assert_close(actual,expected,rtol=2e-5,atol=2e-6)
        self.assertFalse(logp.requires_grad)

    def test_microbatch_and_distributed_global_normalization(self):
        with tempfile.TemporaryDirectory() as temp:
            root=Path(temp);entries,frozen,_=self.f.fixture(root)
            base=LoopMemoryDraft(self.f.config,self.f.embedding)
            requests=kd.prepare_requests(entries,list(range(5)),self.settings,0)
            model=deepcopy(base);stats=kd.accumulate_gradients(model,requests,2,torch.device('cpu'),objective())
            grads={n:p.grad.clone() for n,p in model.named_parameters() if p.grad is not None}
            model.zero_grad(set_to_none=True)
            large=kd.accumulate_gradients(model,requests,9,torch.device('cpu'),objective())
            self.assertAlmostEqual(stats['loss'],large['loss'],places=5)
            for n,p in model.named_parameters():
                if n in grads:torch.testing.assert_close(p.grad,grads[n],rtol=3e-4,atol=2e-5)
            job=root/'job.pt';result=root/'result.pt'
            torch.save(dict(model=base,settings=self.settings,entries=entries,result=str(result)),job)
            mp.spawn(worker,args=(str(root/'rendezvous'),str(job)),nprocs=2,join=True)
            got=torch.load(result,weights_only=True)
            self.assertEqual(got['stats']['request_ids'],stats['request_ids'])
            self.assertEqual(got['stats']['weighted_labels'],stats['weighted_labels'])
            self.assertAlmostEqual(got['stats']['loss'],stats['loss'],places=6)
            for n,g in grads.items():torch.testing.assert_close(got['grads'][n],g,rtol=3e-4,atol=2e-5)

    def test_scratch_resume_keeps_objective_optimizer_and_sampling(self):
        with tempfile.TemporaryDirectory() as temp, redirect_stdout(io.StringIO()):
            root=Path(temp);entries,frozen,_=self.f.fixture(root);base=LoopMemoryDraft(self.f.config,self.f.embedding)
            train.train_epochs(deepcopy(base),entries,self.settings,root/'full',frozen,from_scratch=True,objective=objective())
            train.train_epochs(deepcopy(base),entries,self.settings,root/'split',frozen,from_scratch=True,objective=objective(),stop_after_steps=1)
            model,payload=load_loop_memory_checkpoint(root/'split/latest.pt')
            self.assertEqual(payload['protocol'],kd.PROTOCOL)
            with self.assertRaisesRegex(ValueError,'objective'):
                train.train_epochs(model,entries,self.settings,root/'split',frozen,from_scratch=True,resume_payload=payload)
            train.train_epochs(model,entries,self.settings,root/'split',frozen,from_scratch=True,resume_payload=payload,objective=objective())
            full=torch.load(root/'full/model.pt',weights_only=True);split=torch.load(root/'split/model.pt',weights_only=True)
            self.f.nested_equal(full['model'],split['model']);self.f.nested_equal(full['optimizer'],split['optimizer'])
            self.assertEqual(full['objective'],objective())
