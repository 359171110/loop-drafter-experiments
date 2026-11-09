"""Weighted top-K+tail distillation and deterministic cyclic anchor coverage."""
import hashlib
import math
import random

import torch
import torch.distributed as dist
from torch.nn import functional as F

from .loop_memory_draft import make_loop_memory_batch
from .train import amp_context, load_record, validate_record
from .train_protocol import distributed_context, merge_statistics, request_label_counts

PROTOCOL = 'loop_memory_from_scratch_kd_v1'


def validate_objective(value, depth):
    expected = {'schema', 'loss', 'temperature', 'topk', 'depth_weights', 'anchor_sampling', 'teacher'}
    if (not isinstance(value, dict) or set(value) != expected
            or value['schema'] != 'loop_kd_objective_v1' or value['loss'] != 'kd_only'
            or value['temperature'] != 1.0 or type(value['topk']) is not int or value['topk'] < 1
            or value['anchor_sampling'] != 'seeded_cyclic_without_replacement'
            or value['teacher'] != 'cached_bf16_final_round_fp32_head'):
        raise ValueError('invalid KD objective')
    weights = value['depth_weights']
    if len(weights) != depth or any(not math.isfinite(w) or w <= 0 for w in weights):
        raise ValueError('KD depth weights must be positive and match the draft depth')
    return value


def cyclic_roots(prompt_length, token_count, cap, seed, request_id, epoch):
    """Walk one seeded permutation across epochs; no repeats before coverage."""
    roots = list(range(prompt_length - 1, token_count - 2))
    if min(cap, len(roots)) < 1 or epoch < 0:
        raise ValueError('invalid cyclic anchor budget')
    if len(roots) <= cap:
        return roots
    key = hashlib.blake2b(f'{seed}:{request_id}'.encode(), digest_size=16).digest()
    random.Random(int.from_bytes(key, 'big')).shuffle(roots)
    start = epoch * cap % len(roots)
    return sorted(roots[(start + i) % len(roots)] for i in range(cap))


def prepare_requests(entries, indices, settings, epoch):
    requests = []
    for index in indices:
        entry = entries[index]
        record = load_record(entry['path'])
        roots = cyclic_roots(int(record['prompt_length']), len(record['token_ids']),
                             settings.max_anchors_per_request, settings.seed, entry['id'], epoch)
        requests.append((entry, record, roots))
    return requests


def weighted_count(requests, weights):
    return sum(sum(weights[:min(len(weights), len(record['token_ids']) - root - 2)])
               for _, record, roots in requests for root in roots)


@torch.no_grad()
def teacher_buckets(record, roots, depth, head, topk):
    # Child d=0 is token[root+2]; its teacher state is hidden[root+1].
    # These future states are labels only and never enter drafter inputs.
    roots = torch.as_tensor(roots, dtype=torch.long)
    positions = roots[:, None] + 1 + torch.arange(depth)[None]
    positions = positions.clamp_max(len(record['round_hidden']) - 1)
    unique, inverse = positions.flatten().unique(sorted=True, return_inverse=True)
    hidden = record['round_hidden'][unique, -1].to(device=head.device, dtype=torch.float32)
    with torch.autocast(head.device.type, enabled=False):
        scores = F.linear(hidden, head.float())
        if not 0 < topk < scores.shape[-1]:
            raise ValueError('teacher topk must leave a nonempty tail')
        values, ids = scores.topk(topk, dim=-1)
        tail = scores.scatter(-1, ids, -torch.inf).logsumexp(-1, keepdim=True)
        logp = torch.cat((values, tail), -1) - scores.logsumexp(-1, keepdim=True)
    shape = (len(roots), depth)
    return ids[inverse].reshape(*shape, topk), logp[inverse].reshape(*shape, topk + 1)


def bucket_kl(scores, ids, teacher_logp):
    """KL over teacher top-K identities and one exact aggregate tail bucket."""
    score = scores.float()
    top = score.gather(-1, ids)
    tail = score.scatter(-1, ids, -torch.inf).logsumexp(-1, keepdim=True)
    student_logp = torch.cat((top, tail), -1) - score.logsumexp(-1, keepdim=True)
    return (teacher_logp.exp() * (teacher_logp - student_logp)).sum(-1)


def accumulate_gradients(model, requests, microbatch, device, objective, denominator=None):
    depth = model.config.depth
    weights = objective['depth_weights']
    local_weight = weighted_count(requests, weights)
    denominator = local_weight if denominator is None else denominator
    if denominator <= 0 or microbatch < 1:
        raise ValueError('KD update needs a positive weighted denominator')
    per_request_labels = request_label_counts(requests, depth)
    # Scalar diagnostics are accumulated on device, with one transfer per update.
    sums = torch.zeros(3 + 3 * depth, dtype=torch.float64, device=device)
    for _, record, roots in requests:
        validate_record(record, model.config)
        for start in range(0, len(roots), microbatch):
            selected = roots[start:start + microbatch]
            batch = make_loop_memory_batch(record, selected, depth, model.config.window, device)
            ids, logp = teacher_buckets(record, selected, depth, model.shared_lm_head, objective['topk'])
            with amp_context(device):
                logits = model.forward_batch(batch)
            per_label = bucket_kl(logits, ids, logp)
            weight = batch['valid'] * per_label.new_tensor(weights)
            kd = (per_label * weight).sum()
            if not bool(torch.isfinite(kd)):
                raise FloatingPointError('nonfinite KD loss')
            (kd / denominator).backward()
            with torch.no_grad():
                ce = F.cross_entropy(logits.float().flatten(0, 1), batch['labels'].flatten(),
                                     ignore_index=-100, reduction='none').reshape_as(per_label)
                sums[0] += kd.detach().double()
                sums[1] += ce.sum().double()
                sums[2] += (ce * weight).sum().double()
                sums[3:3 + depth] += batch['valid'].sum(0)
                sums[3 + depth:3 + 2 * depth] += ((logits.argmax(-1) == batch['labels']) & batch['valid']).sum(0)
                sums[3 + 2 * depth:] += (per_label.detach() * batch['valid']).sum(0).double()
    values = sums.cpu().tolist()
    labels = [int(x) for x in values[3:3 + depth]]
    correct = [int(x) for x in values[3 + depth:3 + 2 * depth]]
    return dict(loss=values[0] / max(local_weight, 1), kd_sum=values[0], weighted_ce_sum=values[2],
                weighted_labels=local_weight, ce=values[1] / max(sum(labels), 1),
                nll_sum=values[1], valid_labels=sum(labels), labels_by_depth=labels, correct_by_depth=correct,
                kd_sum_by_depth=values[3 + 2 * depth:], top1_by_depth=[c / max(n, 1) for c, n in zip(correct, labels)],
                anchors=sum(len(roots) for _, _, roots in requests),
                request_ids=[entry['id'] for entry, _, _ in requests],
                request_anchors=[len(roots) for _, _, roots in requests], request_labels=per_request_labels)


def synchronized_gradients(model, requests, microbatch, device, objective):
    _, world_size = distributed_context()
    denominator = torch.tensor(weighted_count(requests, objective['depth_weights']), device=device, dtype=torch.float64)
    if world_size > 1:
        dist.all_reduce(denominator, op=dist.ReduceOp.SUM)
    local = accumulate_gradients(model, requests, microbatch, device, objective, float(denominator))
    if world_size == 1:
        return local
    parameters = [p for p in model.parameters() if p.requires_grad]
    flat = torch.cat([(p.grad if p.grad is not None else torch.zeros_like(p)).reshape(-1) for p in parameters])
    dist.all_reduce(flat, op=dist.ReduceOp.SUM)
    offset = 0
    for p in parameters:
        p.grad = flat[offset:offset + p.numel()].view_as(p)
        offset += p.numel()
    parts = [None] * world_size
    dist.all_gather_object(parts, local)
    result = merge_statistics(parts)
    for key in ('kd_sum', 'weighted_ce_sum', 'weighted_labels'):
        result[key] = sum(p[key] for p in parts)
    result['loss'] = result['kd_sum'] / result['weighted_labels']
    result['ce'] = result['nll_sum'] / result['valid_labels']
    result['kd_sum_by_depth'] = [sum(p['kd_sum_by_depth'][d] for p in parts) for d in range(model.config.depth)]
    return result
