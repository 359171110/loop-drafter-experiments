"""Full short-prefix correction over the unchanged hybrid parallel backbone.

For child d the head receives exactly [bonus, child1, ..., child(d-1)].
No predicted child is fed back through the backbone or Target KV attention.
The frozen Target embedding and merged deployment projection are unchanged.
"""
from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F

from scripts.fest_hybrid.model import (
    Config as HybridConfig,
    HybridDraft as BaseHybridDraft,
    prefix_indices,
)


@dataclass(frozen=True)
class Config(HybridConfig):
    architecture: str = 'fest_bonus_hybrid_prefix_tree_v1'
    prefix_width: int = 128


def teacher_prefixes(previous):
    """Expand [bonus, gold1, ..., gold4] into causal, ordered head inputs.

    Returns ids[B,D,D], mask[B,D,D]; for the child at zero-based depth d,
    only columns 0 through d are valid. Invalid ids are zeroed so even code
    inspecting the returned tensors cannot accidentally read future gold.
    """
    if previous.ndim != 2:
        raise ValueError('previous must have shape [batch, depth]')
    batch, depth = previous.shape
    mask = torch.ones(depth, depth, device=previous.device, dtype=torch.bool).tril()
    mask = mask[None].expand(batch, -1, -1)
    ids = previous[:, None, :].expand(-1, depth, -1).masked_fill(~mask, 0)
    return ids, mask


class _CorrectionHead(nn.Module):
    """Functional-call view of only correction weights, never the backbone."""
    def __init__(self, model):
        super().__init__()
        self.config = model.config
        self.register_buffer('shared_embed', model.shared_embed, persistent=False)
        for name in ('prefix_token_proj', 'prefix_proj', 'previous_proj',
                     'correction_norm', 'correction', 'output_norm'):
            setattr(self, name, getattr(model, name))

    def forward(self, features, prefix_ids, prefix_mask):
        return HybridDraft.correct(self, features, prefix_ids, prefix_mask)


class HybridDraft(BaseHybridDraft):
    def __init__(self, config, embed):
        super().__init__(config, embed)
        c = self.config
        self.prefix_token_proj = nn.Linear(c.target_hidden, c.prefix_width, bias=False)
        self.prefix_proj = nn.Linear(c.depth * c.prefix_width, c.width, bias=False)

    def correct(self, features, prefix_ids, prefix_mask):
        """Correct features[...,W] with ordered prefix ids/mask[...,D].

        Masking happens before embedding lookup and after shared projection.
        Consequently padded ids, including out-of-vocabulary sentinel values,
        cannot affect either output or gradients. The original previous-token
        path uses the last valid slot; no head can inspect a future token.
        """
        if prefix_ids.shape != prefix_mask.shape:
            raise ValueError('prefix ids and masks must have equal shapes')
        if prefix_ids.shape[:-1] != features.shape[:-1]:
            raise ValueError('prefix and feature leading dimensions must match')
        if prefix_ids.shape[-1] != self.config.depth:
            raise ValueError('prefix slots must equal configured child depth')
        if prefix_mask.dtype != torch.bool:
            raise ValueError('prefix_mask must be boolean')
        safe_ids = prefix_ids.masked_fill(~prefix_mask, 0)
        embedded = F.embedding(safe_ids, self.shared_embed)
        encoded = self.prefix_token_proj(embedded)
        encoded = encoded.masked_fill(~prefix_mask[..., None], 0)
        prefix = self.prefix_proj(encoded.flatten(-2))
        slots = torch.arange(1, self.config.depth + 1, device=prefix_ids.device)
        last = (prefix_mask * slots).amax(-1) - 1
        previous = safe_ids.gather(-1, last.clamp_min(0)[..., None]).squeeze(-1)
        prev = self.previous_proj(F.embedding(previous, self.shared_embed))
        prev = prev.masked_fill((last < 0)[..., None], 0)
        return self.output_norm(features + self.correction(
            self.correction_norm(features + prev + prefix)))

    @torch.no_grad()
    def deployment_correct(self, features, prefix_ids, prefix_mask):
        """Score search candidates with deployment-rounded correction weights.

        Training keeps FP32 master parameters, while serving converts them to
        the frozen Target embedding dtype (BF16). Cache just these small head
        weights until optimizer updates change parameter versions. The wrapper
        is deliberately not registered on the model, so checkpoint state keys
        and parameter traversal remain exactly those of the trainable model.
        """
        view = getattr(self, '_deployment_correct_view', None)
        if view is None or view.shared_embed is not self.shared_embed:
            view = _CorrectionHead(self)
            object.__setattr__(self, '_deployment_correct_view', view)
            object.__setattr__(self, '_deployment_correct_signature', None)
        parameters = dict(view.named_parameters())
        signature = tuple((name, id(p), p._version, p.device, p.dtype)
                          for name, p in parameters.items())
        signature += (('target_dtype', self.shared_embed.dtype),)
        if signature != self._deployment_correct_signature:
            weights = {name: p.detach().to(dtype=self.shared_embed.dtype)
                       for name, p in parameters.items()}
            object.__setattr__(self, '_deployment_correct_weights', weights)
            object.__setattr__(self, '_deployment_correct_signature', signature)
        with torch.autocast(device_type=features.device.type, enabled=False):
            return torch.func.functional_call(
                view, self._deployment_correct_weights,
                (features.to(self.shared_embed.dtype), prefix_ids, prefix_mask))

    def forward(self, taps, bonus, kv, indices, positions, previous):
        if previous.shape[-1] != self.config.depth:
            raise ValueError('previous must contain bonus and depth-1 gold children')
        features = self.backbone(taps, bonus, kv, indices, positions)
        ids, mask = teacher_prefixes(previous)
        return self.logits(self.correct(features, ids, mask))
