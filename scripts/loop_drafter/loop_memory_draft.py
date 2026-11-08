"""v1 SSD drafter with token-conditioned final/transition history retrieval.

History contains only confirmed, already processed target tokens. The current
bonus is not part of that memory. Token positions and transition IDs are separate
axes: all three transitions of a token have the same rotary position.
"""
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from .draft import Config, LoopDraft, load_checkpoint, load_frozen_io
from .train import make_batch
from scripts.fest_hybrid_prefix_tree.model import teacher_prefixes


SCHEMA = "loop_memory_drafter_checkpoint_v1"


@dataclass(frozen=True)
class LoopMemoryConfig(Config):
    architecture: str = "ouro_ssd_loop_memory_v1"
    feature_mode: str = "final"
    source_layout: str = "canonical"
    memory_mode: str = "trajectory"
    memory_width: int = 384
    memory_heads: int = 6
    window: int = 64
    memory_rope_theta: float = 10000.

    def __post_init__(self):
        super().__post_init__()
        if self.feature_mode != "final" or self.source_layout != "canonical" or self.rounds != 4:
            raise ValueError("v1 requires the canonical four-round final SSD backbone")
        if self.memory_mode not in ("none", "final", "final_capacity", "trajectory"):
            raise ValueError("unknown memory_mode")
        if min(self.memory_width, self.memory_heads, self.window) < 1:
            raise ValueError("memory dimensions must be positive")
        if self.memory_width % self.memory_heads or (self.memory_width // self.memory_heads) % 2:
            raise ValueError("memory heads need an even rotary head dimension")
        if self.memory_rope_theta <= 0:
            raise ValueError("rotary base must be positive")


@dataclass(frozen=True)
class LoopMemoryCache:
    # Keys are unrotated, so appending/sliding does not reproject old tokens.
    final: tuple[torch.Tensor, torch.Tensor] | None
    transition: tuple[torch.Tensor, torch.Tensor] | None
    mask: torch.Tensor                         # [B,W]
    positions: torch.Tensor                    # [B,W], same for all rounds
    next_position: torch.Tensor                # [B], position of the bonus


class TransitionEncoder(nn.Module):
    """One nonlinear E shared by all three transition IDs and both controls."""
    def __init__(self, hidden, width):
        super().__init__()
        self.state = nn.Linear(hidden, width, bias=False)
        self.delta = nn.Linear(hidden, width, bias=False)
        self.round_embedding = nn.Parameter(torch.randn(3, width) * .02)
        self.mix = nn.Linear(width, width, bias=False)

    def forward(self, state, delta):
        value = (self.state(state.to(self.state.weight.dtype))
                 + self.delta(delta.to(self.delta.weight.dtype)) + self.round_embedding)
        return self.mix(F.silu(value))


class MemoryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        c = config
        self.heads, self.head_dim = c.memory_heads, c.memory_width // c.memory_heads
        self.query = nn.Linear(c.width, c.memory_width, bias=False)
        self.key = nn.Linear(c.memory_width, c.memory_width, bias=False)
        self.value = nn.Linear(c.memory_width, c.memory_width, bias=False)
        self.output = nn.Linear(c.memory_width, c.width, bias=False)
        # No additional zero gate: output receives a gradient on the first step.
        nn.init.zeros_(self.output.weight)
        inverse = c.memory_rope_theta ** (-torch.arange(0, self.head_dim, 2).float() / self.head_dim)
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def project(self, encoded):
        # encoded is [B,W,M] or [B,W,3,M]; preserve the token axis for append.
        shape = (*encoded.shape[:-1], self.heads, self.head_dim)
        return tuple(layer(encoded).reshape(shape).movedim(-2, 1) for layer in (self.key, self.value))

    def rotate(self, value, positions):
        angles = positions.float()[..., None] * self.inverse_frequency.float()
        angles = torch.cat((angles, angles), -1)[:, None]
        half = self.head_dim // 2
        rotated = torch.cat((-value[..., half:], value[..., :half]), -1)
        return (value.float() * angles.cos() + rotated.float() * angles.sin()).to(value.dtype)

    def forward(self, query, query_positions, memory, positions, mask):
        batch, length, _ = query.shape
        q = self.query(query).view(batch, length, self.heads, self.head_dim).transpose(1, 2)
        k, v = memory
        if k.ndim == 5:
            k, v = k.flatten(2, 3), v.flatten(2, 3)
            positions, mask = positions.repeat_interleave(3, 1), mask.repeat_interleave(3, 1)
        q, k = self.rotate(q, query_positions), self.rotate(k.to(q.dtype), positions)
        attended = F.scaled_dot_product_attention(q, k, v.to(q.dtype),
                                                  attn_mask=mask[:, None, None, :], dropout_p=0.)
        return self.output(attended.transpose(1, 2).reshape(batch, length, -1))


class LoopMemoryDraft(LoopDraft):
    def __init__(self, config, embedding, lm_head=None):
        super().__init__(config, embedding, lm_head)
        c = config
        # Allocate the same modules in every mode for seeded, capacity controls.
        # Disabled branches are inactive; equal allocation does not imply equal
        # effective capacity for the none/final ablations.
        self.memory_final_encoder = nn.Linear(c.target_hidden, c.memory_width, bias=False)
        self.memory_transition_encoder = TransitionEncoder(c.target_hidden, c.memory_width)
        self.memory_final_attention = MemoryAttention(c)
        self.memory_transition_attention = MemoryAttention(c)
        if self.memory_parameter_count() > 5_000_000:
            raise ValueError("v1 memory modules exceed the five-million-parameter budget")

    def memory_parameter_count(self):
        return sum(p.numel() for name, p in self.named_parameters() if name.startswith("memory_"))

    def normalized_history(self, history, mask):
        c = self.config
        if history.ndim != 4 or tuple(history.shape[2:]) != (c.rounds, c.target_hidden):
            raise ValueError("history must be [B,W,4,H]")
        if not history.is_floating_point() or not 1 <= history.shape[1] <= c.window:
            raise ValueError("history must be floating and fit the configured window")
        if mask.shape != history.shape[:2] or mask.dtype != torch.bool or mask.device != history.device:
            raise ValueError("history_mask must be a matching boolean tensor")
        if not bool(mask.any(1).all()):
            raise ValueError("every request needs at least one confirmed token")
        clean = history.float().masked_fill(~mask[..., None, None], 0)
        scale = torch.rsqrt(clean[:, :, -1:].square().mean(-1, keepdim=True) + c.normalization_eps)
        final = clean[:, :, -1] * scale[:, :, 0]
        if c.memory_mode == "final_capacity":
            state = final[:, :, None].expand(-1, -1, 3, -1)
            delta = torch.zeros_like(state)
        else:
            state = clean[:, :, :-1] * scale
            delta = (clean[:, :, 1:] - clean[:, :, :-1]) * scale
        return final, state, delta

    def _project_history(self, history, mask):
        final, state, delta = self.normalized_history(history, mask)
        final_kv = transition_kv = None
        if self.config.memory_mode != "none":
            encoded = self.memory_final_encoder(final.to(self.memory_final_encoder.weight.dtype))
            final_kv = self.memory_final_attention.project(encoded)
        if self.config.memory_mode in ("final_capacity", "trajectory"):
            encoded = self.memory_transition_encoder(state, delta)
            transition_kv = self.memory_transition_attention.project(encoded)
        return final_kv, transition_kv

    def prepare_memory(self, history, history_mask):
        final, transition = self._project_history(history, history_mask)
        positions = (history_mask.long().cumsum(1) - 1).clamp_min(0)
        return LoopMemoryCache(final, transition, history_mask, positions, history_mask.sum(1))

    def append_memory(self, memory, confirmed_round_hidden):
        """Append only processed, accepted target states; never speculative suffixes.

        The caller crops the target KV cache independently. Passing a verifier's
        full output here before acceptance filtering would leak rejected tokens.
        """
        count = confirmed_round_hidden.shape[1]
        if count == 0:
            return memory
        if confirmed_round_hidden.shape[0] != memory.mask.shape[0]:
            raise ValueError("memory and appended history batch sizes differ")
        # A caller can append a prefill longer than the window; old slots then
        # disappear while absolute positions still advance by the full count.
        new = confirmed_round_hidden[:, -self.config.window:]
        mask = torch.ones(new.shape[:2], device=new.device, dtype=torch.bool)
        final, transition = self._project_history(new, mask)
        positions = memory.next_position[:, None] + torch.arange(count - new.shape[1], count, device=new.device)
        def join(old, fresh):
            if old is None:
                return None
            return tuple(torch.cat((a, b), dim=2)[:, :, -self.config.window:] for a, b in zip(old, fresh))
        return LoopMemoryCache(join(memory.final, final), join(memory.transition, transition),
                               torch.cat((memory.mask, mask), 1)[:, -self.config.window:],
                               torch.cat((memory.positions, positions), 1)[:, -self.config.window:],
                               memory.next_position + count)

    def correct_with_memory(self, features, prefix_ids, prefix_mask, memory, query_offset=0):
        if self.config.memory_mode == "none":
            return super().correct(features, prefix_ids, prefix_mask)
        if memory is None:
            raise ValueError("active memory mode requires a prepared memory cache")
        # Preserve the original correction exactly, adding context only to its
        # input. The query includes all actual/gold earlier children, never y_j.
        safe_ids = prefix_ids.masked_fill(~prefix_mask, 0)
        embedded = F.embedding(safe_ids, self.shared_embed)
        encoded = self.prefix_token_proj(embedded).masked_fill(~prefix_mask[..., None], 0)
        prefix = self.prefix_proj(encoded.flatten(-2))
        slots = torch.arange(1, self.config.depth + 1, device=prefix_ids.device)
        last = (prefix_mask * slots).amax(-1) - 1
        previous = safe_ids.gather(-1, last.clamp_min(0)[..., None]).squeeze(-1)
        prev = self.previous_proj(F.embedding(previous, self.shared_embed))
        prev = prev.masked_fill((last < 0)[..., None], 0)
        original = features + prev + prefix
        single = original.ndim == 2
        query = self.correction_norm(original[:, None] if single else original)
        positions = memory.next_position[:, None] + query_offset + torch.arange(query.shape[1], device=query.device)
        context = self.memory_final_attention(query, positions, memory.final, memory.positions, memory.mask)
        if memory.transition is not None:
            context = context + self.memory_transition_attention(query, positions, memory.transition,
                                                                 memory.positions, memory.mask)
        if single:
            context = context[:, 0]
        return self.output_norm(features + self.correction(self.correction_norm(original + context)))

    def forward(self, anchor_features, bonus, history, history_mask, teacher_tokens):
        if teacher_tokens.shape != (len(anchor_features), self.config.depth) or not torch.equal(teacher_tokens[:, 0], bonus):
            raise ValueError("teacher_tokens must be previous[B,D], starting with bonus")
        memory = self.prepare_memory(history, history_mask)
        ids, mask = teacher_prefixes(teacher_tokens)
        features = self.backbone(anchor_features, bonus)
        return self.logits(self.correct_with_memory(features, ids, mask, memory))

    def forward_batch(self, batch):
        return self(batch["taps"], batch["bonus"], batch["history"], batch["history_mask"], batch["previous"])

    def step_logits(self, features, prefix_ids, memory=None):
        if prefix_ids.ndim != 2 or not 1 <= prefix_ids.shape[1] <= self.config.depth:
            raise ValueError("prefix must contain bonus and at most depth-1 children")
        if features.shape != (len(prefix_ids), self.config.depth, self.config.width):
            raise ValueError("backbone features have the wrong shape")
        length = prefix_ids.shape[1]
        ids = F.pad(prefix_ids, (0, self.config.depth - length))
        mask = torch.arange(self.config.depth, device=ids.device)[None].expand(len(ids), -1) < length
        return self.logits(self.correct_with_memory(features[:, length - 1], ids, mask, memory, length - 1))

    @torch.inference_mode()
    def generate_draft(self, anchor_features, bonus, history=None, history_mask=None, max_children=None, *, memory=None):
        count = self.config.depth if max_children is None else int(max_children)
        if not 0 <= count <= self.config.depth:
            raise ValueError("max_children must be within the configured depth")
        if count == 0:
            return bonus.new_empty((len(bonus), 0))
        if memory is None and self.config.memory_mode != "none":
            if history is None or history_mask is None:
                raise ValueError("provide history/mask or prepared memory")
            memory = self.prepare_memory(history, history_mask)
        features = self.backbone(anchor_features, bonus)
        prefix = bonus[:, None]
        for _ in range(count):
            child = self.step_logits(features, prefix, memory).argmax(-1)
            prefix = torch.cat((prefix, child[:, None]), 1)
        return prefix[:, 1:]


def make_loop_memory_batch(record, roots, depth, window, device="cpu"):
    if window < 1:
        raise ValueError("window must be positive")
    batch = make_batch(record, roots, depth, device="cpu")
    roots = torch.as_tensor(roots, dtype=torch.long)
    columns = roots[:, None] - window + 1 + torch.arange(window)[None]
    mask = columns >= 0
    history = record["round_hidden"][columns.clamp_min(0)].masked_fill(~mask[..., None, None], 0)
    batch.update(history=history, history_mask=mask)
    return {key: value.to(device) for key, value in batch.items()}


def load_loop_memory_checkpoint(path, frozen_io_path=None, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != SCHEMA:
        raise ValueError("unsupported loop memory drafter checkpoint")
    embedding, head = load_frozen_io(frozen_io_path or payload["frozen_io"])
    model = LoopMemoryDraft(LoopMemoryConfig(**payload["config"]), embedding, head)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def fork_from_ssd(path, memory_mode, frozen_io_path=None, device="cpu", seed=0,
                  *, memory_width=384, memory_heads=6, window=64):
    parent, parent_payload = load_checkpoint(path, frozen_io_path, device="cpu")
    if parent.config.feature_mode != "final" or parent.config.source_layout != "canonical":
        raise ValueError("warmstart requires a canonical final SSD checkpoint")
    config = LoopMemoryConfig(**(parent.config.dict() | dict(
        architecture="ouro_ssd_loop_memory_v1", memory_mode=memory_mode,
        memory_width=memory_width, memory_heads=memory_heads, window=window)))
    tied = parent.shared_embed.data_ptr() == parent.shared_lm_head.data_ptr()
    embedding = parent.shared_embed.detach().clone()
    head = embedding if tied else parent.shared_lm_head.detach().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        model = LoopMemoryDraft(config, embedding, head)
    incompatible = model.load_state_dict(parent.state_dict(), strict=False)
    if incompatible.unexpected_keys or any(not key.startswith("memory_") for key in incompatible.missing_keys):
        raise RuntimeError("SSD warmstart did not preserve the entire backbone")
    payload = {key: value for key, value in parent_payload.items() if key not in ("optimizer", "model")}
    payload.update(schema=SCHEMA, config=config.dict(),
                   model={key: value.detach().clone() for key, value in model.state_dict().items()},
                   frozen_io=str(Path(frozen_io_path or parent_payload["frozen_io"]).resolve()),
                   fork=dict(parent_checkpoint=str(Path(path).resolve()), parent_cursor=parent_payload.get("cursor"),
                             seed=seed, memory_parameters=model.memory_parameter_count()))
    return model.to(device), payload
