"""Parallel future queries over the committed final-round hidden-state history.

The memory ends at processed position s. Its known bonus is token s+1, and
query d predicts token s+2+d. Queries contain no gold or predicted children.
Both modes have identical trainable modules; ``history_residual`` only adds
the latest original h4 vector before the frozen target vocabulary projection.
"""
from dataclasses import asdict, dataclass
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.fest_hybrid.model import RMSNorm, SwiGLU
from .draft import load_frozen_io
from .train import make_batch


@dataclass(frozen=True)
class HistoryConfig:
    architecture: str = "loop_history_causal_queries_v1"
    feature_mode: str = "history"
    target_hidden: int = 2048
    vocab_size: int = 49152
    rounds: int = 4
    width: int = 512
    mlp_width: int = 1536
    layers: int = 2
    heads: int = 8
    window: int = 64
    depth: int = 5
    normalization_eps: float = 1e-6
    rope_theta: float = 10000.

    def __post_init__(self):
        if self.feature_mode not in ("history", "history_residual"):
            raise ValueError("feature_mode must be history or history_residual")
        for name in ("target_hidden", "vocab_size", "rounds", "width", "mlp_width",
                     "layers", "heads", "window", "depth"):
            if getattr(self, name) < 1:
                raise ValueError(f"{name} must be positive")
        if self.width % self.heads or (self.width // self.heads) % 2:
            raise ValueError("width must divide into attention heads of even dimension")
        if self.normalization_eps <= 0 or self.rope_theta <= 0:
            raise ValueError("normalization epsilon and RoPE base must be positive")

    def dict(self):
        return asdict(self)


class HistoryAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.heads = config.heads
        self.head_dim = config.width // config.heads
        self.qkv = nn.Linear(config.width, 3 * config.width, bias=False)
        self.out = nn.Linear(config.width, config.width, bias=False)
        inverse = config.rope_theta ** (-torch.arange(0, self.head_dim, 2, dtype=torch.float32) / self.head_dim)
        self.register_buffer("inverse_frequency", inverse, persistent=False)

    def rotate(self, value, positions):
        angles = positions.float()[..., None] * self.inverse_frequency.float()
        angles = torch.cat((angles, angles), dim=-1)[:, None]
        half = self.head_dim // 2
        rotated = torch.cat((-value[..., half:], value[..., :half]), dim=-1)
        return (value.float() * angles.cos() + rotated.float() * angles.sin()).to(value.dtype)

    def forward(self, hidden, positions, allowed):
        batch, length, width = hidden.shape
        packed = self.qkv(hidden).view(batch, length, 3, self.heads, self.head_dim)
        q, k, v = packed.unbind(dim=2)
        q, k, v = q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
        q, k = self.rotate(q, positions), self.rotate(k, positions)
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=allowed[:, None], dropout_p=0.)
        return self.out(attended.transpose(1, 2).reshape(batch, length, width))


class HistoryLayer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.attention_norm = RMSNorm(config.width, config.normalization_eps)
        self.attention = HistoryAttention(config)
        self.ffn_norm = RMSNorm(config.width, config.normalization_eps)
        self.ffn = SwiGLU(config.width, config.mlp_width)

    def forward(self, hidden, positions, allowed, valid):
        hidden = hidden + self.attention(self.attention_norm(hidden), positions, allowed)
        hidden = hidden + self.ffn(self.ffn_norm(hidden))
        return hidden.masked_fill(~valid[..., None], 0)


class HistoryDraft(nn.Module):
    def __init__(self, config: HistoryConfig, embedding: torch.Tensor,
                 lm_head: torch.Tensor | None = None):
        super().__init__()
        self.config = c = config
        lm_head = embedding if lm_head is None else lm_head
        expected = (c.vocab_size, c.target_hidden)
        if tuple(embedding.shape) != expected or tuple(lm_head.shape) != expected:
            raise ValueError(f"frozen embedding/head must both have shape {expected}")
        tied = embedding.data_ptr() == lm_head.data_ptr()
        frozen_embedding = embedding.detach().float()
        self.register_buffer("shared_embed", frozen_embedding, persistent=False)
        self.register_buffer("shared_lm_head", frozen_embedding if tied else lm_head.detach().float(),
                             persistent=False)
        self.context_proj = nn.Linear(c.target_hidden, c.width, bias=False)
        self.bonus_proj = nn.Linear(c.target_hidden, c.width, bias=False)
        self.offset_embedding = nn.Parameter(torch.randn(c.depth, c.width) * .02)
        self.layers = nn.ModuleList(HistoryLayer(c) for _ in range(c.layers))
        self.output_norm = RMSNorm(c.width, c.normalization_eps)
        self.head_proj = nn.Linear(c.width, c.target_hidden, bias=False)

    def prepare_queries(self, history, history_mask, bonus):
        """Construct causal history/query inputs without reading future tokens."""
        c = self.config
        if history.ndim != 3 or history.shape[2] != c.target_hidden or not history.is_floating_point():
            raise ValueError("history must be floating [batch, window, target_hidden]")
        batch, length, _ = history.shape
        if batch < 1 or not 1 <= length <= c.window:
            raise ValueError("history batch must be nonempty and its length within config.window")
        if history_mask.shape != history.shape[:2] or history_mask.dtype != torch.bool:
            raise ValueError("history_mask must be bool[batch, window] with True for valid memory")
        if history_mask.device != history.device or bonus.device != history.device:
            raise ValueError("history, history_mask and bonus must be on the same device")
        if bonus.shape != (batch,) or bonus.dtype != torch.long:
            raise ValueError("bonus must be int64[batch]")
        if not bool(history_mask.any(dim=1).all()):
            raise ValueError("every request needs at least one valid historical position")
        # Clear invalid memory before projection: arbitrary values or NaNs in
        # padding cannot enter attention or the residual branch.
        memory = history.masked_fill(~history_mask[..., None], 0)
        slots = torch.arange(1, length + 1, device=history.device)
        latest_index = (history_mask.long() * slots).amax(dim=1) - 1
        latest_h4 = memory[torch.arange(batch, device=history.device), latest_index]
        context = self.context_proj(memory.to(self.context_proj.weight.dtype))
        latest_context = context[torch.arange(batch, device=history.device), latest_index]
        bonus_context = self.bonus_proj(F.embedding(bonus, self.shared_embed))
        queries = (bonus_context[:, None] + latest_context[:, None]
                   + self.offset_embedding[None].to(context.dtype))
        hidden = torch.cat((context, queries), dim=1)
        valid = torch.cat((history_mask, torch.ones(batch, c.depth, device=history.device, dtype=torch.bool)), dim=1)
        # Relative positions are independent of left-padding length. Query 0
        # occupies the bonus position and predicts its following child token.
        memory_positions = (history_mask.long().cumsum(dim=1) - 1).clamp_min(0)
        query_positions = history_mask.sum(dim=1, keepdim=True) + torch.arange(c.depth, device=history.device)[None]
        positions = torch.cat((memory_positions, query_positions), dim=1)
        total = length + c.depth
        causal = torch.ones(total, total, device=history.device, dtype=torch.bool).tril()
        allowed = causal[None] & valid[:, None, :]
        # Padded query rows would otherwise have no allowed keys. Their isolated
        # self-edge avoids all-masked softmax rows; valid queries cannot read them.
        allowed = allowed | ((~valid)[..., None] & torch.eye(total, device=history.device, dtype=torch.bool)[None])
        return hidden, positions, allowed, valid, latest_h4

    def predict_hidden(self, history, history_mask, bonus):
        hidden, positions, allowed, valid, latest_h4 = self.prepare_queries(history, history_mask, bonus)
        for layer in self.layers:
            hidden = layer(hidden, positions, allowed, valid)
        predicted = self.head_proj(self.output_norm(hidden[:, -self.config.depth:]))
        if self.config.feature_mode == "history_residual":
            predicted = predicted + latest_h4[:, None].to(predicted.dtype)
        return predicted

    def forward(self, history, history_mask, bonus):
        return F.linear(self.predict_hidden(history, history_mask, bonus), self.shared_lm_head)

    @torch.inference_mode()
    def generate_draft(self, history, history_mask, bonus, max_children=None):
        count = self.config.depth if max_children is None else int(max_children)
        if not 0 <= count <= self.config.depth:
            raise ValueError("max_children must be between zero and depth")
        if count == 0:
            return bonus.new_empty((len(bonus), 0))
        return self(history, history_mask, bonus)[:, :count].argmax(dim=-1)


def make_history_batch(record, roots, depth, window, device="cpu"):
    """Select only t <= anchor from h4; labels retain the bonus-native contract."""
    if window < 1:
        raise ValueError("history window must be positive")
    # Reuse the existing checked temporal contract, building it on CPU to avoid
    # transferring unused multi-round taps or teacher-prefix IDs to the GPU.
    base = make_batch(record, roots, depth, device="cpu")
    roots = torch.as_tensor(roots, dtype=torch.long)
    columns = roots[:, None] - window + 1 + torch.arange(window)[None]
    valid_memory = columns >= 0
    # Selecting round -1 is a view of the memory-mapped request, not a full copy.
    # Advanced indexing materializes only this microbatch's bounded windows.
    h4 = record["round_hidden"][:, -1]
    history = h4[columns.clamp_min(0)].masked_fill(~valid_memory[..., None], 0)
    return dict(history=history.to(device), history_mask=valid_memory.to(device),
                bonus=base["bonus"].to(device), labels=base["labels"].to(device), valid=base["valid"].to(device))


def load_history_checkpoint(path, frozen_io_path=None, device="cpu"):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != "loop_history_drafter_checkpoint_v1":
        raise ValueError("unsupported history drafter checkpoint")
    embedding, lm_head = load_frozen_io(frozen_io_path or payload["frozen_io"])
    model = HistoryDraft(HistoryConfig(**payload["config"]), embedding, lm_head)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload
