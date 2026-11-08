"""A frozen-target, bonus-native Hybrid drafter conditioned on loop states.

All modes use R * H source features and the same trainable projections. The
trajectory representation is [h_R, h_2-h_1, ..., h_R-h_(R-1)]. A single RMS
scale derived from h_R is applied to every slot; individual differences are
never normalized to unit length, so their relative magnitude is retained.
Canonical layouts place h4 first for every mode and initialize the other
three source projection blocks to zero. Legacy layouts retain pilot behavior.
"""
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import torch
from torch import nn
from torch.nn import functional as F

from scripts.fest_hybrid.model import HybridLayer, RMSNorm, SwiGLU
from scripts.fest_hybrid_prefix_tree.model import teacher_prefixes


@dataclass(frozen=True)
class Config:
    architecture: str = "ouro_loop_hybrid_no_kv_v1"
    target_hidden: int = 2048
    vocab_size: int = 49152
    rounds: int = 4
    feature_mode: str = "trajectory"
    source_layout: str = "legacy"
    width: int = 768
    mlp_width: int = 2304
    layers: int = 4
    init_rank: int = 192
    prefix_width: int = 128
    depth: int = 5
    state_dim: int = 64
    state_head_dim: int = 64
    state_expand: int = 2
    normalization_eps: float = 1e-6

    def __post_init__(self):
        if self.feature_mode not in ("final", "multiloop", "trajectory"):
            raise ValueError("feature_mode must be final, multiloop, or trajectory")
        if self.source_layout not in ("legacy", "canonical"):
            raise ValueError("source_layout must be legacy or canonical")
        if self.source_layout == "canonical" and self.rounds != 4:
            raise ValueError("canonical source layout requires exactly four rounds")
        for key in ("rounds", "depth", "width", "layers", "init_rank", "target_hidden",
                    "vocab_size", "state_dim", "state_head_dim", "state_expand"):
            if getattr(self, key) < 1:
                raise ValueError(f"{key} must be positive")
        if self.width * self.state_expand % self.state_head_dim:
            raise ValueError("expanded width must divide into state heads")

    def dict(self):
        return asdict(self)


class LoopDraft(nn.Module):
    def __init__(self, config: Config, embedding: torch.Tensor,
                 lm_head: torch.Tensor | None = None):
        super().__init__()
        self.config = c = config
        lm_head = embedding if lm_head is None else lm_head
        expected = (c.vocab_size, c.target_hidden)
        if tuple(embedding.shape) != expected or tuple(lm_head.shape) != expected:
            raise ValueError(f"frozen embedding/head must both have shape {expected}")
        # FP32 buffers support CPU tests and FP32 training; CUDA autocast handles
        # BF16 execution. These frozen target weights are omitted from checkpoints.
        tied = embedding.data_ptr() == lm_head.data_ptr()
        frozen_embedding = embedding.detach().float()
        self.register_buffer("shared_embed", frozen_embedding, persistent=False)
        self.register_buffer("shared_lm_head", frozen_embedding if tied else lm_head.detach().float(),
                             persistent=False)
        self.source_weight = nn.Parameter(torch.ones(c.target_hidden))
        self.source_rank = nn.Linear(c.rounds * c.target_hidden, c.init_rank, bias=False)
        if c.source_layout == "canonical":
            # The first slot always contains h4. Extra states begin with no
            # functional effect, but their projection blocks remain trainable.
            nn.init.zeros_(self.source_rank.weight[:, c.target_hidden:])
        self.state_proj = nn.Linear(c.init_rank, c.width * c.state_expand * c.state_dim, bias=False)
        self.token_proj = nn.Linear(c.target_hidden, c.width, bias=False)
        self.mask = nn.Parameter(torch.randn(c.width) * .02)
        # Reuse the existing initialized Mamba2/SSD and MLP without KV attention.
        self.layers = nn.ModuleList(HybridLayer(c, attention=False) for _ in range(c.layers))
        self.prefix_token_proj = nn.Linear(c.target_hidden, c.prefix_width, bias=False)
        self.prefix_proj = nn.Linear(c.depth * c.prefix_width, c.width, bias=False)
        self.previous_proj = nn.Linear(c.target_hidden, c.width, bias=False)
        self.correction_norm = RMSNorm(c.width)
        self.correction = SwiGLU(c.width, c.width)
        self.output_norm = RMSNorm(c.width)
        self.head_proj = nn.Linear(c.width, c.target_hidden, bias=False)

    def source_features(self, taps):
        c = self.config
        if taps.ndim != 3 or tuple(taps.shape[1:]) != (c.rounds, c.target_hidden):
            raise ValueError("taps must have shape [batch, rounds, target_hidden]")
        h = taps.float()
        final = h[:, -1:]
        scale = torch.rsqrt(final.square().mean(-1, keepdim=True) + c.normalization_eps)
        if c.source_layout == "canonical":
            if c.feature_mode == "final":
                source = torch.cat((final, torch.zeros_like(h[:, :-1])), dim=1)
            elif c.feature_mode == "multiloop":
                source = torch.cat((final, h[:, :-1]), dim=1)
            else:
                source = torch.cat((final, h[:, 1:] - h[:, :-1]), dim=1)
        else:
            # Keep old checkpoints' slot order and repeated-final behavior.
            if c.feature_mode == "final":
                source = final.expand(-1, c.rounds, -1)
            elif c.feature_mode == "multiloop":
                source = h
            else:
                source = torch.cat((final, h[:, 1:] - h[:, :-1]), dim=1)
        return source * scale * self.source_weight.float() / c.rounds ** .5

    def initial_state(self, taps):
        c = self.config
        source = self.source_features(taps).flatten(1).to(self.source_rank.weight.dtype)
        state = self.state_proj(F.silu(self.source_rank(source)))
        return state.view(len(taps), c.width * c.state_expand // c.state_head_dim,
                          c.state_head_dim, c.state_dim)

    def backbone(self, taps, bonus):
        if bonus.shape != (len(taps),):
            raise ValueError("bonus must have shape [batch]")
        state = self.initial_state(taps)
        bonus_embed = self.token_proj(F.embedding(bonus, self.shared_embed))[:, None]
        x = torch.cat((bonus_embed, self.mask[None, None].expand(len(taps), self.config.depth, -1)
                       .to(bonus_embed.dtype)), dim=1)
        for layer in self.layers:
            x = layer(x, state, None, None, None)
        return x[:, 1:]

    def correct(self, features, prefix_ids, prefix_mask):
        """Existing Hybrid full-prefix correction, with no future-token access."""
        if prefix_ids.shape != prefix_mask.shape or prefix_ids.shape[:-1] != features.shape[:-1]:
            raise ValueError("prefix and feature leading shapes must match")
        if prefix_ids.shape[-1] != self.config.depth or prefix_mask.dtype != torch.bool:
            raise ValueError("prefix must have depth slots and a boolean mask")
        safe_ids = prefix_ids.masked_fill(~prefix_mask, 0)
        embedded = F.embedding(safe_ids, self.shared_embed)
        encoded = self.prefix_token_proj(embedded).masked_fill(~prefix_mask[..., None], 0)
        prefix = self.prefix_proj(encoded.flatten(-2))
        slots = torch.arange(1, self.config.depth + 1, device=prefix_ids.device)
        last = (prefix_mask * slots).amax(-1) - 1
        previous = safe_ids.gather(-1, last.clamp_min(0)[..., None]).squeeze(-1)
        prev = self.previous_proj(F.embedding(previous, self.shared_embed))
        prev = prev.masked_fill((last < 0)[..., None], 0)
        return self.output_norm(features + self.correction(self.correction_norm(features + prev + prefix)))

    def logits(self, features):
        return F.linear(self.head_proj(features), self.shared_lm_head)

    def forward(self, taps, bonus, previous):
        if previous.shape != (len(taps), self.config.depth):
            raise ValueError("previous must be [batch, depth], starting with bonus")
        if not torch.equal(previous[:, 0], bonus):
            raise ValueError("the first prefix token must equal bonus")
        ids, mask = teacher_prefixes(previous)
        return self.logits(self.correct(self.backbone(taps, bonus), ids, mask))

    def step_logits(self, features, prefix_ids):
        """Score the next child from cached backbone features and its real prefix.

        prefix_ids[B,k] contains bonus followed by k-1 proposed children.
        This function does not rerun the backbone and can also score branches.
        """
        if prefix_ids.ndim != 2 or not 1 <= prefix_ids.shape[1] <= self.config.depth:
            raise ValueError("prefix_ids must contain bonus and at most depth-1 children")
        b, length = prefix_ids.shape
        if features.shape != (b, self.config.depth, self.config.width):
            raise ValueError("features must be the matching backbone output")
        ids = F.pad(prefix_ids, (0, self.config.depth - length))
        mask = torch.arange(self.config.depth, device=ids.device)[None].expand(b, -1) < length
        return self.logits(self.correct(features[:, length - 1], ids, mask))

    @torch.inference_mode()
    def generate_draft(self, taps, bonus, max_children=None):
        """Greedy children only, excluding bonus; target verification is external."""
        count = self.config.depth if max_children is None else int(max_children)
        if not 0 <= count <= self.config.depth:
            raise ValueError("max_children must be between zero and depth")
        if count == 0:
            return bonus.new_empty((len(bonus), 0))
        features = self.backbone(taps, bonus)
        prefix = bonus[:, None]
        for _ in range(count):
            child = self.step_logits(features, prefix).argmax(-1)
            prefix = torch.cat((prefix, child[:, None]), dim=1)
        return prefix[:, 1:]


def load_frozen_io(path):
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    embedding = payload["embedding"]
    return embedding, payload.get("lm_head", embedding)


def load_checkpoint(path, frozen_io_path=None, device="cpu"):
    """Return (model, payload); optimizer/RNG fields remain available for resume."""
    payload = torch.load(Path(path), map_location="cpu", weights_only=True)
    if payload.get("schema") != "loop_drafter_checkpoint_v1":
        raise ValueError("unsupported loop drafter checkpoint")
    io_path = frozen_io_path or payload["frozen_io"]
    embedding, lm_head = load_frozen_io(io_path)
    model = LoopDraft(Config(**payload["config"]), embedding, lm_head)
    model.load_state_dict(payload["model"], strict=True)
    return model.to(device), payload


def fork_from_final(checkpoint_path, feature_mode, frozen_io_path=None, device="cpu"):
    """Fork a canonical final-state checkpoint without changing its function.

    Every learned parameter is restored, including the final-state projection
    and its learned RMS weight. Only the extra source projection blocks are
    reset to zero. The returned payload describes the child and retains the
    parent cursor for provenance; callers must initialize a new training cursor
    and optimizer. Optimizer state is intentionally not inherited.
    """
    if feature_mode not in ("final", "multiloop", "trajectory"):
        raise ValueError("feature_mode must be final, multiloop, or trajectory")
    parent, parent_payload = load_checkpoint(checkpoint_path, frozen_io_path, device="cpu")
    if parent.config.source_layout != "canonical" or parent.config.feature_mode != "final":
        raise ValueError("fork_from_final requires a canonical final parent checkpoint")
    config = replace(parent.config, feature_mode=feature_mode)
    embedding = parent.shared_embed.detach().clone()
    tied = parent.shared_embed.data_ptr() == parent.shared_lm_head.data_ptr()
    lm_head = embedding if tied else parent.shared_lm_head.detach().clone()
    child = LoopDraft(config, embedding, lm_head)
    child.load_state_dict(parent.state_dict(), strict=True)
    with torch.no_grad():
        child.source_rank.weight[:, config.target_hidden:].zero_()
    if child.config.normalization_eps != parent.config.normalization_eps:
        raise RuntimeError("Fork changed the source normalization epsilon")
    if not torch.equal(child.source_weight, parent.source_weight):
        raise RuntimeError("Fork changed the learned shared h4 normalization weight")
    if not torch.equal(child.source_rank.weight[:, :config.target_hidden],
                       parent.source_rank.weight[:, :config.target_hidden]):
        raise RuntimeError("Fork changed the baseline h4 source projection")
    parent_parameters = dict(parent.named_parameters())
    for name, parameter in child.named_parameters():
        if parameter.data_ptr() == parent_parameters[name].data_ptr():
            raise RuntimeError(f"Fork shares parameter storage with its parent: {name}")
    # Avoid copying a potentially large optimizer state. Other metadata records
    # the same data split and training context as the source checkpoint.
    payload = {key: value for key, value in parent_payload.items()
               if key not in ("model", "optimizer", "optimizer_state_dict", "cursor")}
    payload.update(
        config=config.dict(),
        model={name: value.detach().cpu().clone() for name, value in child.state_dict().items()},
        frozen_io=str(Path(frozen_io_path or parent_payload["frozen_io"]).resolve()),
        cursor=parent_payload.get("cursor", dict(step=0, epoch=0, next_request=0)),
        fork=dict(parent_checkpoint=str(Path(checkpoint_path).resolve()),
                  parent_config=parent.config.dict(), parent_cursor=parent_payload.get("cursor"),
                  optimizer_reset_required=True, extra_projection_zero=True),
    )
    return child.to(device), payload
