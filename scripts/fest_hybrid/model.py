"""H-Spec-inspired hybrid backbone with a bonus-native input contract.

The source taps and Target KV end BEFORE the known bonus. A block consists
of [bonus, mask, ..., mask]. All child features are computed together. Only
the small Markov correction depends on the immediately preceding candidate.

The short-block SSD expression is exactly the selective Mamba-2 recurrence,
including a differentiable initial state. It evaluates all positions in
parallel, without a runtime dependency on a version-specific CUDA extension.
The independent preflight compares it with the installed Mamba scan kernel.
"""
from dataclasses import asdict, dataclass
import math

import torch
from torch import nn
from torch.nn import functional as F


@dataclass(frozen=True)
class Config:
    architecture: str = 'fest_bonus_hybrid_kv_v1'
    width: int = 512
    mlp_width: int = 1536
    layers: int = 4
    target_hidden: int = 2560
    vocab_size: int = 151936
    kv_layers: tuple = (8, 16, 24)
    target_taps: tuple = (1, 9, 17, 25, 33)
    q_heads: int = 32
    kv_heads: int = 8
    head_dim: int = 128
    state_dim: int = 64
    state_head_dim: int = 64
    state_expand: int = 2
    init_rank: int = 128
    depth: int = 5
    window: int = 2048
    visual_budget: int = 256
    rope_theta: float = 5000000.0
    init_state: bool = True
    read_target_kv: bool = True

    def dict(self):
        return asdict(self)


class RMSNorm(nn.Module):
    def __init__(self, width, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(width))
        self.eps = eps

    def forward(self, x):
        y = x.float() * torch.rsqrt(x.float().square().mean(-1, keepdim=True) + self.eps)
        return (y * self.weight.float()).to(x.dtype)


def selective_scan_parallel(x, dt, A, B, C, initial):
    """Exact SSD for short blocks. x[B,L,H,P], B/C[B,L,N], S[B,H,P,N]."""
    # Lower-triangular semiseparable matrix; mask BEFORE exp to avoid overflow.
    decay = (dt.float() * A.float()[None, None]).cumsum(1).transpose(1, 2)
    length = x.shape[1]
    causal = torch.ones(length, length, device=x.device, dtype=torch.bool).tril()
    differences = decay[..., :, None] - decay[..., None, :]
    transition = differences.masked_fill(~causal, -torch.inf).exp()
    bc = torch.matmul(C.float(), B.float().transpose(-1, -2))[:, None]
    coefficients = transition * bc * dt.float().transpose(1, 2)[:, :, None, :]
    injected = torch.einsum('bln,bhpn->bhlp', C.float(), initial.float())
    output = torch.matmul(coefficients, x.float().transpose(1, 2))
    output = output + decay.exp()[..., None] * injected
    return output.transpose(1, 2).to(x.dtype)


class InitializedMamba2(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.inner = c.width * c.state_expand
        self.head_dim, self.state_dim = c.state_head_dim, c.state_dim
        self.heads = self.inner // self.head_dim
        self.in_proj = nn.Linear(c.width, 2*self.inner + 2*c.state_dim + self.heads, bias=False)
        self.conv = nn.Conv1d(self.inner + 2*c.state_dim, self.inner + 2*c.state_dim,
                              4, padding=3, groups=self.inner + 2*c.state_dim)
        self.A_log = nn.Parameter(torch.empty(self.heads).uniform_(1., 16.).log())
        dt = torch.empty(self.heads).uniform_(math.log(.001), math.log(.1)).exp()
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.D = nn.Parameter(torch.ones(self.heads))
        self.norm = RMSNorm(self.inner, eps=1e-5)
        self.out_proj = nn.Linear(self.inner, c.width, bias=False)

    def forward(self, hidden, initial):
        b, length, _ = hidden.shape
        z, xbc, raw_dt = self.in_proj(hidden).split(
            [self.inner, self.inner + 2*self.state_dim, self.heads], dim=-1)
        xbc = F.silu(self.conv(xbc.transpose(1, 2))[..., :length].transpose(1, 2))
        x, B, C = xbc.split([self.inner, self.state_dim, self.state_dim], -1)
        x = x.reshape(b, length, self.heads, self.head_dim)
        dt = F.softplus(raw_dt.float() + self.dt_bias.float())
        y = selective_scan_parallel(x, dt, -self.A_log.float().exp(), B, C, initial)
        y = y + x * self.D.to(x.dtype)[None, None, :, None]
        # Mamba2 norm_before_gate=False, one normalization group.
        y = self.norm(y.reshape(b, length, self.inner) * F.silu(z))
        return self.out_proj(y)


def rotate_text_positions(x, positions, theta):
    """Generated text has equal T/H/W positions; interleaved mRoPE reduces to RoPE."""
    inv = theta ** (-torch.arange(0, x.shape[-1], 2, device=x.device, dtype=torch.float32) / x.shape[-1])
    angle = positions.float()[..., None] * inv
    cos, sin = torch.cat((angle.cos(), angle.cos()), -1)[:, None], torch.cat((angle.sin(), angle.sin()), -1)[:, None]
    half = x.shape[-1] // 2
    rotated = torch.cat((-x[..., half:], x[..., :half]), -1)
    return (x.float()*cos + rotated.float()*sin).to(x.dtype)


class ReusedKVAttention(nn.Module):
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.q_proj = nn.Linear(c.width, c.q_heads*c.head_dim, bias=False)
        self.k_proj = nn.Linear(c.width, c.kv_heads*c.head_dim, bias=False)
        self.v_proj = nn.Linear(c.width, c.kv_heads*c.head_dim, bias=False)
        self.q_norm = RMSNorm(c.head_dim)
        self.k_norm = RMSNorm(c.head_dim)
        self.out_proj = nn.Linear(c.q_heads*c.head_dim, c.width, bias=False)

    def forward(self, hidden, kv, indices, positions, paged=False):
        c = self.c
        b, length, _ = hidden.shape
        q = self.q_norm(self.q_proj(hidden).view(b,length,c.q_heads,c.head_dim)).transpose(1,2)
        k = self.k_norm(self.k_proj(hidden).view(b,length,c.kv_heads,c.head_dim)).transpose(1,2)
        v = self.v_proj(hidden).view(b,length,c.kv_heads,c.head_dim).transpose(1,2)
        q, k = rotate_text_positions(q,positions,c.rope_theta), rotate_text_positions(k,positions,c.rope_theta)
        if paged:
            from .paged_attention import attend
            out = attend(q, kv[0], kv[1], indices, k, v)
        else:
            # Only training gathers; inference reads the Target pool in place.
            safe = indices.clamp_min(0)
            pk = kv[0][safe].transpose(1,2)
            pv = kv[1][safe].transpose(1,2)
            keys, values = torch.cat((pk,k),2), torch.cat((pv,v),2)
            prefix_mask = (indices >= 0)[:,None,None,:].expand(b,1,length,-1)
            block_mask = torch.ones(length,length,device=hidden.device,dtype=torch.bool).tril()[None,None].expand(b,1,-1,-1)
            mask = torch.cat((prefix_mask,block_mask),-1)
            out = F.scaled_dot_product_attention(q,keys,values,attn_mask=mask,enable_gqa=True)
        return self.out_proj(out.transpose(1,2).reshape(b,length,-1))


class SwiGLU(nn.Module):
    def __init__(self, width, middle):
        super().__init__()
        self.up = nn.Linear(width, middle*2, bias=False)
        self.down = nn.Linear(middle, width, bias=False)

    def forward(self, x):
        gate, value = self.up(x).chunk(2,-1)
        return self.down(F.silu(gate)*value)


class HybridLayer(nn.Module):
    def __init__(self, c, attention):
        super().__init__()
        self.ssm_norm = RMSNorm(c.width)
        self.ssm = InitializedMamba2(c)
        self.attn_norm = RMSNorm(c.width) if attention else None
        self.attn = ReusedKVAttention(c) if attention else None
        self.ffn_norm = RMSNorm(c.width)
        self.ffn = SwiGLU(c.width,c.mlp_width)

    def forward(self, x, initial, kv, indices, positions, paged=False):
        x = x + self.ssm(self.ssm_norm(x),initial)
        if self.attn is not None:
            x = x + self.attn(self.attn_norm(x),kv,indices,positions,paged)
        return x + self.ffn(self.ffn_norm(x))


class HybridDraft(nn.Module):
    def __init__(self, config, embed):
        super().__init__()
        c = self.config = config
        self.register_buffer('shared_embed',embed.detach(),persistent=False)
        self.token_proj = nn.Linear(c.target_hidden,c.width,bias=False)
        self.mask = nn.Parameter(torch.randn(c.width)*.02)
        self.source_norm = RMSNorm(c.target_hidden)
        self.source_rank = nn.Linear(len(c.target_taps)*c.target_hidden,c.init_rank,bias=False)
        self.state_proj = nn.Linear(c.init_rank,c.width*c.state_expand*c.state_dim,bias=False)
        self.layers = nn.ModuleList([HybridLayer(c,i<len(c.kv_layers)) for i in range(c.layers)])
        self.output_norm = RMSNorm(c.width)
        self.previous_proj = nn.Linear(c.target_hidden,c.width,bias=False)
        self.correction_norm = RMSNorm(c.width)
        self.correction = SwiGLU(c.width,c.width)
        self.head_proj = nn.Linear(c.width,c.target_hidden,bias=False)
        self.register_buffer('merged_head',None,persistent=False)
        self.register_buffer('vocab_ids',None,persistent=False)

    def initial_state(self, taps):
        c = self.config
        source = self.source_norm(taps).flatten(1)
        initial = self.state_proj(F.silu(self.source_rank(source)))
        initial = initial.view(len(taps),c.width*c.state_expand//c.state_head_dim,c.state_head_dim,c.state_dim)
        return initial if c.init_state else initial*0

    def backbone(self, taps, bonus, kv, indices, positions, paged=False):
        c = self.config
        initial = self.initial_state(taps)
        bonus_embed = self.token_proj(F.embedding(bonus,self.shared_embed))[:,None]
        x = torch.cat((bonus_embed,self.mask[None,None].expand(len(taps),c.depth,-1).to(bonus_embed.dtype)),1)
        if not c.read_target_kv:
            indices = torch.full_like(indices,-1)
        for i,layer in enumerate(self.layers):
            x = layer(x,initial,kv[i] if i<len(kv) else None,indices,positions,paged)
        return x[:,1:]

    def correct(self, features, previous):
        prev = self.previous_proj(F.embedding(previous,self.shared_embed))
        return self.output_norm(features + self.correction(self.correction_norm(features+prev)))

    def logits(self, features):
        if self.merged_head is not None:
            return F.linear(features,self.merged_head)
        return F.linear(self.head_proj(features),self.shared_embed)

    def forward(self, taps, bonus, kv, indices, positions, previous):
        features = self.backbone(taps,bonus,kv,indices,positions)
        return self.logits(self.correct(features,previous))

    @torch.no_grad()
    def merge_head(self, vocab_ids):
        self.vocab_ids = vocab_ids.to(self.shared_embed.device)
        # FP32 one-time projection, cast to the deployment dtype.
        self.merged_head = (self.shared_embed[self.vocab_ids].float() @ self.head_proj.weight.float()).to(self.shared_embed.dtype)


def prefix_indices(source_lengths, visual_columns, config):
    """Logical positions: bounded visual rows outside the causal recent window."""
    c = config
    device = source_lengths.device
    recent = source_lengths[:,None] - c.window + torch.arange(c.window,device=device)[None]
    recent = recent.masked_fill(recent<0,-1)
    if c.visual_budget == 0:
        return recent
    visual = torch.full((c.visual_budget,),-1,device=device,dtype=torch.long)
    if visual_columns.numel():
        count = min(c.visual_budget,visual_columns.numel())
        selected = torch.div(torch.arange(count,device=device)*visual_columns.numel(),count,rounding_mode='floor')
        visual[:count] = visual_columns[selected]
    visual = visual[None].expand(len(source_lengths),-1)
    visual = visual.masked_fill((visual<0)|(visual>=source_lengths[:,None]-c.window),-1)
    return torch.cat((visual,recent),1)
