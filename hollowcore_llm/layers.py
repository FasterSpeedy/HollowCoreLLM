import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import HollowCoreConfig
from .deps import load_chunk_gdn2, load_flash_attn_func


class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        y = x.float()
        y = y * torch.rsqrt(y.pow(2).mean(dim=-1, keepdim=True) + self.eps)
        return y.to(x.dtype) * self.weight


def rotate_half(x: torch.Tensor) -> torch.Tensor:
    a, b = x[..., ::2], x[..., 1::2]
    return torch.stack((-b, a), dim=-1).flatten(-2)


def apply_scaled_rope(
    q: torch.Tensor,
    k: torch.Tensor,
    cfg: HollowCoreConfig,
    pos_offset: int = 0,
) -> tuple[torch.Tensor, torch.Tensor]:
    s = q.size(1)
    device = q.device
    dtype = q.dtype
    pos = (torch.arange(s, device=device, dtype=torch.float32) + pos_offset) / cfg.rope_scale
    inv = cfg.rope_base ** (-torch.arange(0, cfg.head_dim, 2, device=device, dtype=torch.float32) / cfg.head_dim)
    freqs = torch.einsum("s,d->sd", pos, inv)
    emb = torch.repeat_interleave(freqs, 2, dim=-1)[None, :, None, :]
    cos = emb.cos().to(dtype)
    sin = emb.sin().to(dtype)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class DeltaFlashMixer(nn.Module):
    def __init__(self, cfg: HollowCoreConfig, layer_idx: int) -> None:
        super().__init__()
        self.cfg = cfg
        h = cfg.hidden_size
        self.qkv = nn.Linear(h, 3 * h, bias=False)
        self.f_proj = nn.Linear(h, h, bias=False)
        self.b_proj = nn.Linear(h, h, bias=False)
        self.w_proj = nn.Linear(h, h, bias=False)
        self.local_gate = nn.Linear(h, h, bias=True)
        self.out = nn.Linear(h, h, bias=False)
        self.A_log = nn.Parameter(torch.log(torch.empty(cfg.num_heads).uniform_(1, 16)))
        dt = torch.exp(torch.rand(h) * (math.log(0.1) - math.log(0.001)) + math.log(0.001)).clamp(min=1e-4)
        self.dt_bias = nn.Parameter(dt + torch.log(-torch.expm1(-dt)))
        self.layer_idx = layer_idx

    def forward(
        self,
        x: torch.Tensor,
        gdn2_state=None,
        kv_halo=None,
        pos_offset: int = 0,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple | None]:
        cfg = self.cfg
        bsz, seq, _ = x.shape
        qkv = self.qkv(x).view(bsz, seq, 3, cfg.num_heads, cfg.head_dim)
        q, k, v = qkv.unbind(dim=2)

        q_local, k_local = apply_scaled_rope(q, k, cfg, pos_offset=pos_offset)  #1
        flash_attn_func = load_flash_attn_func()
        if kv_halo is not None:
            k_halo, v_halo = kv_halo
            k_cat = torch.cat([k_halo, k_local], dim=1)
            v_cat = torch.cat([v_halo, v], dim=1)
        else:
            k_cat, v_cat = k_local, v
        local = flash_attn_func(
            q_local.contiguous(),
            k_cat.contiguous(),
            v_cat.contiguous(),
            dropout_p=0.0,
            causal=True,
            window_size=(cfg.local_window - 1, 0),
        )  #2
        w = cfg.local_window - 1
        new_kv_halo = (k_local[:, -w:].detach(), v[:, -w:].detach())

        g = F.softplus(self.f_proj(x).float() + self.dt_bias).view(bsz, seq, cfg.num_heads, cfg.head_dim)
        g = -self.A_log.float().exp().view(1, 1, cfg.num_heads, 1) * g
        erase = self.b_proj(x).sigmoid().view(bsz, seq, cfg.num_heads, cfg.head_dim)
        write = self.w_proj(x).sigmoid().view(bsz, seq, cfg.num_heads, cfg.head_dim)

        chunk_gdn2 = load_chunk_gdn2()
        delta, new_gdn2_state = chunk_gdn2(
            q=q.contiguous(),
            k=k.contiguous(),
            v=v.contiguous(),
            g=g.contiguous(),
            b=erase.contiguous(),
            w=write.contiguous(),
            initial_state=gdn2_state,
            output_final_state=True,
            use_qk_l2norm_in_kernel=True,
            cu_seqlens=None,
        )  #3

        gate = self.local_gate(x).sigmoid().view(bsz, seq, cfg.num_heads, cfg.head_dim)
        mixed = delta + gate * local
        return self.out(mixed.reshape(bsz, seq, cfg.hidden_size)), new_gdn2_state, new_kv_halo


class SwiGLUExpert(nn.Module):
    def __init__(self, hidden_size: int, intermediate_size: int) -> None:
        super().__init__()
        self.gate = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down = nn.Linear(intermediate_size, hidden_size, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down(F.silu(self.gate(x)) * self.up(x))


class Top1MoE(nn.Module):
    def __init__(self, cfg: HollowCoreConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.router = nn.Linear(cfg.hidden_size, cfg.num_experts, bias=False)
        self.experts = nn.ModuleList(
            SwiGLUExpert(cfg.hidden_size, cfg.expert_intermediate_size)
            for _ in range(cfg.num_experts)
        )

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        bsz, seq, hidden = x.shape
        flat = x.reshape(-1, hidden)
        logits = self.router(flat)
        probs = F.softmax(logits.float(), dim=-1)
        weight, idx = probs.max(dim=-1)
        out = torch.zeros_like(flat)

        for expert_id, expert in enumerate(self.experts):
            mask = idx == expert_id
            if mask.any():
                out[mask] = expert(flat[mask]) * weight[mask].to(flat.dtype).unsqueeze(-1)

        dispatch = F.one_hot(idx, self.cfg.num_experts).float()
        aux = self.cfg.num_experts * (dispatch.mean(0) * probs.mean(0)).sum()
        z = torch.logsumexp(logits.float(), dim=-1).pow(2).mean()
        return out.view(bsz, seq, hidden), aux, z
