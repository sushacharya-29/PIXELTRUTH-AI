"""
Streams: vit / srm / dct / fft / noise / jpeg  (same as ForensicsAI)
Inference output: { prediction, confidence, uncertainty, stream_importance,
                    stream_reliability, stream_contradiction_strength }

VRAM: ≤ 3.5 GB @ batch_size=8 ✅
"""

import os
import math
import hmac
import hashlib
import pickle
import json
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Tuple, List, Dict, NamedTuple
from collections import defaultdict, deque
from dataclasses import dataclass, field
from datetime import datetime

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_CKPT_SECRET = b'spatialRobust_v2_signing_key_change_me'


# ── VRAM helper ────────────────────────────────────────────────────────────────

def _vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


# ── Drop-path ──────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or self.drop_prob == 0.0:
            return x
        keep  = 1.0 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.dim() - 1)
        mask  = torch.rand(shape, device=x.device) < keep
        return x * mask / keep


# ── Depthwise Sep Conv ─────────────────────────────────────────────────────────

class DepthwiseSepConv(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                            padding=kernel // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.gelu(self.bn(self.pw(self.dw(x))))


# ══════════════════════════════════════════════════════════════════════
# AR-1: Heteroscedastic + Evidential Uncertainty Head
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EvidentialOutput:
    """Structured output from HeteroscedasticEvidentialHead."""
    reliability:  torch.Tensor  # (B, 1)
    aleatoric:    torch.Tensor  # (B, 1)
    epistemic:    torch.Tensor  # (B, 1)
    calibration:  torch.Tensor  # (B, 1)
    effective:    torch.Tensor  # (B, 1)
    nig_loss:     torch.Tensor  # scalar


class HeteroscedasticEvidentialHead(nn.Module):
    """
    AR-1: Per-stream uncertainty-aware reliability head (single forward pass).

    Outputs: μ (reliability), log σ² (aleatoric), NIG params ν,α,β (epistemic),
    calibration surprise (OOD detection via EMA feature norm tracking).

    effective = reliability × exp(-(aleatoric + epistemic)) × (1 - calibration)
    """
    def __init__(self, in_dim: int, evidential_coeff: float = 0.01):
        super().__init__()
        hidden = max(32, in_dim // 4)
        self.evidential_coeff = evidential_coeff

        self.encoder = nn.Sequential(
            nn.Linear(in_dim, hidden, bias=False),
            nn.LayerNorm(hidden),
            nn.GELU(),
        )
        self.mu_head      = nn.Linear(hidden, 1)
        self.log_var_head = nn.Linear(hidden, 1)
        self.nu_head    = nn.Linear(hidden, 1)
        self.alpha_head = nn.Linear(hidden, 1)
        self.beta_head  = nn.Linear(hidden, 1)

        self.register_buffer('_feat_ema',  torch.zeros(1))
        self.register_buffer('_feat_ema2', torch.ones(1))
        self._ema_alpha = 0.01

        nn.init.zeros_(self.mu_head.weight)
        nn.init.constant_(self.mu_head.bias, 2.0)
        nn.init.zeros_(self.log_var_head.weight)
        nn.init.constant_(self.log_var_head.bias, -2.0)
        for h in [self.nu_head, self.alpha_head, self.beta_head]:
            nn.init.zeros_(h.weight)
            nn.init.constant_(h.bias, 0.5)

    def _update_calibration_ema(self, feat_norm: torch.Tensor):
        if not self.training:
            return
        with torch.no_grad():
            a  = self._ema_alpha
            m  = feat_norm.mean().detach()
            m2 = (feat_norm ** 2).mean().detach()
            self._feat_ema.copy_((1 - a) * self._feat_ema  + a * m)
            self._feat_ema2.copy_((1 - a) * self._feat_ema2 + a * m2)

    def _calibration_surprise(self, feat: torch.Tensor) -> torch.Tensor:
        feat_norm = feat.norm(dim=1, keepdim=True)
        self._update_calibration_ema(feat_norm)
        running_mean = self._feat_ema.clamp(min=1e-6)
        running_std  = (self._feat_ema2 - self._feat_ema ** 2).clamp(min=1e-8).sqrt()
        z = ((feat_norm - running_mean) / (running_std + 1e-6)).abs()
        return torch.sigmoid(z - 2.0)

    def _evidential_nig_loss(self, mu: torch.Tensor, log_var: torch.Tensor,
                              nu: torch.Tensor, alpha: torch.Tensor,
                              beta: torch.Tensor) -> torch.Tensor:
        var = torch.exp(log_var).clamp(min=1e-6)
        nll = 0.5 * (math.log(2 * math.pi) + log_var + (mu - mu.detach()) ** 2 / var)
        evidence = nu + alpha
        reg = evidence.clamp(min=0.0) * nll.abs()
        return self.evidential_coeff * reg.mean()

    def forward(self, x: torch.Tensor) -> EvidentialOutput:
        h = self.encoder(x)

        mu      = torch.sigmoid(self.mu_head(h))
        log_var = self.log_var_head(h).clamp(-6.0, 2.0)
        var     = torch.exp(log_var)

        nu    = F.softplus(self.nu_head(h))    + 0.01
        alpha = F.softplus(self.alpha_head(h)) + 1.01
        beta  = F.softplus(self.beta_head(h))  + 0.01
        epistemic = (beta / (nu * (alpha - 1.0))).clamp(0.0, 5.0)

        calibration = self._calibration_surprise(x)

        decay     = torch.exp(-(var + epistemic).clamp(0.0, 4.0))
        effective = mu * decay * (1.0 - calibration.clamp(0.0, 0.85))
        effective = effective.clamp(0.05, 1.0)

        nig_loss = self._evidential_nig_loss(mu, log_var, nu, alpha, beta)

        return EvidentialOutput(
            reliability=mu,
            aleatoric=var,
            epistemic=epistemic,
            calibration=calibration,
            effective=effective,
            nig_loss=nig_loss,
        )


# ══════════════════════════════════════════════════════════════════════
# FusionStats — richer diagnostics
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FusionStats:
    """Carries per-stream trust diagnostics out of the fusion module (all tensors detached)."""
    gates:        torch.Tensor
    reliability:  Dict[str, torch.Tensor]
    aleatoric:    Dict[str, torch.Tensor]
    epistemic:    Dict[str, torch.Tensor]
    calibration:  Dict[str, torch.Tensor]
    effective:    Dict[str, torch.Tensor]
    nig_losses:   Dict[str, torch.Tensor]
    stream_names: List[str]

    def mean_gates(self)       -> Dict[str, float]:
        return {n: self.gates[:, i].mean().item()
                for i, n in enumerate(self.stream_names)}
    def mean_reliability(self) -> Dict[str, float]:
        return {n: v.mean().item() for n, v in self.reliability.items()}
    def mean_aleatoric(self)   -> Dict[str, float]:
        return {n: v.mean().item() for n, v in self.aleatoric.items()}
    def mean_epistemic(self)   -> Dict[str, float]:
        return {n: v.mean().item() for n, v in self.epistemic.items()}
    def mean_calibration(self) -> Dict[str, float]:
        return {n: v.mean().item() for n, v in self.calibration.items()}
    def mean_effective(self)   -> Dict[str, float]:
        return {n: v.mean().item() for n, v in self.effective.items()}

    def total_nig_loss(self) -> torch.Tensor:
        losses = [v for v in self.nig_losses.values() if v.requires_grad or True]
        if not losses:
            return torch.tensor(0.0)
        return sum(losses)

    def uncertainty_summary(self) -> Dict[str, Dict[str, float]]:
        return {
            'gates':       self.mean_gates(),
            'reliability': self.mean_reliability(),
            'aleatoric':   self.mean_aleatoric(),
            'epistemic':   self.mean_epistemic(),
            'calibration': self.mean_calibration(),
            'effective':   self.mean_effective(),
        }


# ══════════════════════════════════════════════════════════════════════
# AR-2: Parity Gate — balanced semantic/forensic authority
# ══════════════════════════════════════════════════════════════════════

class ParityGate(nn.Module):
    """
    AR-2: Stream trust gate with balanced semantic-forensic authority.
    Forensic streams vote first (α ∈ [0.3,0.7], init 0.55 forensic-leaning).
    """
    def __init__(self, shared_dim: int, n_streams: int, context_dim: int = 128,
                 forensic_names: Optional[List[str]] = None):
        super().__init__()
        self.n_streams = n_streams

        self.mlp_forensic = nn.Sequential(
            nn.Linear(shared_dim, context_dim, bias=False),
            nn.GELU(),
            nn.Linear(context_dim, n_streams),
        )
        self.mlp_semantic = nn.Sequential(
            nn.Linear(shared_dim, context_dim, bias=False),
            nn.GELU(),
            nn.Linear(context_dim, n_streams),
        )
        self.alpha_raw = nn.Parameter(torch.tensor(0.25))

        nn.init.zeros_(self.mlp_forensic[-1].weight)
        nn.init.zeros_(self.mlp_forensic[-1].bias)
        nn.init.zeros_(self.mlp_semantic[-1].weight)
        nn.init.zeros_(self.mlp_semantic[-1].bias)

    @property
    def alpha(self) -> torch.Tensor:
        return 0.3 + 0.4 * torch.sigmoid(self.alpha_raw)

    @property
    def beta(self) -> torch.Tensor:
        return 1.0 - self.alpha

    def forward(self, vit_eff: torch.Tensor,
                forensic_eff: Dict[str, torch.Tensor],
                sparse_k: Optional[int] = None) -> torch.Tensor:
        if forensic_eff:
            forensic_ctx = torch.stack(
                list(forensic_eff.values()), dim=1).mean(dim=1)
        else:
            forensic_ctx = torch.zeros_like(vit_eff)

        f_vote = self.mlp_forensic(forensic_ctx)
        s_vote = self.mlp_semantic(vit_eff)

        alpha = self.alpha
        gate_logits = alpha * f_vote + (1.0 - alpha) * s_vote

        if sparse_k is not None and sparse_k < self.n_streams:
            return self._sparse_softmax(gate_logits, sparse_k)
        return torch.softmax(gate_logits, dim=1)

    def _sparse_softmax(self, logits: torch.Tensor, k: int) -> torch.Tensor:
        if self.training:
            soft = F.gumbel_softmax(logits, tau=0.5, hard=False)
            _, topk_idx = soft.topk(k, dim=1)
            hard = torch.zeros_like(soft).scatter_(1, topk_idx, 1.0)
            return hard - soft.detach() + soft
        else:
            probs = torch.softmax(logits, dim=1)
            _, topk_idx = probs.topk(k, dim=1)
            gates = torch.zeros_like(probs).scatter_(
                1, topk_idx, probs.gather(1, topk_idx))
            return gates / (gates.sum(1, keepdim=True) + 1e-8)


# ══════════════════════════════════════════════════════════════════════
# AR-4: Forensic Confidence Weighter
# ══════════════════════════════════════════════════════════════════════

class ForensicConfidenceWeighter(nn.Module):
    """AR-4: Inverse-variance forensic weighting with learnable global dampening."""
    def __init__(self, dim: int, n_forensic: int,
                 init_dampening: float = 0.7):
        super().__init__()
        self.n_forensic = n_forensic

        self.mu_proj    = nn.Linear(dim, dim, bias=False)
        self.sigma_proj = nn.Linear(dim, dim, bias=False)
        self.norm_mu    = nn.LayerNorm(dim)
        self.norm_sigma = nn.LayerNorm(dim)

        raw = math.log(init_dampening / (1.0 - init_dampening + 1e-8))
        self.dampening_raw = nn.Parameter(torch.tensor(raw))

        nn.init.orthogonal_(self.mu_proj.weight)
        nn.init.eye_(self.sigma_proj.weight)

    @property
    def dampening(self) -> torch.Tensor:
        return 0.1 + 0.9 * torch.sigmoid(self.dampening_raw)

    def forward(self, forensic_stack: torch.Tensor) -> torch.Tensor:
        B, N, D = forensic_stack.shape

        mu    = self.norm_mu(self.mu_proj(forensic_stack))
        sigma = F.softplus(self.norm_sigma(self.sigma_proj(forensic_stack))) + 1e-4

        inv_var = 1.0 / (sigma ** 2 + 1e-8)
        norm_w  = inv_var / (inv_var.sum(dim=1, keepdim=True) + 1e-8)
        weighted = mu * norm_w * N

        return weighted * self.dampening


# ══════════════════════════════════════════════════════════════════════
# AR-3: Iterative Semantic-Forensic Contradiction Reasoner (ISFCR)
# ══════════════════════════════════════════════════════════════════════

class IterativeSFContradictionReasoner(nn.Module):
    """AR-3: Iterative semantic ↔ forensic contradiction reasoning (n_iters=2)."""
    def __init__(self, dim: int, n_forensic: int,
                 n_iters: int = 2, dropout: float = 0.05):
        super().__init__()
        assert n_forensic > 0 and n_iters >= 1
        self.n_iters    = n_iters
        self.dim        = dim
        self.n_forensic = n_forensic

        n_heads = max(1, dim // 64)

        self.forensic_sa_norm = nn.LayerNorm(dim)
        self.forensic_sa = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True)

        self.cross_sf_norm_q  = nn.LayerNorm(dim)
        self.cross_sf_norm_kv = nn.LayerNorm(dim)
        self.cross_sf = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True)

        self.cross_fs_norm_q  = nn.LayerNorm(dim)
        self.cross_fs_norm_kv = nn.LayerNorm(dim)
        self.cross_fs = nn.MultiheadAttention(
            embed_dim=dim, num_heads=n_heads,
            dropout=dropout, batch_first=True)

        self.gru_cell = nn.GRUCell(input_size=dim, hidden_size=dim)
        self.gru_gate = nn.Sequential(
            nn.Linear(dim * 2, 1), nn.Sigmoid())

        self.ff_sem = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.ff_for = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2, bias=False),
            nn.GELU(),
            nn.Linear(dim * 2, dim, bias=False),
        )
        self.drop = nn.Dropout(dropout)

    def forward(self, vit_feat: torch.Tensor,
                forensic_stack: torch.Tensor
                ) -> Tuple[torch.Tensor, torch.Tensor]:
        B = vit_feat.shape[0]
        v = vit_feat
        f = forensic_stack

        h = torch.zeros(B, self.dim, device=vit_feat.device, dtype=vit_feat.dtype)

        for _ in range(self.n_iters):
            fn = self.forensic_sa_norm(f)
            f_sa, _ = self.forensic_sa(fn, fn, fn)
            f = f + self.drop(f_sa)

            q_sf  = self.cross_sf_norm_q(v).unsqueeze(1)
            kv_sf = self.cross_sf_norm_kv(f)
            ca_sf, _ = self.cross_sf(q_sf, kv_sf, kv_sf)
            contradiction_signal = self.drop(ca_sf.squeeze(1))

            h = self.gru_cell(contradiction_signal, h)

            gate_input = torch.cat([v, h], dim=1)
            gate       = self.gru_gate(gate_input)
            v = v + gate * h

            q_fs  = self.cross_fs_norm_q(f)
            kv_fs = self.cross_fs_norm_kv(v).unsqueeze(1)
            ca_fs, _ = self.cross_fs(q_fs, kv_fs, kv_fs)
            f = f + self.drop(ca_fs)

        vit_out      = v + self.ff_sem(v)
        forensic_out = f + self.ff_for(f)

        return vit_out, forensic_out


# ══════════════════════════════════════════════════════════════════════
# AR-2 + AR-3 + AR-4: Authenticity-Reasoning Gated Fusion
# ══════════════════════════════════════════════════════════════════════

class AuthenticityReasoningFusion(nn.Module):
    """Full authenticity reasoning fusion (AR-1 through AR-4)."""
    def __init__(self,
                 stream_dims:   Dict[str, int],
                 shared_dim:    int  = 256,
                 context_dim:   int  = 128,
                 n_iters:       int  = 2,
                 use_isfcr:     bool = True,
                 use_fcw:       bool = True,
                 sparse_k:      Optional[int] = None,
                 evidential_coeff: float = 0.01):
        super().__init__()
        assert 'vit' in stream_dims
        self.stream_names   = list(stream_dims.keys())
        self.shared_dim     = shared_dim
        self.out_dim        = shared_dim
        self.use_isfcr      = use_isfcr
        self.use_fcw        = use_fcw
        self.sparse_k       = sparse_k
        self.forensic_names = [n for n in self.stream_names if n != 'vit']
        n_forensic          = len(self.forensic_names)
        n_streams           = len(self.stream_names)

        self.align_mlps = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(dim, shared_dim, bias=False),
                nn.LayerNorm(shared_dim),
                nn.GELU(),
            ) for name, dim in stream_dims.items()
        })

        self.evid_heads = nn.ModuleDict({
            name: HeteroscedasticEvidentialHead(shared_dim, evidential_coeff)
            for name in self.stream_names
        })

        if use_isfcr and n_forensic > 0:
            self.isfcr = IterativeSFContradictionReasoner(
                dim=shared_dim, n_forensic=n_forensic, n_iters=n_iters)

        if use_fcw and n_forensic > 0:
            self.fcw = ForensicConfidenceWeighter(
                dim=shared_dim, n_forensic=n_forensic, init_dampening=0.7)

        self.parity_gate = ParityGate(
            shared_dim=shared_dim, n_streams=n_streams,
            context_dim=context_dim, forensic_names=self.forensic_names)

        self.out_norm = nn.LayerNorm(shared_dim)

        self._ema_contrib = {n: 1.0 / n_streams for n in self.stream_names}
        self._ema_alpha   = 0.05

    def _update_ema(self, gates: torch.Tensor):
        mg = gates.detach().mean(0)
        a  = self._ema_alpha
        for i, n in enumerate(self.stream_names):
            self._ema_contrib[n] = (1 - a) * self._ema_contrib[n] + a * mg[i].item()

    def forward(self, streams: Dict[str, torch.Tensor]
                ) -> Tuple[torch.Tensor, FusionStats]:
        assert set(streams.keys()) == set(self.stream_names)

        normed  = {n: F.normalize(streams[n].float(), dim=1, eps=1e-8)
                   for n in self.stream_names}
        aligned = {n: self.align_mlps[n](normed[n])
                   for n in self.stream_names}

        evid_outs: Dict[str, EvidentialOutput] = {}
        for n in self.stream_names:
            evid_outs[n] = self.evid_heads[n](aligned[n])

        feat_eff = {n: aligned[n] * evid_outs[n].effective
                    for n in self.stream_names}

        vit_eff = feat_eff['vit']
        if self.use_isfcr and self.forensic_names and hasattr(self, 'isfcr'):
            f_stack = torch.stack(
                [feat_eff[n] for n in self.forensic_names], dim=1)
            vit_refined, forensic_refined = self.isfcr(vit_eff, f_stack)
            attended = {'vit': vit_refined}
            for i, n in enumerate(self.forensic_names):
                attended[n] = forensic_refined[:, i, :]
        else:
            attended = feat_eff

        if self.use_fcw and self.forensic_names and hasattr(self, 'fcw'):
            f_attn_stack = torch.stack(
                [attended[n] for n in self.forensic_names], dim=1)
            f_weighted = self.fcw(f_attn_stack)
            for i, n in enumerate(self.forensic_names):
                attended[n] = f_weighted[:, i, :]

        forensic_eff_dict = {n: attended[n] for n in self.forensic_names}
        gates = self.parity_gate(
            attended['vit'], forensic_eff_dict, self.sparse_k)

        if self.training:
            self._update_ema(gates)

        stream_stack = torch.stack(
            [attended[n] for n in self.stream_names], dim=1)
        stream_stack = stream_stack.clamp(-10., 10.)
        fused = (stream_stack * gates.unsqueeze(2)).sum(dim=1)
        fused = torch.nan_to_num(fused, nan=0.0)
        fused = self.out_norm(fused)

        stats = FusionStats(
            gates       = gates.detach(),
            reliability = {n: evid_outs[n].reliability.squeeze(1).detach()
                           for n in self.stream_names},
            aleatoric   = {n: evid_outs[n].aleatoric.squeeze(1).detach()
                           for n in self.stream_names},
            epistemic   = {n: evid_outs[n].epistemic.squeeze(1).detach()
                           for n in self.stream_names},
            calibration = {n: evid_outs[n].calibration.squeeze(1).detach()
                           for n in self.stream_names},
            effective   = {n: evid_outs[n].effective.squeeze(1).detach()
                           for n in self.stream_names},
            nig_losses  = {n: evid_outs[n].nig_loss.detach()
                           for n in self.stream_names},
            stream_names= self.stream_names,
        )
        return fused, stats

    def get_gate_weights(self, streams: Dict[str, torch.Tensor]) -> Dict:
        with torch.no_grad():
            _, stats = self.forward(streams)
        return stats.uncertainty_summary()

    def get_low_contribution_streams(self, threshold: float = 0.03) -> List[str]:
        return [n for n in self.stream_names
                if self._ema_contrib[n] < threshold]

    @property
    def use_cross_attn(self) -> bool:
        return self.use_isfcr

    @property
    def reliability_heads(self) -> nn.ModuleDict:
        return self.evid_heads


# ══════════════════════════════════════════════════════════════════════
# Forensic Stream Modules (all retained from ForensicsAI, no changes)
# ══════════════════════════════════════════════════════════════════════

def _make_srm_kernels_3x3(n_filters: int = 9) -> torch.Tensor:
    srm3 = [
        [[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]],
        [[-1, 2,-1],[ 2,-4, 2],[-1, 2,-1]],
        [[ 1,-2, 1],[-2, 4,-2],[ 1,-2, 1]],
        [[ 0, 0, 0],[-1, 2,-1],[ 0, 0, 0]],
        [[ 0,-1, 0],[ 0, 2, 0],[ 0,-1, 0]],
        [[-1, 0, 1],[ 0, 0, 0],[ 1, 0,-1]],
        [[ 1, 0,-1],[ 0, 0, 0],[-1, 0, 1]],
        [[ 0, 1, 0],[ 1,-4, 1],[ 0, 1, 0]],
        [[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]],
    ]
    w = torch.zeros(n_filters, 3, 3, 3)
    for i, k in enumerate(srm3[:n_filters]):
        kf = torch.tensor(k, dtype=torch.float32) / 4.0
        for c in range(3): w[i, c] = kf
    if n_filters > len(srm3):
        nn.init.kaiming_normal_(w[len(srm3):], mode='fan_out')
    return w


def _make_srm_kernels_5x5(n_filters: int = 30) -> torch.Tensor:
    bank = [
        [[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],[0,0,1,0,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,-1,0,1,0],[0,0,0,0,0],[0,1,0,-1,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,1,0,-1,0],[0,0,0,0,0],[0,-1,0,1,0],[0,0,0,0,0]],
        [[0,0,0,0,0],[0,0,1,0,0],[0,-1,0,1,0],[0,0,-1,0,0],[0,0,0,0,0]],
        [[0,0,-1,0,0],[0,0,3,0,0],[-1,3,-8,3,-1],[0,0,3,0,0],[0,0,-1,0,0]],
        [[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]],
    ]
    weights = torch.zeros(n_filters, 3, 5, 5)
    for i, k in enumerate(bank[:min(len(bank), n_filters)]):
        kf = torch.tensor(k, dtype=torch.float32) / 4.0
        for c in range(3): weights[i, c] = kf
    if n_filters > len(bank):
        nn.init.kaiming_normal_(weights[len(bank):], mode='fan_out')
    return weights


def _make_srm_kernels_7x7(n_filters: int = 15) -> torch.Tensor:
    w = torch.zeros(n_filters, 3, 7, 7)
    center = torch.zeros(7, 7)
    for r in range(7):
        for c in range(7):
            dist = abs(r-3)+abs(c-3)
            if dist == 0:   center[r,c] = 24.0
            elif dist == 1: center[r,c] = -4.0
            elif dist == 2: center[r,c] = -1.0
    center /= 24.0
    for i in range(min(6, n_filters)):
        for ch in range(3): w[i,ch] = center * ((-1)**i)
    if n_filters > 6:
        nn.init.kaiming_normal_(w[6:], mode='fan_out')
    return w


class MultiScaleSRM(nn.Module):
    def __init__(self, out_ch: int = 64):
        super().__init__()
        self.srm3 = nn.Conv2d(3, 9,  3, padding=1, bias=False)
        self.srm5 = nn.Conv2d(3, 30, 5, padding=2, bias=False)
        self.srm7 = nn.Conv2d(3, 15, 7, padding=3, bias=False)
        with torch.no_grad():
            self.srm3.weight.copy_(_make_srm_kernels_3x3(9))
            self.srm5.weight.copy_(_make_srm_kernels_5x5(30))
            self.srm7.weight.copy_(_make_srm_kernels_7x7(15))
        self.fuse = nn.Sequential(
            nn.Conv2d(54, out_ch, 1, bias=False),
            nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fuse(torch.cat([
            torch.tanh(self.srm3(x)),
            torch.tanh(self.srm5(x)),
            torch.tanh(self.srm7(x))], dim=1))


class PixelFrequencyHead(nn.Module):
    def __init__(self, in_ch: int = 64, out_dim: int = 128):
        super().__init__()
        self._pixel_attn: Optional[torch.Tensor] = None
        self.score_conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32), nn.GELU(),
            nn.Conv2d(32, 1, 1))
        self.refine = nn.Sequential(
            DepthwiseSepConv(in_ch, in_ch, 3),
            DepthwiseSepConv(in_ch, out_dim, 3))
        self.norm = nn.LayerNorm(out_dim)

    def forward(self, srm_feat: torch.Tensor) -> torch.Tensor:
        B, C, H, W = srm_feat.shape
        score = self.score_conv(srm_feat)
        attn  = torch.softmax(score.view(B,1,-1), dim=-1).view(B,1,H,W)
        self._pixel_attn = attn.detach()
        return self.norm(self.refine(srm_feat * attn).mean(dim=[2,3]))


class FrequencyStream(nn.Module):
    _LAPLACIAN = torch.tensor(
        [[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]], dtype=torch.float32)

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim
        self._feat_l1 = self._feat_l2 = self._feat_l3 = None
        self.register_buffer('laplacian',
            self._LAPLACIAN.view(1,1,3,3).repeat(3,1,1,1))
        self.ms_srm     = MultiScaleSRM(out_ch=64)
        self.pixel_head = PixelFrequencyHead(in_ch=64, out_dim=out_dim//2)
        self.layer1 = DepthwiseSepConv(67, 64, 3, stride=2)
        self.layer2 = DepthwiseSepConv(64, 96, 3, stride=2)
        self.layer3 = DepthwiseSepConv(96, out_dim//2, 3, stride=2)
        self.norm_global = nn.LayerNorm(out_dim//2)
        self.fusion = nn.Sequential(
            nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))

    @property
    def _pixel_attn(self):
        return self.pixel_head._pixel_attn

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = (x * self._std + self._mean).clamp(0.,1.)
        lap   = F.conv2d(x_raw, self.laplacian, padding=1, groups=3).clamp(-1.,1.)
        srm_f = self.ms_srm(x_raw)
        p_out = self.pixel_head(srm_f)
        f = torch.cat([srm_f, lap], dim=1)
        f = self.layer1(f); self._feat_l1 = f
        f = self.layer2(f); self._feat_l2 = f
        f = self.layer3(f); self._feat_l3 = f
        g = self.norm_global(f.mean(dim=[2,3]))
        return self.fusion(torch.cat([p_out, g], dim=1))


class MultiScaleDCTBlock(nn.Module):
    _BLOCK_SIZES = [8, 16, 32]

    def __init__(self, proj_dim: int = 96):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(4032, 192), nn.LayerNorm(192), nn.GELU(),
            nn.Linear(192, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        for bs in self._BLOCK_SIZES:
            self.register_buffer(f'_dct_basis_{bs}', self._build_basis(bs))

    @staticmethod
    def _build_basis(N: int) -> torch.Tensor:
        n = torch.arange(N, dtype=torch.float32)
        k = torch.arange(N, dtype=torch.float32)
        return torch.cos(math.pi/N * (n.unsqueeze(1)+0.5) * k.unsqueeze(0))

    def _dct2_blocks(self, x: torch.Tensor, bs: int) -> torch.Tensor:
        B, C, H, W = x.shape
        ph = (bs - H%bs)%bs; pw = (bs - W%bs)%bs
        if ph>0 or pw>0: x = F.pad(x,(0,pw,0,ph))
        blocks = x.unfold(2,bs,bs).unfold(3,bs,bs).contiguous().view(B,-1,bs,bs)
        basis  = getattr(self, f'_dct_basis_{bs}')
        dct_h  = blocks @ basis
        dct_2d = (basis.T @ dct_h.transpose(-2,-1)).transpose(-2,-1)
        return torch.log1p((dct_2d**2).mean(dim=1).view(B, bs*bs))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = (x * self._std + self._mean).clamp(0.,1.)
        r,g,b = x_raw[:,0:1], x_raw[:,1:2], x_raw[:,2:3]
        Y  =  0.299*r + 0.587*g + 0.114*b
        Cb = -0.1687*r - 0.3313*g + 0.5*b + 0.5
        Cr =  0.5*r - 0.4187*g - 0.0813*b + 0.5
        feats = [
            self._dct2_blocks(ch, bs)
            for ch in [Y, Cb, Cr]
            for bs in self._BLOCK_SIZES
        ]
        return self.proj(torch.cat(feats, dim=1))


class FFTPhaseStream(nn.Module):
    def __init__(self, out_dim: int = 64, coherence_window: int = 7):
        super().__init__()
        self.net = nn.Sequential(
            DepthwiseSepConv(2, 32, 3, stride=2),
            DepthwiseSepConv(32, out_dim, 3, stride=2))
        self.norm = nn.LayerNorm(out_dim)
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        gauss = self._build_gaussian(coherence_window, coherence_window/3.0)
        self.register_buffer('_gauss_weight', gauss)

    @staticmethod
    def _build_gaussian(window: int, sigma: float) -> torch.Tensor:
        c = torch.arange(window, dtype=torch.float32) - window//2
        g = torch.exp(-0.5*(c/sigma)**2); g /= g.sum()+1e-8
        return (g.unsqueeze(0)*g.unsqueeze(1)).view(1,1,window,window)

    def _phase_coherence(self, phase: torch.Tensor) -> torch.Tensor:
        w=self._gauss_weight.shape[-1]; pad=w//2
        gauss = self._gauss_weight
        sin_m = F.conv2d(F.pad(torch.sin(phase),[pad]*4,'reflect'),gauss,padding=0)
        cos_m = F.conv2d(F.pad(torch.cos(phase),[pad]*4,'reflect'),gauss,padding=0)
        return (sin_m**2 + cos_m**2).clamp(0.).sqrt()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = (x * self._std + self._mean).clamp(0., 1.)
        gray  = 0.299*x_raw[:,0:1] + 0.587*x_raw[:,1:2] + 0.114*x_raw[:,2:3]
        fft   = torch.fft.fftshift(torch.fft.fft2(gray.squeeze(1)), dim=[-2,-1])
        raw_mag = torch.log1p(fft.abs() + 1e-8).unsqueeze(1)
        B = raw_mag.shape[0]
        mag_mean = raw_mag.view(B,-1).mean(1,keepdim=True).view(B,1,1,1)
        mag_std  = raw_mag.view(B,-1).std(1,keepdim=True).view(B,1,1,1) + 1e-6
        mag  = (raw_mag - mag_mean) / mag_std
        phase = torch.nan_to_num(fft.angle(), nan=0.0)
        coh  = self._phase_coherence(phase.unsqueeze(1))
        feat = self.net(torch.cat([mag, coh], dim=1).clamp(-5., 5.))
        out  = self.norm(feat.mean(dim=[2,3]))
        return torch.nan_to_num(out, nan=0.0)


class NoiseConsistencyBlock(nn.Module):
    def __init__(self, out_dim: int = 32, smooth_sigma: float = 1.5):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(15, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        self.register_buffer('_gauss_1d', self._gauss1d(7, smooth_sigma))

    @staticmethod
    def _gauss1d(ks=7, sigma=1.5) -> torch.Tensor:
        c = torch.arange(ks, dtype=torch.float32) - ks//2
        g = torch.exp(-0.5*(c/sigma)**2); return g/g.sum()

    def _smooth(self, x: torch.Tensor) -> torch.Tensor:
        C=x.shape[1]; k=self._gauss_1d; ks=k.shape[0]; pad=ks//2
        kh=k.view(1,1,1,ks).expand(C,1,1,ks)
        x=F.conv2d(F.pad(x,[pad,pad,0,0],'reflect'),kh,groups=C)
        kv=k.view(1,1,ks,1).expand(C,1,ks,1)
        return F.conv2d(F.pad(x,[0,0,pad,pad],'reflect'),kv,groups=C)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = (x * self._std + self._mean).clamp(0., 1.)
        residual = (x_raw - self._smooth(x_raw)).view(x_raw.shape[0], 3, -1)
        noise = F.normalize(residual, dim=2, eps=1e-6)
        feats = []
        for c in range(3):
            n  = noise[:, c:c+1].view(x_raw.shape[0], 1, x_raw.shape[2], x_raw.shape[3])
            ch = (n[:,:,:,:-1] * n[:,:,:,1:]).mean([2,3])
            cv = (n[:,:,:-1,:] * n[:,:,1:,:]).mean([2,3])
            cd = (n[:,:,:-1,:-1] * n[:,:,1:,1:]).mean([2,3])
            ns = n.std([2,3])
            nm = n.mean([2,3])
            # FIX: clamp is applied to the FULL skewness expression (not just denominator)
            sk = (((n - nm.unsqueeze(-1).unsqueeze(-1))**3).mean([2,3]) /
                  (ns**3 + 1e-6)).clamp(-10., 10.)
            feats.extend([ch, cv, cd, ns, sk])
        out = torch.cat(feats, dim=1)
        out = torch.nan_to_num(out, nan=0.0)
        return self.proj(out)


class JPEGAwareBlock(nn.Module):
    """
    Returns raw JPEG grid strength scores (B, 3) — one per YCbCr channel.
    IMPORTANT: kept identical to ForensicsAI (no projection layer).
    stream_dims['jpeg'] = 3 ensures checkpoint compatibility.
    """
    def __init__(self):
        super().__init__()
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))

    def _grid_strength(self, chan: torch.Tensor) -> torch.Tensor:
        B,_,H,W = chan.shape
        fft  = torch.fft.fft2(chan.squeeze(1))
        power= fft.abs()**2
        total= power.mean([1,2])+1e-8
        grid = torch.zeros(B, device=chan.device)
        for hb in [H//8, 2*H//8, 3*H//8]:
            for wb in [W//8, 2*W//8, 3*W//8]:
                if 0<hb<H and 0<wb<W:
                    grid+=power[:,hb,wb]+power[:,H-hb,wb]+power[:,hb,W-wb]
        return (grid/(total*9+1e-8)).clamp(0.,1.).unsqueeze(1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_raw = (x*self._std+self._mean).clamp(0.,1.)
        r,g,b = x_raw[:,0:1],x_raw[:,1:2],x_raw[:,2:3]
        Y  =  0.299*r+0.587*g+0.114*b
        Cb = -0.1687*r-0.3313*g+0.5*b+0.5
        Cr =  0.5*r-0.4187*g-0.0813*b+0.5
        return torch.cat([self._grid_strength(Y),
                          self._grid_strength(Cb),
                          self._grid_strength(Cr)], dim=1)


# ── Patch Attention Pool ───────────────────────────────────────────────────────

class PatchAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim), nn.Linear(dim, hidden),
            nn.Tanh(), nn.Linear(hidden, 1, bias=False))

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w = torch.softmax(self.attn(x), dim=1)
        return (w * x).sum(dim=1), w.squeeze(-1)


# ── CrossLevelArtifactAttention ────────────────────────────────────────────────

class CrossLevelArtifactAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(
            dim, num_heads, dropout=dropout, batch_first=True, add_bias_kv=True)
        self.proj    = nn.Linear(dim, dim)
        self.drop    = nn.Dropout(dropout)
        self.gate_mlp= nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,3))

    def forward(self, cls, early_p, mid_p, final_p) -> torch.Tensor:
        gates = torch.softmax(self.gate_mlp(cls), dim=-1)
        kv = torch.cat([
            gates[:,0:1].unsqueeze(2)*early_p,
            gates[:,1:2].unsqueeze(2)*mid_p,
            gates[:,2:3].unsqueeze(2)*final_p], dim=1)
        q_n  = self.norm_q(cls).unsqueeze(1)
        out,_= self.attn(q_n, self.norm_kv(kv), self.norm_kv(kv))
        return self.drop(self.proj((out+q_n).squeeze(1)))


# ── WarmupCosineClassifier ────────────────────────────────────────────────────

class WarmupCosineClassifier(nn.Module):
    def __init__(self, in_features: int, num_classes: int,
                 scale_start: float = 12.0, scale_end: float = 20.0,
                 warmup_steps: int = 600):
        super().__init__()
        self.weight       = nn.Parameter(torch.empty(num_classes, in_features))
        self.scale        = nn.Parameter(torch.tensor(scale_start))
        self.scale.requires_grad = False
        self.fake_bias    = nn.Parameter(torch.tensor(0.0))
        self.scale_start  = scale_start
        self.scale_end    = scale_end
        self.warmup_steps = warmup_steps
        self.register_buffer('_step', torch.tensor(0))
        nn.init.orthogonal_(self.weight)

    def step_warmup(self):
        if self._step.item() < self.warmup_steps:
            frac = self._step.item() / self.warmup_steps
            self.scale.data.fill_(
                self.scale_start + frac*(self.scale_end-self.scale_start))
            self._step += 1
        else:
            self.scale.requires_grad = True

    def unfreeze_scale(self):
        self.scale.requires_grad = True

    def adapt_bias(self, delta: float):
        with torch.no_grad():
            self.fake_bias.data.add_(delta).clamp_(-1.0, 1.0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        xn = F.normalize(x, dim=1)
        wn = F.normalize(self.weight, dim=1)
        logits = self.scale * (xn @ wn.T)
        bias = torch.stack(
            [self.fake_bias, torch.zeros(1,device=x.device).squeeze()], dim=0)
        return torch.clamp(logits + bias.unsqueeze(0), -8., 8.)


# ══════════════════════════════════════════════════════════════════════
# AR-5: Authenticity Consistency Loss
# ══════════════════════════════════════════════════════════════════════

class AuthenticityConsistencyLoss(nn.Module):
    """AR-5: Multi-component aux loss for authenticity reasoning."""
    def __init__(self,
                 w_agreement:  float = 0.05,
                 w_parity:     float = 0.03,
                 w_nig:        float = 1.0,
                 w_freq:       float = 0.01,
                 parity_threshold: float = 0.3):
        super().__init__()
        self.w_agreement = w_agreement
        self.w_parity    = w_parity
        self.w_nig       = w_nig
        self.w_freq      = w_freq
        self.parity_thr  = parity_threshold

    def forward(self,
                logits:      torch.Tensor,
                labels:      torch.Tensor,
                fusion_stats: FusionStats,
                pixel_attn:  Optional[torch.Tensor]
                ) -> Tuple[torch.Tensor, Dict[str, float]]:
        device = logits.device

        eff_vals = torch.stack(
            list(fusion_stats.effective.values()), dim=1)
        is_fake = (labels == 0).float()
        is_real = (labels == 1).float()
        stream_std  = eff_vals.std(dim=1)
        agree_real  = (F.relu(stream_std - 0.1) * is_real).mean()
        stream_min  = eff_vals.min(dim=1).values
        agree_fake  = (stream_min * is_fake).mean()
        L_agree     = (agree_real - agree_fake).clamp(-1.5, 1.5)

        vit_eff  = fusion_stats.effective.get('vit', eff_vals[:,0:1].squeeze(1))
        for_effs = [v for k, v in fusion_stats.effective.items() if k != 'vit']
        if for_effs:
            for_eff = torch.stack(for_effs, dim=1).mean(dim=1)
        else:
            for_eff = vit_eff

        vit_dominance = F.relu(vit_eff - for_eff - self.parity_thr)
        L_parity = (vit_dominance * is_fake).mean()

        L_nig = fusion_stats.total_nig_loss()
        try:
            if not L_nig.requires_grad:
                L_nig = torch.tensor(0.0, device=device)
        except Exception:
            L_nig = torch.tensor(0.0, device=device)

        if pixel_attn is not None:
            real_conf = torch.softmax(logits, dim=1)[:, 1]
            max_attn  = pixel_attn.view(pixel_attn.shape[0], -1).max(dim=1).values
            L_freq    = F.relu(max_attn - 0.7) * real_conf
            L_freq    = self.w_freq * L_freq.mean()
        else:
            L_freq = torch.tensor(0.0, device=device)

        total = (self.w_agreement * L_agree +
                 self.w_parity    * L_parity +
                 self.w_nig       * L_nig +
                 L_freq)

        return total, {
            'L_agree':  L_agree.item(),
            'L_parity': L_parity.item(),
            'L_nig':    L_nig.item() if hasattr(L_nig, 'item') else 0.0,
            'L_freq':   L_freq.item(),
        }


FrequencyConsistencyLoss = AuthenticityConsistencyLoss


# ══════════════════════════════════════════════════════════════════════
# AR-6: Deep Postprocessing Invariance Augmentation
# ForensicsAI probabilities restored. Four safe v3 aug ops added.
# Total: 14 aug types (10 from ForensicsAI + 4 new from v3).
# ══════════════════════════════════════════════════════════════════════

class DeepPostprocessingAug(nn.Module):
    """
    AR-6: Postprocessing invariance augmentation.

    ForensicsAI's original probabilities are RESTORED (higher = more robust).
    Four new safe ops from v3 added (style_jitter, latent_grid, local_blur,
    tone_map) — these only affect augmentation, not model structure.
    chain_prob=0.30 and chain_len=3 restored (v3 lowered these, hurting robustness).
    eval() mode: identity pass-through.
    """
    def __init__(self,
                 freq_aug_prob:    float = 0.30,
                 jpeg_sim_prob:    float = 0.25,
                 blur_prob:        float = 0.25,
                 sharpen_prob:     float = 0.20,
                 resize_prob:      float = 0.25,
                 noise_prob:       float = 0.25,
                 chroma_prob:      float = 0.20,
                 diffusion_prob:   float = 0.20,
                 isp_prob:         float = 0.15,
                 thumbnail_prob:   float = 0.15,
                 # New safe ops from v3
                 style_jitter_prob: float = 0.12,
                 latent_grid_prob:  float = 0.10,
                 local_blur_prob:   float = 0.12,
                 tone_map_prob:     float = 0.10,
                 # Chain
                 chain_prob:       float = 0.30,
                 chain_len:        int   = 3,
                 noise_scale:      float = 0.02,
                 blur_sigma_max:   float = 2.0,
                 resize_min_frac:  float = 0.5):
        super().__init__()
        self.probs = dict(
            freq_aug=freq_aug_prob, jpeg_sim=jpeg_sim_prob,
            blur=blur_prob, sharpen=sharpen_prob, resize=resize_prob,
            noise=noise_prob, chroma=chroma_prob, diffusion=diffusion_prob,
            isp=isp_prob, thumbnail=thumbnail_prob,
            style_jitter=style_jitter_prob, latent_grid=latent_grid_prob,
            local_blur=local_blur_prob, tone_map=tone_map_prob)
        self.chain_prob      = chain_prob
        self.chain_len       = chain_len
        self.noise_scale     = noise_scale
        self.blur_sigma_max  = blur_sigma_max
        self.resize_min_frac = resize_min_frac

    # ── ForensicsAI original ops ───────────────────────────────────────

    def _spectral_noise(self, x: torch.Tensor) -> torch.Tensor:
        n = torch.randn_like(x) * self.noise_scale
        s = F.avg_pool2d(F.pad(n,[1,1,1,1],'reflect'),3,stride=1,padding=0)
        return (x+n-s).clamp(-3.,3.)

    def _jpeg_simulate(self, x: torch.Tensor) -> torch.Tensor:
        H,W = x.shape[-2],x.shape[-1]
        fft = torch.fft.rfft2(x); fh,fw = fft.shape[-2],fft.shape[-1]
        kh  = max(1,int(fh*(0.4+0.5*random.random())))
        kw  = max(1,int(fw*(0.4+0.5*random.random())))
        mask = torch.zeros_like(fft.real)
        mask[...,:kh,:kw] = 1.0
        return torch.fft.irfft2(fft*mask,s=(H,W)).clamp(-3.,3.)

    def _gaussian_blur(self, x: torch.Tensor) -> torch.Tensor:
        sigma = random.uniform(0.3, self.blur_sigma_max)
        ks = max(3, int(sigma*3)*2+1)
        c = torch.arange(ks, dtype=torch.float32, device=x.device) - ks//2
        g1d = torch.exp(-0.5*(c/sigma)**2); g1d /= g1d.sum()
        C = x.shape[1]
        kh = g1d.view(1,1,1,ks).expand(C,1,1,ks)
        x = F.conv2d(F.pad(x,[ks//2,ks//2,0,0],'reflect'), kh, groups=C)
        kv = g1d.view(1,1,ks,1).expand(C,1,ks,1)
        return F.conv2d(F.pad(x,[0,0,ks//2,ks//2],'reflect'), kv, groups=C).clamp(-3.,3.)

    def _unsharp_sharpen(self, x: torch.Tensor) -> torch.Tensor:
        blurred = self._gaussian_blur(x)
        return (x + random.uniform(0.3,1.2)*(x-blurred)).clamp(-3.,3.)

    def _resize_perturb(self, x: torch.Tensor) -> torch.Tensor:
        H,W = x.shape[-2],x.shape[-1]
        frac = random.uniform(self.resize_min_frac, 0.95)
        H2,W2 = max(32,int(H*frac)), max(32,int(W*frac))
        mode1 = random.choice(['bilinear','bicubic','nearest-exact'])
        kw1   = {} if mode1=='nearest-exact' else {'align_corners':False}
        mode2 = random.choice(['bilinear','bicubic'])
        return F.interpolate(
            F.interpolate(x,size=(H2,W2),mode=mode1,**kw1),
            size=(H,W), mode=mode2, align_corners=False).clamp(-3.,3.)

    def _gaussian_noise(self, x: torch.Tensor) -> torch.Tensor:
        return (x + torch.randn_like(x)*random.uniform(0.005,0.04)).clamp(-3.,3.)

    def _chroma_shift(self, x: torch.Tensor) -> torch.Tensor:
        shift = (torch.rand(1,x.shape[1],1,1,device=x.device)-0.5)*0.12
        return (x+shift).clamp(-3.,3.)

    def _diffusion_patch_perturb(self, x: torch.Tensor) -> torch.Tensor:
        H,W  = x.shape[-2],x.shape[-1]
        scale  = random.uniform(0.01, 0.04)
        ph, pw = max(1, H//8), max(1, W//8)
        noise_small = torch.randn(1, x.shape[1], ph, pw, device=x.device)*scale
        noise_up    = F.interpolate(noise_small,(H,W),mode='bilinear',align_corners=False)
        return (x + noise_up).clamp(-3., 3.)

    def _isp_simulate(self, x: torch.Tensor) -> torch.Tensor:
        gain = 1.0 + (torch.rand(1, x.shape[1], 1, 1, device=x.device)-0.5)*0.15
        dm_noise = torch.randn(1, x.shape[1], x.shape[2]//4, x.shape[3]//4,
                               device=x.device) * 0.01
        dm_noise = F.interpolate(dm_noise, size=x.shape[-2:], mode='nearest')
        return (x * gain + dm_noise).clamp(-3., 3.)

    def _thumbnail_crop_pad(self, x: torch.Tensor) -> torch.Tensor:
        H,W  = x.shape[-2],x.shape[-1]
        frac = random.uniform(0.7, 0.95)
        ch,cw = int(H*frac), int(W*frac)
        y1 = random.randint(0, H-ch); x1 = random.randint(0, W-cw)
        cropped = x[:,:,y1:y1+ch,x1:x1+cw]
        return F.interpolate(cropped,(H,W),mode='bilinear',align_corners=False).clamp(-3.,3.)

    # ── New safe ops from v3 ───────────────────────────────────────────

    def _style_jitter(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate style-transfer color/contrast jitter used by modern GANs."""
        B,C,H,W = x.shape
        alpha = 1.0 + (torch.rand(B,C,1,1,device=x.device)-0.5)*0.3
        beta  = (torch.rand(B,C,1,1,device=x.device)-0.5)*0.15
        return (alpha*x + beta).clamp(-3.,3.)

    def _latent_grid_artifact(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate checkerboard artifacts from deconvolution/bilinear upsampling."""
        H,W = x.shape[-2:]; stride=random.choice([4,8,16])
        grid = torch.zeros_like(x); amp = random.uniform(0.005, 0.02)
        grid[:,:,::stride,::stride]   =  amp
        grid[:,:,::stride,1::stride]  = -amp/2
        grid[:,:,1::stride,::stride]  = -amp/2
        return (x + grid).clamp(-3.,3.)

    def _local_content_blur(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate GAN depth-of-field synthesis: blur a random background region."""
        H,W = x.shape[-2:]; blurred = self._gaussian_blur(x)
        mask = torch.zeros(1,1,H,W,device=x.device)
        y1=random.randint(H//4,H//2); y2=random.randint(H//2,3*H//4)
        x1_=random.randint(W//4,W//2); x2_=random.randint(W//2,3*W//4)
        mask[:,:,y1:y2,x1_:x2_] = 1.0
        mask = F.avg_pool2d(F.pad(mask,[8,8,8,8],'reflect'),17,1,0)
        return (x*(1-mask) + blurred*mask).clamp(-3.,3.)

    def _tone_map_simulate(self, x: torch.Tensor) -> torch.Tensor:
        """Simulate HDR tone mapping used by diffusion models."""
        gamma = random.uniform(0.7, 1.4)
        xr    = (x + 3.0) / 6.0
        xr    = xr.clamp(0,1)**gamma
        return (xr*6.0 - 3.0).clamp(-3.,3.)

    def _get_all_ops(self) -> List:
        return [
            ('freq_aug',    self._spectral_noise),
            ('jpeg_sim',    self._jpeg_simulate),
            ('blur',        self._gaussian_blur),
            ('sharpen',     self._unsharp_sharpen),
            ('resize',      self._resize_perturb),
            ('noise',       self._gaussian_noise),
            ('chroma',      self._chroma_shift),
            ('diffusion',   self._diffusion_patch_perturb),
            ('isp',         self._isp_simulate),
            ('thumbnail',   self._thumbnail_crop_pad),
            ('style_jitter',self._style_jitter),
            ('latent_grid', self._latent_grid_artifact),
            ('local_blur',  self._local_content_blur),
            ('tone_map',    self._tone_map_simulate),
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training:
            return x

        out = x.clone()
        target_hw = out.shape[-2:]
        B = x.shape[0]
        all_ops = self._get_all_ops()

        # Batch-level chain augmentation (faster: apply to whole batch at once)
        if random.random() < self.chain_prob:
            chosen = random.sample(all_ops, k=min(self.chain_len, len(all_ops)))
            for _, fn in chosen:
                aug = fn(out)
                if aug.shape[-2:] != target_hw:
                    aug = F.interpolate(aug, size=target_hw,
                                        mode="bilinear", align_corners=False)
                out = aug
        else:
            # Per-op probabilistic application over the whole batch
            for name, fn in all_ops:
                if random.random() < self.probs.get(name, 0.0):
                    aug = fn(out)
                    if aug.shape[-2:] != target_hw:
                        aug = F.interpolate(aug, size=target_hw,
                                            mode="bilinear", align_corners=False)
                    out = aug

        return out


RobustAugWrapper   = DeepPostprocessingAug
FreqAugmentWrapper = DeepPostprocessingAug


# ══════════════════════════════════════════════════════════════════════
# AR-7: Adaptive Stream Router + Contribution Tracker
# ══════════════════════════════════════════════════════════════════════

class AdaptiveStreamRouter:
    """AR-7: Stream contribution tracking (pure Python, zero VRAM)."""
    EMA_ALPHA  = 0.05

    def __init__(self, stream_names: List[str]):
        self.stream_names = list(stream_names)
        n = len(stream_names)
        self._ema = {
            'gates':      {s: 1.0/n for s in stream_names},
            'reliability':{s: 1.0   for s in stream_names},
            'aleatoric':  {s: 0.0   for s in stream_names},
            'epistemic':  {s: 0.0   for s in stream_names},
            'effective':  {s: 1.0/n for s in stream_names},
        }
        self._var_gates = {s: 0.0 for s in stream_names}
        self._step      = 0
        self._pruned: set = set()

    def update(self, stats):
        self._step += 1
        a = self.EMA_ALPHA
        g = stats.mean_gates()
        r = stats.mean_reliability()
        al= stats.mean_aleatoric()
        ep= stats.mean_epistemic()
        ef= stats.mean_effective()
        for n in self.stream_names:
            old_g = self._ema['gates'][n]
            new_g = g.get(n, old_g)
            self._ema['gates'][n]       = (1-a)*old_g + a*new_g
            self._ema['reliability'][n] = (1-a)*self._ema['reliability'][n] + a*r.get(n,1.)
            self._ema['aleatoric'][n]   = (1-a)*self._ema['aleatoric'][n]   + a*al.get(n,0.)
            self._ema['epistemic'][n]   = (1-a)*self._ema['epistemic'][n]   + a*ep.get(n,0.)
            self._ema['effective'][n]   = (1-a)*self._ema['effective'][n]   + a*ef.get(n,1./len(self.stream_names))
            self._var_gates[n] = (1-a)*self._var_gates[n] + a*(new_g - self._ema['gates'][n])**2

    def get_low_contribution_streams(self, threshold: float = 0.03) -> List[str]:
        return [n for n in self.stream_names
                if n not in self._pruned
                and self._ema['effective'][n] < threshold]

    def prune_stream(self, name: str):
        if name in self.stream_names and name != 'vit':
            self._pruned.add(name)
            print(f"  [AdaptiveStreamRouter] Stream '{name}' marked as pruned.")

    def get_routing_report(self) -> str:
        lines = [f"AdaptiveStreamRouter (step {self._step})",
                 f"{'Stream':<12} {'Gate':>8} {'Relbl':>8} {'Aleat':>8} "
                 f"{'Epist':>8} {'Effec':>8} {'Status':>10}"]
        lines.append("-" * 70)
        for n in self.stream_names:
            status = "PRUNED" if n in self._pruned else (
                "⚠️ LOW" if self._ema['effective'][n] < 0.05 else "OK")
            lines.append(
                f"  {n:<10} {self._ema['gates'][n]:>8.4f} "
                f"{self._ema['reliability'][n]:>8.4f} "
                f"{self._ema['aleatoric'][n]:>8.4f} "
                f"{self._ema['epistemic'][n]:>8.4f} "
                f"{self._ema['effective'][n]:>8.4f} {status:>10}")
        return "\n".join(lines)

    def get_summary(self) -> Dict:
        return {
            'step':      self._step,
            'gate_ema':  {n: round(v,4) for n,v in self._ema['gates'].items()},
            'relbl_ema': {n: round(v,4) for n,v in self._ema['reliability'].items()},
            'uncert_ema':{n: round(v,4) for n,v in self._ema['epistemic'].items()},
            'effective': {n: round(v,4) for n,v in self._ema['effective'].items()},
            'pruned':    list(self._pruned),
        }


StreamImportanceTracker = AdaptiveStreamRouter


# ── Error Learning Memory ──────────────────────────────────────────────────────

class ErrorLearningMemory:
    """
    Learns from both CORRECT and INCORRECT predictions.
    EMA_ALPHA=0.04: balanced responsiveness (~25-step half-life).
    v3 improvement: collapse guard — doesn't adjust weights when errors are near-total.
    """
    EMA_ALPHA = 0.04

    def __init__(self, max_records: int = 500):
        self.max_records   = max_records
        self.records: List[Dict]         = []
        self.success_records: List[Dict] = []
        self.error_stats   = {
            'real_as_fake': [], 'fake_as_real': [],
            'error_patterns': defaultdict(int)}
        self.success_stats = {
            'real_correct': [], 'fake_correct': [],
            'success_patterns': defaultdict(int)}
        self._ema_fake_err_rate = 0.0
        self._ema_real_err_rate = 0.0
        self._error_stream_norms:   Dict[str,List[float]] = defaultdict(list)
        self._success_stream_norms: Dict[str,List[float]] = defaultdict(list)

    def record(self, true_label: int, pred_label: int, confidence: float,
               stream_norms: Optional[Dict]=None, image_path: str=''):
        label_map = {0:'FAKE', 1:'REAL'}
        if pred_label == true_label:
            self._record_success(true_label, confidence, stream_norms, image_path)
            return
        error_type = f"{label_map[true_label]}_as_{label_map[pred_label]}"
        key = error_type.lower().replace('-','_')
        if key in self.error_stats: self.error_stats[key].append(confidence)
        dominant = None
        if stream_norms:
            valid = {k: v for k, v in stream_norms.items()
                     if isinstance(v, (int, float))}
            dominant = max(valid, key=valid.get) if valid else "unknown"
            self.error_stats['error_patterns'][f"{error_type}_{dominant}"] += 1
            for k,v in stream_norms.items(): self._error_stream_norms[k].append(v)
        is_fake_err = (true_label==0 and pred_label==1)
        self._ema_fake_err_rate = (1-self.EMA_ALPHA)*self._ema_fake_err_rate + self.EMA_ALPHA*float(is_fake_err)
        self._ema_real_err_rate = (1-self.EMA_ALPHA)*self._ema_real_err_rate + self.EMA_ALPHA*float(not is_fake_err)
        self.records.append({
            'ts':datetime.now().isoformat(), 'true':label_map[true_label],
            'pred':label_map[pred_label], 'confidence':round(confidence,4),
            'dominant_stream':dominant, 'image_path':image_path})
        if len(self.records) > self.max_records: self.records.pop(0)

    def _record_success(self, tl, conf, snorms, ip):
        lm={0:'FAKE',1:'REAL'}; st=f"{'fake' if tl==0 else 'real'}_correct"
        self.success_stats[st].append(conf)
        if snorms:
            valid = {k: v for k, v in snorms.items() if isinstance(v, (int, float))}
            if valid:
                d = max(valid, key=valid.get)
                self.success_stats['success_patterns'][f"{lm[tl]}_ok_{d}"] += 1
                for k, v in valid.items():
                    self._success_stream_norms[k].append(v)
        self.success_records.append({
            'ts':datetime.now().isoformat(),'true':lm[tl],
            'confidence':round(conf,4),'image_path':ip})
        if len(self.success_records) > self.max_records:
            self.success_records.pop(0)

    def get_loss_weights(self, base_weights: List[float]) -> List[float]:
        """
        Adapt class weights based on recent error rates.
        v3 collapse guard: if errors near-total, trust base weights.
        """
        w = list(base_weights)
        # Collapse guard: don't adjust when one class dominates errors
        if self._ema_fake_err_rate < 0.8:
            w[0] = min(base_weights[0] * (1.0 + 0.30 * self._ema_fake_err_rate),
                       base_weights[0] * 2.0)
        if self._ema_real_err_rate < 0.8:
            w[1] = max(
                base_weights[1],
                min(base_weights[1] * (1.0 + 0.60 * self._ema_real_err_rate),
                    base_weights[1] * 2.5),
            )
        return w

    def suggest_bias_adjustment(self) -> float:
        fe=self._ema_fake_err_rate; re=self._ema_real_err_rate
        # Don't adjust bias during collapse
        if fe > 0.85 or re > 0.85: return 0.0
        # Direct error-rate difference, scaled to produce meaningful adjustments
        diff = fe - re
        magnitude = abs(diff) / (max(fe, re) + 1e-8)
        return float(0.08 * np.clip(diff * magnitude, -1., 1.))

    def summary(self) -> Dict:
        rw=self.error_stats['real_as_fake']; fw=self.error_stats['fake_as_real']
        return {
            'total_errors':      len(self.records),
            'total_successes':   len(self.success_records),
            'real_as_fake':      len(rw), 'fake_as_real': len(fw),
            'ema_fake_err_rate': round(self._ema_fake_err_rate,4),
            'ema_real_err_rate': round(self._ema_real_err_rate,4),
            'avg_conf_real_err': round(float(sum(rw)/max(len(rw),1)),4),
            'avg_conf_fake_err': round(float(sum(fw)/max(len(fw),1)),4),
            'top_error_patterns': dict(sorted(
                self.error_stats['error_patterns'].items(),
                key=lambda x:x[1],reverse=True)[:5])}

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path,'w') as f:
            json.dump({'records':self.records[-100:],
                       'success_records':self.success_records[-100:],
                       'summary':self.summary()}, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path): return
        with open(path) as f: data=json.load(f)
        self.records=data.get('records',[]); self.success_records=data.get('success_records',[])


# ── Confidence History Tracker ─────────────────────────────────────────────────

class ConfidenceHistoryTracker:
    def __init__(self, window_size: int=500, shift_threshold: float=0.22):
        self.window_size=window_size; self.shift_threshold=shift_threshold
        self._history: deque=deque(maxlen=window_size)
        self._baseline: Optional[float]=None
        self._shift_count: int=0

    def update(self, confidence: float):
        self._history.append(confidence)
        if len(self._history)==self.window_size and self._baseline is None:
            self._baseline=float(np.mean(self._history))

    def check_shift(self) -> Dict:
        if len(self._history)<50:
            return {'shift_detected':False,'reason':'insufficient_data'}
        recent=float(np.mean(list(self._history)[-50:]))
        if self._baseline is None:
            return {'shift_detected':False,'reason':'baseline_not_set','recent_mean':round(recent,4)}
        delta=abs(recent-self._baseline)
        shift_detected = delta>self.shift_threshold
        if shift_detected:
            self._shift_count += 1
            if self._shift_count >= 3:
                self._baseline = recent
                self._shift_count = 0
        else:
            self._shift_count = 0
        return {
            'shift_detected': shift_detected,
            'baseline':       round(self._baseline,4),
            'recent_mean':    round(recent,4), 'delta': round(delta,4),
            'alert': ('⚠️ Distribution shift — possible unseen generator type'
                      if shift_detected else '✅ Stable')}

    def reset_baseline(self):
        self._baseline=float(np.mean(self._history)) if self._history else None


# ── Backbone loader ────────────────────────────────────────────────────────────

def _load_backbone() -> Tuple[nn.Module, int, str]:
    vram = _vram_gb()
    if vram >= 10.0: candidates=[('ViT-L-14',1024),('ViT-B-16',512),('ViT-B-32',512)]
    elif vram >= 3.5: candidates=[('ViT-B-16',512),('ViT-B-32',512)]
    else: candidates=[('ViT-B-32',512)]
    print(f"  VRAM: {vram:.1f} GB  ->  candidates: {[c[0] for c in candidates]}")
    try:
        import open_clip
        for mid, edim in candidates:
            try:
                m,_,_=open_clip.create_model_and_transforms(mid,pretrained='openai')
                print(f"  Loaded {mid} via open_clip (embed_dim={edim})")
                return m, edim, f'openclip_{mid}'
            except Exception as e: print(f"  {mid} failed: {e}")
    except ImportError: print("  open_clip not installed, trying openai/clip")
    try:
        import clip as oc
        nm={'ViT-B-32':'ViT-B/32','ViT-B-16':'ViT-B/16','ViT-L-14':'ViT-L/14'}
        for mid, edim in candidates:
            slash=nm.get(mid,'ViT-B/32')
            try:
                m,_=oc.load(slash,device='cpu')
                print(f"  Loaded {slash} via openai/clip (embed_dim={edim})")
                return m, edim, f'openai_{mid}'
            except Exception as e: print(f"  {slash} failed: {e}")
    except ImportError: print("  openai/clip not installed, using torchvision ViT fallback")
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    m=vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    print("  Loaded torchvision ViT-B/16 as fallback (embed_dim=768)")
    return m, 768, 'torchvision_vitb16'


# ── Multi-Scale Feature Extractor ─────────────────────────────────────────────

class MultiScaleCLIPExtractor(nn.Module):
    def __init__(self, model: nn.Module, embed_dim: int, backbone_type: str):
        super().__init__()
        self.model=model; self.embed_dim=embed_dim; self.backbone_type=backbone_type
        self._early_out=self._mid_out=self._final_out=None
        self._handles: List=[]
        self._register_hooks()

    def _make_hook(self, attr):
        def _hook(m,i,out):
            t=out[0] if isinstance(out,tuple) else out
            if t.dim()==3 and t.shape[0]>t.shape[1]: t=t.permute(1,0,2).contiguous()
            setattr(self, attr, t)
        return _hook

    def _register_hooks(self):
        try:
            if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
                blocks=list(self.model.visual.transformer.resblocks.children())
            elif self.backbone_type=='torchvision_vitb16':
                blocks=list(self.model.encoder.layers.children())
            else: return
            n=len(blocks); ei=min(3,n-3); mi=min(8,n-2); fi=n-1
            self._handles=[
                blocks[ei].register_forward_hook(self._make_hook('_early_out')),
                blocks[mi].register_forward_hook(self._make_hook('_mid_out')),
                blocks[fi].register_forward_hook(self._make_hook('_final_out'))]
            print(f"  Multi-scale hooks: block {ei} + {mi} + {fi}")
        except (AttributeError, IndexError) as e:
            print(f"  Multi-scale hooks failed ({e}) — CLS fallback")

    def forward(self, x: torch.Tensor):
        self._early_out=self._mid_out=self._final_out=None
        if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
            cls=self.model.encode_image(x)
        elif self.backbone_type=='torchvision_vitb16':
            x2=self.model._process_input(x)
            ct=self.model.class_token.expand(x2.shape[0],-1,-1)
            x2=torch.cat([ct,x2],dim=1)+self.model.encoder.pos_embedding
            x2=self.model.encoder.ln(self.model.encoder.layers(self.model.encoder.dropout(x2)))
            cls=x2[:,0]
        else:
            out=self.model(x); cls=out[0] if isinstance(out,tuple) else out
            if cls.dim()>2: cls=cls[:,0]
        if cls.dim()==1: cls=cls.unsqueeze(0)
        cls=cls[:,:self.embed_dim]
        def _safe(h):
            if h is None: return cls.unsqueeze(1)
            p=h[:,1:,:]; return p[...,:self.embed_dim] if p.shape[-1]!=self.embed_dim else p
        return cls, _safe(self._early_out), _safe(self._mid_out), _safe(self._final_out)

    def __del__(self):
        for h in self._handles:
            try: h.remove()
            except Exception: pass


# ── Multi-Resolution Frequency GradCAM ────────────────────────────────────────

class GradCAM:
    def __init__(self, model: 'ForensicEngine'):
        self.model=model
        self._grads_l1=self._acts_l1=None
        self._grads_l2=self._acts_l2=None
        self._grads_l3=self._acts_l3=None
        self._handles: List=[]

    def _register_freq_hooks(self):
        for ln,g,a in [('layer1','_grads_l1','_acts_l1'),
                       ('layer2','_grads_l2','_acts_l2'),
                       ('layer3','_grads_l3','_acts_l3')]:
            tgt=getattr(self.model.freq_stream,ln)
            def make(g_,a_):
                def fwd(m,i,out): setattr(self,a_,out.detach())
                def bwd(m,gi,go):
                    if go[0] is not None: setattr(self,g_,go[0].detach())
                return fwd,bwd
            fh,bh=make(g,a)
            self._handles+=[tgt.register_forward_hook(fh),
                             tgt.register_full_backward_hook(bh)]

    def _remove_hooks(self):
        for h in self._handles:
            try: h.remove()
            except: pass
        self._handles.clear()

    @staticmethod
    def _normalize_cam(cam: torch.Tensor) -> torch.Tensor:
        cam=cam.clamp(min=0)
        return (cam-cam.min())/(cam.max()-cam.min()+1e-8)

    def _layer_cam(self, grads, acts, size):
        if grads is None or acts is None: return None
        w=grads.mean([2,3],keepdim=True); cam=(w*acts).sum(1)[0]
        return F.interpolate(self._normalize_cam(cam).unsqueeze(0).unsqueeze(0),
                             size=(size,size),mode='bilinear',align_corners=False).squeeze().cpu()

    @torch.enable_grad()
    def generate(self, image: torch.Tensor, class_idx: int=1, size: int=224) -> torch.Tensor:
        self.model.eval()
        self._grads_l1=self._acts_l1=self._grads_l2=self._acts_l2=self._grads_l3=self._acts_l3=None
        self._register_freq_hooks()
        try:
            logits=self.model(image.clone()); self.model.zero_grad()
            logits[0,class_idx].backward()
            cams=[(self._layer_cam(self._grads_l1,self._acts_l1,size),0.50),
                  (self._layer_cam(self._grads_l2,self._acts_l2,size),0.30),
                  (self._layer_cam(self._grads_l3,self._acts_l3,size),0.20)]
            valid=[(c,w) for c,w in cams if c is not None]
            if not valid: return torch.ones(size,size)*0.5
            tw=sum(w for _,w in valid)
            return self._normalize_cam(sum(c*(w/tw) for c,w in valid))
        finally: self._remove_hooks()

    def generate_pixel_attn_cam(self, image: torch.Tensor, size: int=224) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad(): _=self.model(image)
        attn=self.model.freq_stream.pixel_head._pixel_attn
        if attn is None: return torch.ones(size,size)*0.5
        return self._normalize_cam(F.interpolate(
            attn[0:1],size=(size,size),mode='bilinear',align_corners=False).squeeze()).cpu()

    @torch.enable_grad()
    def generate_overlay(self, image, class_idx=1, size=224,
                         freq_weight=0.6, pixel_weight=0.4) -> Dict:
        fc=self.generate(image,class_idx,size)
        pc=self.generate_pixel_attn_cam(image,size)
        fused=freq_weight*fc+pixel_weight*pc
        return {'freq_cam':fc,'pixel_cam':pc,'fused_cam':self._normalize_cam(fused)}


# ── Adversarial Tester (with v3's CW attack added) ────────────────────────────

class AdversarialTester:
    def __init__(self, model: 'ForensicEngine', device: torch.device):
        self.model=model; self.device=device

    @torch.enable_grad()
    def fgsm(self, images, labels, eps=8/255):
        images=images.clone().detach().requires_grad_(True).to(self.device)
        labels=labels.to(self.device); self.model.eval()
        F.cross_entropy(self.model(images),labels).backward()
        with torch.no_grad(): adv=(images+eps*images.grad.sign()).clamp(-3.,3.)
        return adv.detach()

    @torch.enable_grad()
    def pgd(self, images, labels, eps=8/255, alpha=2/255, steps=7):
        images=images.to(self.device); labels=labels.to(self.device)
        adv=(images.clone().detach()+torch.empty_like(images).uniform_(-eps,eps)).clamp(-3.,3.)
        self.model.eval()
        for _ in range(steps):
            adv=adv.detach().requires_grad_(True)
            F.cross_entropy(self.model(adv),labels).backward()
            with torch.no_grad():
                adv=torch.min(torch.max(adv+alpha*adv.grad.sign(),images-eps),images+eps).clamp(-3.,3.)
        return adv.detach()

    @torch.enable_grad()
    def cw_l2(self, images, labels, c=1e-3, steps=20, lr=0.01):
        """Carlini-Wagner L2 attack (v3 addition — for eval only)."""
        images=images.to(self.device); labels=labels.to(self.device)
        delta=torch.zeros_like(images, requires_grad=True)
        opt  =torch.optim.Adam([delta], lr=lr)
        self.model.eval()
        for _ in range(steps):
            adv = (images + delta).clamp(-3.,3.)
            logits = self.model(adv)
            target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            other_logits  = (logits - 1e9*F.one_hot(labels, logits.shape[1])).max(1).values
            loss_adv = F.relu(other_logits - target_logits + 0.01).mean()
            loss_reg = c * delta.norm(p=2)
            (loss_adv + loss_reg).backward()
            opt.step(); opt.zero_grad()
            delta.data.clamp_(-0.1, 0.1)
        return (images + delta).clamp(-3.,3.).detach()

    def evaluate(self, val_loader, attack='pgd', eps=8/255, max_batches=10):
        self.model.eval()
        all_preds=[]; all_labels=[]; all_orig=[]
        for i,(imgs,labels,*_) in enumerate(val_loader):
            if i >= max_batches: break
            imgs=imgs.to(self.device); labels=labels.to(self.device)
            if attack=='fgsm':   adv=self.fgsm(imgs,labels,eps)
            elif attack=='pgd':  adv=self.pgd(imgs,labels,eps)
            elif attack=='cw':   adv=self.cw_l2(imgs,labels)
            else:                adv=self.fgsm(imgs,labels,eps)
            with torch.no_grad():
                ap=self.model(adv).argmax(1); op=self.model(imgs).argmax(1)
            all_preds.extend(ap.cpu()); all_labels.extend(labels.cpu()); all_orig.extend(op.cpu())
        ap=np.array(all_preds); al=np.array(all_labels); ao=np.array(all_orig)
        adv_acc=(ap==al).mean()*100; orig_acc=(ao==al).mean()*100
        bal_drop=orig_acc-adv_acc
        return {'adv_acc':adv_acc,'orig_acc':orig_acc,'bal_drop':bal_drop,
                'verdict':'ROBUST' if bal_drop<10 else 'VULNERABLE' if bal_drop>25 else 'MODERATE'}


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp=path+'.tmp'; torch.save(state,tmp); os.replace(tmp,path)

def load_checkpoint(path: str, map_location='cpu') -> dict:
    if not os.path.exists(path): return {}
    try: return torch.load(path,map_location=map_location,weights_only=False)
    except Exception as e:
        print(f"  [WARN] Could not load checkpoint {path}: {e}"); return {}

def save_signed(checkpoint: dict, path: str, secret: bytes=_CKPT_SECRET):
    data=pickle.dumps(checkpoint)
    sig=hmac.new(secret,data,hashlib.sha256).hexdigest()
    torch.save({'payload':checkpoint,'signature':sig},path)

def load_signed(path: str, secret: bytes=_CKPT_SECRET, strict: bool=True) -> dict:
    saved=torch.load(path,map_location='cpu',weights_only=False)
    if 'signature' not in saved:
        if strict: raise ValueError(f"No signature in {path}")
        return saved
    data=pickle.dumps(saved['payload'])
    expected=hmac.new(secret,data,hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected,saved['signature']):
        raise ValueError("Checkpoint signature mismatch — file may be tampered")
    return saved['payload']


# ── Mix/CutMix ────────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    if alpha<=0 or not torch.is_tensor(x): return x,y,y,1.0
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=0.5):
    if alpha<=0: return x,y,y,1.0
    lam=np.random.beta(alpha,alpha); B,C,H,W=x.shape
    idx=torch.randperm(B,device=x.device); cr=math.sqrt(1.-lam)
    cw,ch=int(W*cr),int(H*cr); cx,cy=random.randint(0,W),random.randint(0,H)
    x1,x2=max(cx-cw//2,0),min(cx+cw//2,W); y1,y2=max(cy-ch//2,0),min(cy+ch//2,H)
    mix=x.clone(); mix[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
    return mix, y, y[idx], 1.-(x2-x1)*(y2-y1)/(W*H)

def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam*criterion(logits,y_a)+(1-lam)*criterion(logits,y_b)


# ── Loss Functions ────────────────────────────────────────────────────────────

class AsymmetricFocalLoss(nn.Module):
    """
    Asymmetric Focal Loss with per-class gamma and a hard-REAL floor.
    Identical to ForensicsAI — proven stable.
    """
    def __init__(self, gamma_fake: float = 2.0, gamma_real: float = 2.5,
                 weight=None, label_smoothing: float = 0.03,
                 real_floor_weight: float = 0.05):
        super().__init__()
        self.gamma_fake        = gamma_fake
        self.gamma_real        = gamma_real
        self.weight            = weight
        self.label_smoothing   = label_smoothing
        self.real_floor_weight = real_floor_weight

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce = F.cross_entropy(
            logits, targets,
            weight         = self.weight,
            label_smoothing= self.label_smoothing,
            reduction      = 'none',
        )
        pt    = torch.exp(-ce.clamp(max=80.0))
        gamma = torch.where(
            targets == 0,
            torch.full_like(ce, self.gamma_fake),
            torch.full_like(ce, self.gamma_real),
        )
        focal = ((1.0 - pt) ** gamma * ce)

        if self.real_floor_weight > 0.0:
            real_mask  = (targets == 1).float()
            real_probs = torch.softmax(logits, dim=1)[:, 1].clamp(1e-6, 1.0 - 1e-6)
            floor_bce  = -torch.log(real_probs) * real_mask
            focal = focal + self.real_floor_weight * floor_bce

        return focal.mean()

    def update_weight(self, w: torch.Tensor):
        dev = self.weight.device if self.weight is not None else torch.device('cpu')
        self.weight = w.to(dev)

    @property
    def rfw(self) -> float:
        return self.real_floor_weight

    @rfw.setter
    def rfw(self, value: float):
        self.real_floor_weight = float(value)

class FocalLoss(AsymmetricFocalLoss):
    def __init__(self, gamma=2., weight=None, label_smoothing=0.03):
        super().__init__(gamma_fake=gamma,gamma_real=gamma,
                         weight=weight,label_smoothing=label_smoothing)


# ── Global MC-Dropout uncertainty (model-level) ───────────────────────────────

def predict_with_uncertainty(model: 'ForensicEngine',
                              image: torch.Tensor,
                              n_passes: int = 10) -> Dict:
    """Global MC-Dropout uncertainty via multiple train-mode forward passes."""
    was_training = model.training
    model.train()
    ll = []
    with torch.no_grad():
        for _ in range(n_passes):
            ll.append(model(image))
    model.train(was_training)
    stacked = torch.stack(ll)
    probs   = torch.softmax(stacked, dim=-1)
    mp = probs.mean(0); sp = probs.std(0)
    return {
        'prediction':  mp.argmax(1),
        'confidence':  mp.max(1).values,
        'uncertainty': sp.max(1).values,
        'mean_probs':  mp,
    }


# ══════════════════════════════════════════════════════════════════════
# Helper for AdaptiveStreamRouter compat
# ══════════════════════════════════════════════════════════════════════

class _MinimalFusionStats:
    """Wraps gate_info dict into FusionStats-compatible object for AdaptiveStreamRouter."""
    def __init__(self, gate_info: Dict, stream_names: List[str]):
        self._gi = gate_info
        self.stream_names = stream_names

    def mean_gates(self)       -> Dict[str,float]: return self._gi.get('gates',{})
    def mean_reliability(self) -> Dict[str,float]: return self._gi.get('reliability',{})
    def mean_aleatoric(self)   -> Dict[str,float]: return self._gi.get('aleatoric',{})
    def mean_epistemic(self)   -> Dict[str,float]: return self._gi.get('epistemic',{})
    def mean_calibration(self) -> Dict[str,float]: return self._gi.get('calibration',{})
    def mean_effective(self)   -> Dict[str,float]: return self._gi.get('effective',{})
    def total_nig_loss(self)   -> torch.Tensor:    return torch.tensor(0.)


# ══════════════════════════════════════════════════════════════════════
# MAIN MODEL — ForensicEngine (Stable Edition)
# ══════════════════════════════════════════════════════════════════════

class ForensicEngine(nn.Module):
    """
    ForensicEngine — Stable Edition.

    Streams: ViT (CLIP multi-scale) + SRM + DCT + FFT + Noise + JPEG
    Fusion:  AuthenticityReasoningFusion (AR-1…AR-4)
    NO generator head (removed: was misleading and untrained).
    Checkpoint-compatible with ForensicsAI.py (identical stream_dims).

    Outputs: logits (2-class FAKE/REAL), full diagnostics via predict() and
             forward_with_streams().
    VRAM: ≤ 3.5 GB @ batch_size=8 ✅
    """

    def __init__(
        self,
        num_classes:         int   = 2,
        dropout:             float = 0.25,
        drop_path_rate:      float = 0.08,
        temperature:         float = 1.0,
        freeze_backbone:     bool  = True,
        unfreeze_last_n:     int   = 0,
        use_grad_checkpoint: bool  = True,
        input_size:          int   = 224,
        freq_dim:            int   = 128,
        dct_dim:             int   = 96,
        fft_dim:             int   = 64,
        noise_dim:           int   = 32,
        # Ablation flags
        use_srm:             bool  = True,
        use_fft:             bool  = True,
        use_dct:             bool  = True,
        use_noise:           bool  = True,
        # Fusion
        fusion_shared_dim:   int   = 256,
        fusion_context_dim:  int   = 128,
        # AR options
        use_isfcr:           bool  = True,
        use_fcw:             bool  = True,
        n_iters:             int   = 2,
        sparse_k:            Optional[int] = None,
        evidential_coeff:    float = 0.01,
    ):
        super().__init__()
        self.input_size         = input_size
        self.unfrozen_count     = 0
        self.freq_dim           = freq_dim
        self.dct_dim            = dct_dim
        self.fft_dim            = fft_dim
        self.noise_dim          = noise_dim
        self.use_srm            = use_srm
        self.use_fft            = use_fft
        self.use_dct            = use_dct
        self.use_noise          = use_noise
        self.use_fcw            = use_fcw

        self.register_buffer('temperature',
                             torch.tensor(temperature, dtype=torch.float32))

        raw_model, embed_dim, backbone_type = _load_backbone()
        self.embed_dim     = embed_dim
        self.backbone_type = backbone_type
        self.extractor     = MultiScaleCLIPExtractor(raw_model, embed_dim, backbone_type)

        if freeze_backbone:
            for p in self.extractor.model.parameters():
                p.requires_grad = False
            if unfreeze_last_n > 0:
                self._unfreeze_last_n(unfreeze_last_n)

        if use_grad_checkpoint:
            self._enable_grad_checkpoint()

        self.early_pool = PatchAttentionPool(embed_dim, hidden=128)
        self.mid_pool   = PatchAttentionPool(embed_dim, hidden=128)
        self.final_pool = PatchAttentionPool(embed_dim, hidden=128)

        self.cross_attn = CrossLevelArtifactAttention(embed_dim, num_heads=4,
                                                       dropout=dropout/3)
        self.drop_path  = DropPath(drop_path_rate)

        # Forensic stream modules (identical set to ForensicsAI)
        if use_srm:   self.freq_stream = FrequencyStream(out_dim=freq_dim)
        if use_dct:   self.dct_block   = MultiScaleDCTBlock(proj_dim=dct_dim)
        if use_fft:   self.fft_stream  = FFTPhaseStream(out_dim=fft_dim)
        if use_noise: self.noise_block  = NoiseConsistencyBlock(out_dim=noise_dim)
        self.jpeg_block = JPEGAwareBlock()

        # stream_dims IDENTICAL to ForensicsAI — checkpoints load cleanly
        vit_total   = embed_dim * 4
        stream_dims = {'vit': vit_total}
        if use_srm:   stream_dims['srm']   = freq_dim
        if use_dct:   stream_dims['dct']   = dct_dim
        if use_fft:   stream_dims['fft']   = fft_dim
        if use_noise: stream_dims['noise'] = noise_dim
        stream_dims['jpeg'] = 3  # raw (B,3) — matches ForensicsAI exactly

        self.stream_fusion = AuthenticityReasoningFusion(
            stream_dims     = stream_dims,
            shared_dim      = fusion_shared_dim,
            context_dim     = fusion_context_dim,
            n_iters         = n_iters,
            use_isfcr       = use_isfcr,
            use_fcw         = use_fcw,
            sparse_k        = sparse_k,
            evidential_coeff= evidential_coeff,
        )
        fusion_out_dim = self.stream_fusion.out_dim

        self.mlp = nn.Sequential(
            nn.LayerNorm(fusion_out_dim),
            nn.Linear(fusion_out_dim, 512),
            nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(), nn.Dropout(dropout/2),
        )

        self.classifier = WarmupCosineClassifier(
            256, num_classes, scale_start=8., scale_end=15., warmup_steps=1000)

        # AR-5
        self.authenticity_loss = AuthenticityConsistencyLoss(
            w_agreement=0.02, w_parity=0.02, w_nig=1.0, w_freq=0.01)

        self._init_weights()
        self.error_memory       = ErrorLearningMemory(max_records=500)
        self.confidence_tracker = ConfidenceHistoryTracker(window_size=200)
        self.stream_router      = AdaptiveStreamRouter(list(stream_dims.keys()))

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        active    = [k for k,v in {'SRM':use_srm,'DCT':use_dct,
                                    'FFT':use_fft,'Noise':use_noise}.items() if v]
        print(f"  ForensicEngine (Stable) — {total:,} total | {trainable:,} trainable "
              f"({100*trainable/total:.1f}%)")
        print(f"  Forensic streams: {active + ['JPEG']}")
        print(f"  Fusion: AR-1..4 (shared={fusion_shared_dim}, "
              f"iters={n_iters}, isfcr={use_isfcr}, fcw={use_fcw})")
        print(f"  AR-2 parity alpha (init): {self.stream_fusion.parity_gate.alpha.item():.3f}")
        print(f"  Generator head: REMOVED (was misleading)")

    def _unfreeze_last_n(self, n: int):
        if n <= 0: return
        try:
            if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
                blocks=list(self.extractor.model.visual.transformer.resblocks.children())
            elif self.backbone_type=='torchvision_vitb16':
                blocks=list(self.extractor.model.encoder.layers.children())
            else: return
            for b in blocks:
                for p in b.parameters(): p.requires_grad=False
            for b in blocks[-n:]:
                for p in b.parameters(): p.requires_grad=True
            self.unfrozen_count=n; print(f"  Backbone: last {n} ViT blocks unfrozen")
        except AttributeError as e:
            print(f"  Could not unfreeze blocks (non-critical): {e}")

    def _enable_grad_checkpoint(self):
        try:
            if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
                blocks=list(self.extractor.model.visual.transformer.resblocks.children())
            elif self.backbone_type=='torchvision_vitb16':
                blocks=list(self.extractor.model.encoder.layers.children())
            else: return
            cnt=0
            for b in blocks:
                of=b.forward
                def wrap(fwd):
                    def w(*a,**kw): return grad_checkpoint(fwd,*a,use_reentrant=False,**kw)
                    return w
                b.forward=wrap(of); cnt+=1
            print(f"  Gradient checkpointing: ALL {cnt} backbone blocks")
        except AttributeError as e:
            print(f"  Gradient checkpointing: skipped ({e})")

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None: nn.init.zeros_(m.bias)
        for pool in [self.early_pool, self.mid_pool, self.final_pool]:
            for layer in pool.attn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None: nn.init.zeros_(layer.bias)

    def _resize_if_needed(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2]!=self.input_size or x.shape[-1]!=self.input_size:
            x=F.interpolate(x,size=(self.input_size,self.input_size),
                            mode='bilinear',align_corners=False)
        return x

    def _build_streams(self, x, early_p, mid_p, final_p, cross_out
                       ) -> Dict[str, torch.Tensor]:
        s = {'vit': torch.cat([early_p, mid_p, final_p, cross_out], dim=1)}
        if self.use_srm:   s['srm']   = self.freq_stream(x)
        if self.use_dct:   s['dct']   = self.dct_block(x)
        if self.use_fft:   s['fft']   = self.fft_stream(x)
        if self.use_noise: s['noise'] = self.noise_block(x)
        s['jpeg'] = self.jpeg_block(x)
        return s

    def _forward_core(self, x: torch.Tensor):
        """Shared forward. Returns (feat, streams, FusionStats, attn_weights)."""
        x = self._resize_if_needed(x)
        if not torch.isfinite(x).all():
            x = torch.nan_to_num(x, nan=0.0, posinf=3.0, neginf=-3.0)

        cls, early_p, mid_p, final_p = self.extractor(x)
        ep, ew = self.early_pool(early_p)
        mp, mw = self.mid_pool(mid_p)
        fp, fw = self.final_pool(final_p)

        vit_cat   = self.drop_path(torch.cat([ep, mp, fp], dim=1))
        ep        = vit_cat[:, :self.embed_dim]
        mp        = vit_cat[:, self.embed_dim:2*self.embed_dim]
        fp        = vit_cat[:, 2*self.embed_dim:]
        cross_out = self.cross_attn(cls, early_p, mid_p, final_p)
        streams   = self._build_streams(x, ep, mp, fp, cross_out)

        fused, stats = self.stream_fusion(streams)
        fused = torch.nan_to_num(fused, nan=0.0)
        feat  = self.mlp(fused)
        return feat, streams, stats, (ew, mw, fw)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        feat, _, _, _ = self._forward_core(x)
        return self.classifier(feat) / self.temperature

    def predict(self, x: torch.Tensor) -> Dict:
        
        self.eval()
        with torch.no_grad():
            feat, streams, stats, _ = self._forward_core(x)
            logits = self.classifier(feat) / self.temperature
            probs  = torch.softmax(logits, dim=1)

            preds      = probs.argmax(1)
            # Use raw (un-temperature-scaled) logits for genuine confidence
            # to avoid temperature artificially compressing confidence scores
            raw_logits = self.classifier(feat)
            raw_probs  = torch.softmax(raw_logits, dim=1)
            confidence = raw_probs.max(1).values

            ep_vals    = list(stats.mean_epistemic().values())
            uncertainty = float(sum(ep_vals) / max(len(ep_vals), 1))

            eff        = list(stats.mean_effective().values())
            eff_t      = torch.tensor(eff)
            contradiction_strength = float(eff_t.std()) if len(eff) > 1 else 0.0

        return {
            'prediction':             preds,
            'confidence':             confidence,
            'uncertainty':            uncertainty,
            'stream_importance':      stats.mean_gates(),
            'stream_reliability':     stats.mean_reliability(),
            'contradiction_strength': contradiction_strength,
        }

    def forward_with_entropy(self, x: torch.Tensor, labels: Optional[torch.Tensor]=None):
        
        feat, streams, stats, (ew, mw, fw) = self._forward_core(x)
        logits = self.classifier(feat) / self.temperature

        e_ent = -(ew * (ew+1e-8).log()).sum(1).mean()
        m_ent = -(mw * (mw+1e-8).log()).sum(1).mean()
        f_ent = -(fw * (fw+1e-8).log()).sum(1).mean()

        pixel_attn = self.freq_stream._pixel_attn if self.use_srm else None

        if labels is not None:
            aux_loss, components = self.authenticity_loss(
                logits, labels, stats, pixel_attn)
        else:
            nig = stats.total_nig_loss()
            aux_loss   = nig
            components = {'L_nig': nig.item() if hasattr(nig,'item') else 0.}

        components['e_ent'] = e_ent.item()
        components['m_ent'] = m_ent.item()
        components['f_ent'] = f_ent.item()

        return logits, aux_loss, components

    def forward_with_streams(self, x: torch.Tensor) -> Tuple[torch.Tensor, Dict]:
        """Validation forward. Returns (logits, stream_info_dict)."""
        feat, streams, stats, (ew, mw, fw) = self._forward_core(x)
        logits = self.classifier(feat) / self.temperature

        self.stream_router.update(stats)

        info = {k: v.norm(dim=1).mean().item()
                for k,v in streams.items() if k!='jpeg'}
        info['jpeg_strength_mean']  = streams['jpeg'].mean().item()
        info['early_attn_entropy']  = -(ew*(ew+1e-8).log()).sum(1).mean().item()
        info['mid_attn_entropy']    = -(mw*(mw+1e-8).log()).sum(1).mean().item()
        info['final_attn_entropy']  = -(fw*(fw+1e-8).log()).sum(1).mean().item()
        info['_gate_weights']       = stats.mean_gates()
        info['_reliability']        = stats.mean_reliability()
        info['_aleatoric']          = stats.mean_aleatoric()
        info['_epistemic']          = stats.mean_epistemic()
        info['_calibration']        = stats.mean_calibration()
        info['_effective']          = stats.mean_effective()
        info['_parity_alpha']       = self.stream_fusion.parity_gate.alpha.item()
        if self.use_fcw and hasattr(self.stream_fusion,'fcw'):
            info['_forensic_dampening'] = self.stream_fusion.fcw.dampening.item()

        for old,new in [('srm','freq_out'),('fft','fft_out'),
                         ('dct','dct_out'),('noise','noise_out'),
                         ('vit','cross_out')]:
            if old in info: info[new] = info.pop(old)
        return logits, info

    def set_temperature(self, t: float):
        self.temperature.fill_(t)

    def calibrate(self, val_loader, device, temp_range=(0.5,1.5)) -> float:
        try: from scipy.optimize import minimize_scalar
        except ImportError:
            print("scipy not found — skipping calibration"); return 1.0
        self.eval(); self.set_temperature(1.0)
        ll,labs = [],[]
        with torch.no_grad():
            for batch in val_loader:
                ll.append(self.forward(batch[0].to(device)).cpu())
                labs.append(batch[1])
        al=torch.cat(ll); ab=torch.cat(labs)
        res=minimize_scalar(
            lambda T: F.cross_entropy(al/float(T),ab).item(),
            bounds=temp_range,method='bounded')
        T_opt=float(np.clip(res.x,temp_range[0],temp_range[1]))
        self.set_temperature(T_opt)
        print(f"  Calibrated temperature: {T_opt:.4f}"); return T_opt

    def step_error_memory(self, true_labels, pred_labels, confidences, stream_norms):
        for tl,pl,conf in zip(true_labels, pred_labels, confidences):
            self.error_memory.record(int(tl),int(pl),float(conf),stream_norms)
            self.confidence_tracker.update(float(conf))
        delta=self.error_memory.suggest_bias_adjustment()
        if abs(delta)>0.008: self.classifier.adapt_bias(delta)

    def training_step_hook(self):
        """Call once per gradient step."""
        self.classifier.step_warmup()

    def get_stream_importance(self, x: torch.Tensor) -> Dict:
        with torch.no_grad():
            x=self._resize_if_needed(x)
            cls,ep,mp,fp = self.extractor(x)
            ep_p,_=self.early_pool(ep); mp_p,_=self.mid_pool(mp); fp_p,_=self.final_pool(fp)
            co = self.cross_attn(cls,ep,mp,fp)
            streams = self._build_streams(x,ep_p,mp_p,fp_p,co)
            gate_info = self.stream_fusion.get_gate_weights(streams)
        self.stream_router.update(
            _MinimalFusionStats(gate_info, self.stream_fusion.stream_names))
        return {**gate_info, 'routing_summary': self.stream_router.get_summary()}

    def get_routing_report(self) -> str:
        return self.stream_router.get_routing_report()

    def get_low_importance_streams(self, threshold: float=0.03) -> List[str]:
        return self.stream_router.get_low_contribution_streams(threshold)

    def prune_stream(self, stream_name: str):
        self.stream_router.prune_stream(stream_name)

    def get_explainability_report(self, x: torch.Tensor) -> Dict:
        """Lightweight explainability output for a batch of images."""
        self.eval()
        with torch.no_grad():
            _, _, stats, _ = self._forward_core(x)
            self.stream_router.update(_MinimalFusionStats(
                stats.uncertainty_summary(), self.stream_fusion.stream_names))

        eff_vals = list(stats.mean_effective().values())
        eff_t    = torch.tensor(eff_vals)
        contradiction_strength = float(eff_t.std()) if len(eff_vals) > 1 else 0.0

        report: Dict = {
            'stream_contribution':   stats.mean_gates(),
            'stream_reliability':    stats.mean_reliability(),
            'stream_uncertainty':    stats.mean_epistemic(),
            'parity_alpha':          self.stream_fusion.parity_gate.alpha.item(),
            'contradiction_strength': contradiction_strength,
            'routing_report':        self.stream_router.get_routing_report(),
        }
        if self.use_fcw and hasattr(self.stream_fusion, 'fcw'):
            report['forensic_dampening'] = self.stream_fusion.fcw.dampening.item()
        return report


# ── Factory functions ──────────────────────────────────────────────────────────

def get_forensic_engine(
    num_classes:         int   = 2,
    dropout:             float = 0.25,
    drop_path_rate:      float = 0.08,
    freeze_backbone:     bool  = True,
    unfreeze_last_n:     int   = 0,
    use_grad_checkpoint: bool  = True,
    input_size:          int   = 224,
    freq_dim:            int   = 128,
    dct_dim:             int   = 96,
    fft_dim:             int   = 64,
    noise_dim:           int   = 32,
    weights_path:        Optional[str] = None,
    signed:              bool  = False,
    prior_weights_path:  Optional[str] = None,
    use_srm:             bool  = True,
    use_fft:             bool  = True,
    use_dct:             bool  = True,
    use_noise:           bool  = True,
    fusion_shared_dim:   int   = 256,
    fusion_context_dim:  int   = 128,
    use_isfcr:           bool  = True,
    use_fcw:             bool  = True,
    n_iters:             int   = 2,
    sparse_k:            Optional[int] = None,
    evidential_coeff:    float = 0.01,
) -> ForensicEngine:

    model = ForensicEngine(
        num_classes=num_classes, dropout=dropout, drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone, unfreeze_last_n=unfreeze_last_n,
        use_grad_checkpoint=use_grad_checkpoint, input_size=input_size,
        freq_dim=freq_dim, dct_dim=dct_dim, fft_dim=fft_dim,
        noise_dim=noise_dim,
        use_srm=use_srm, use_fft=use_fft, use_dct=use_dct, use_noise=use_noise,
        fusion_shared_dim=fusion_shared_dim, fusion_context_dim=fusion_context_dim,
        use_isfcr=use_isfcr, use_fcw=use_fcw, n_iters=n_iters,
        sparse_k=sparse_k, evidential_coeff=evidential_coeff,
    )

    def _load(path, strict_sig=False):
        ckpt  = load_signed(path,strict=strict_sig) if signed else load_checkpoint(path)
        state = ckpt.get('model_state_dict', ckpt)
        m,u   = model.load_state_dict(state, strict=False)
        if m: print(f"  Missing keys: {len(m)}")
        if u: print(f"  Unexpected keys: {len(u)}")

    if weights_path and os.path.exists(weights_path):
        try: _load(weights_path, strict_sig=signed); print(f"  Loaded: {weights_path}")
        except Exception as e: print(f"  Could not load weights: {e}")

    if prior_weights_path and os.path.exists(prior_weights_path) and not weights_path:
        try:
            _load(prior_weights_path)
            print(f"  Migrated from prior: {prior_weights_path}")
        except Exception as e: print(f"  Migration failed: {e}")

    return model


# Backward-compat aliases
get_spatial_robust_model = get_forensic_engine
SpatialRobustModel       = ForensicEngine


def get_baseline_model(dropout: float=0.25,
                       freeze_backbone: bool=True) -> ForensicEngine:
    """Minimal baseline: ViT + SRM only."""
    print("  [Baseline] ViT + SRM + PixelFrequencyHead only")
    return get_forensic_engine(
        dropout=dropout, freeze_backbone=freeze_backbone,
        use_srm=True, use_fft=False, use_dct=False, use_noise=False,
        use_isfcr=False, use_fcw=False, n_iters=1)
