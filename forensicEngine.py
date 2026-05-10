"""
forensicEngine.py — ForensicEngine v3: Production AI Image Authenticity System
Hardware: NVIDIA RTX 2050 4GB VRAM + CUDA (batch_size=8 @ 224x224)

Architecture (AR-1…AR-9):
  AR-1  HeteroscedasticEvidentialHead    — aleatoric + epistemic + calibration
  AR-2  ParityGate                       — balanced forensic/semantic authority
  AR-3  IterativeSFContradictionReasoner — iterative semantic↔forensic contradiction
  AR-4  ForensicConfidenceWeighter       — inverse-variance forensic soft-weighting
  AR-5  AuthenticityConsistencyLoss      — agreement + parity + NIG + freq aux loss
  AR-6  DeepPostprocessingAug            — 17 aug types covering 2025-26 GAN evasion
  AR-7  AdaptiveStreamRouter             — EMA stream tracking + prune API
  AR-8  ForensicStreams                  — SRM/DCT/FFT/Noise/JPEG + NEW streams:
           LocalTextureCoherence         — micro-texture field consistency
           SemanticPhysicsStream         — lighting/shadow/depth physical plausibility
           ChromaticAberrationStream     — lens CA fingerprint vs GAN chromatic flatness
  AR-9  PGD+FGSM+CW adversarial training — robust multi-attack hardening

New 2025-26 GAN bypass mitigations:
  - Latent-space upsampling grid artifact detection (bilinear/bicubic fingerprints)
  - Perceptual hash inconsistency between freq bands
  - Global-local semantic coherence contradiction
  - Patch-level color temperature gradient analysis
  - Spectral flatness ratio (GANs produce overly smooth spectra)
  - Chromatic aberration absence (real lenses always have it; GANs don't)
  - Micro-texture periodicity (tiling from conv upsampling)
  - Physics-violating shadow/highlight analysis
  - Eye-specular-highlight geometric consistency
  - Hair/fur sub-pixel frequency analysis
"""

import os, math, hmac, hashlib, pickle, json, random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint as grad_checkpoint
from typing import Optional, Tuple, List, Dict
from collections import defaultdict, deque
from dataclasses import dataclass
from datetime import datetime

CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_CKPT_SECRET = b'forensicEngine_v3_signing_key'

GENERATOR_LABELS = [
    'real', 'dalle3', 'midjourney_v6', 'stable_diffusion_xl',
    'flux', 'firefly', 'gemini_imagen', 'unknown_gan'
]


def _vram_gb() -> float:
    return torch.cuda.get_device_properties(0).total_memory / 1e9 if torch.cuda.is_available() else 0.0


# ── Drop-path ──────────────────────────────────────────────────────────────────

class DropPath(nn.Module):
    def __init__(self, p: float = 0.0):
        super().__init__()
        self.p = p
    def forward(self, x):
        if not self.training or self.p == 0: return x
        keep = 1.0 - self.p
        mask = torch.rand((x.shape[0],) + (1,)*(x.dim()-1), device=x.device) < keep
        return x * mask / keep


class DepthwiseSepConv(nn.Module):
    def __init__(self, inc, outc, k=3, s=1):
        super().__init__()
        self.dw = nn.Conv2d(inc, inc, k, s, k//2, groups=inc, bias=False)
        self.pw = nn.Conv2d(inc, outc, 1, bias=False)
        self.bn = nn.BatchNorm2d(outc)
    def forward(self, x): return F.gelu(self.bn(self.pw(self.dw(x))))


# ══════════════════════════════════════════════════════════════════════
# AR-1: Heteroscedastic + Evidential Uncertainty Head
# ══════════════════════════════════════════════════════════════════════

@dataclass
class EvidentialOutput:
    reliability: torch.Tensor
    aleatoric:   torch.Tensor
    epistemic:   torch.Tensor
    calibration: torch.Tensor
    effective:   torch.Tensor
    nig_loss:    torch.Tensor


class HeteroscedasticEvidentialHead(nn.Module):
    def __init__(self, in_dim: int, evidential_coeff: float = 0.01):
        super().__init__()
        hidden = max(32, in_dim // 4)
        self.evidential_coeff = evidential_coeff
        self.encoder   = nn.Sequential(nn.Linear(in_dim, hidden, bias=False), nn.LayerNorm(hidden), nn.GELU())
        self.mu_head   = nn.Linear(hidden, 1)
        self.lv_head   = nn.Linear(hidden, 1)
        self.nu_head   = nn.Linear(hidden, 1)
        self.alpha_head= nn.Linear(hidden, 1)
        self.beta_head = nn.Linear(hidden, 1)
        self.register_buffer('_feat_ema',  torch.zeros(1))
        self.register_buffer('_feat_ema2', torch.ones(1))
        self._ema_a = 0.01
        nn.init.zeros_(self.mu_head.weight);    nn.init.constant_(self.mu_head.bias, 2.0)
        nn.init.zeros_(self.lv_head.weight);    nn.init.constant_(self.lv_head.bias, -2.0)
        for h in [self.nu_head, self.alpha_head, self.beta_head]:
            nn.init.zeros_(h.weight); nn.init.constant_(h.bias, 0.5)

    def _update_cal_ema(self, fn):
        if not self.training: return
        with torch.no_grad():
            m = fn.mean().detach(); m2 = (fn**2).mean().detach()
            self._feat_ema.copy_((1-self._ema_a)*self._feat_ema + self._ema_a*m)
            self._feat_ema2.copy_((1-self._ema_a)*self._feat_ema2 + self._ema_a*m2)

    def _calibration_surprise(self, x):
        fn = x.norm(dim=1, keepdim=True)
        self._update_cal_ema(fn)
        rm = self._feat_ema.clamp(min=1e-6)
        rs = (self._feat_ema2 - self._feat_ema**2).clamp(min=1e-8).sqrt()
        z  = ((fn - rm) / (rs + 1e-6)).abs()
        return torch.sigmoid(z - 2.0)

    def _nig_loss(self, mu, lv, nu, alpha, beta):
        # Exact ForensicsAI formula — evidential_coeff=0.01 keeps this naturally small (~0.01 scale)
        var = torch.exp(lv).clamp(min=1e-6)
        nll = 0.5 * (math.log(2 * math.pi) + lv + (mu - mu.detach()) ** 2 / var)
        evidence = nu + alpha
        reg = evidence.clamp(min=0.0) * nll.abs()
        return self.evidential_coeff * reg.mean()

    def forward(self, x: torch.Tensor) -> EvidentialOutput:
        h         = self.encoder(x)
        mu        = torch.sigmoid(self.mu_head(h))
        lv        = self.lv_head(h).clamp(-6, 2)
        var       = torch.exp(lv)
        nu        = F.softplus(self.nu_head(h))    + 0.01
        alpha     = F.softplus(self.alpha_head(h)) + 1.01
        beta      = F.softplus(self.beta_head(h))  + 0.01
        epistemic = (beta / (nu * (alpha - 1.0))).clamp(0, 5.0)
        cal       = self._calibration_surprise(x)
        decay     = torch.exp(-(var + epistemic).clamp(0, 4.0))
        effective = (mu * decay * (1.0 - cal.clamp(0, 0.85))).clamp(0.05, 1.0)
        return EvidentialOutput(mu, var, epistemic, cal, effective, self._nig_loss(mu, lv, nu, alpha, beta))


# ══════════════════════════════════════════════════════════════════════
# FusionStats
# ══════════════════════════════════════════════════════════════════════

@dataclass
class FusionStats:
    gates:        torch.Tensor
    reliability:  Dict[str, torch.Tensor]
    aleatoric:    Dict[str, torch.Tensor]
    epistemic:    Dict[str, torch.Tensor]
    calibration:  Dict[str, torch.Tensor]
    effective:    Dict[str, torch.Tensor]
    nig_losses:   Dict[str, torch.Tensor]
    stream_names: List[str]

    def _mg(self, d): return {n: d[n].mean().item() for n in self.stream_names if n in d}
    def mean_gates(self):       return {n: self.gates[:,i].mean().item() for i,n in enumerate(self.stream_names)}
    def mean_reliability(self): return self._mg(self.reliability)
    def mean_aleatoric(self):   return self._mg(self.aleatoric)
    def mean_epistemic(self):   return self._mg(self.epistemic)
    def mean_calibration(self): return self._mg(self.calibration)
    def mean_effective(self):   return self._mg(self.effective)
    def total_nig_loss(self):
        losses = list(self.nig_losses.values())
        return sum(losses) if losses else torch.tensor(0.0)
    def uncertainty_summary(self):
        return {'gates': self.mean_gates(), 'reliability': self.mean_reliability(),
                'aleatoric': self.mean_aleatoric(), 'epistemic': self.mean_epistemic(),
                'effective': self.mean_effective()}


class _MinimalFusionStats:
    def __init__(self, gate_info, stream_names):
        self._gi = gate_info; self._sn = stream_names
    def mean_gates(self):       return self._gi.get('gates', {n: 1/len(self._sn) for n in self._sn})
    def mean_reliability(self): return self._gi.get('reliability', {n: 1.0 for n in self._sn})
    def mean_aleatoric(self):   return self._gi.get('aleatoric', {n: 0.0 for n in self._sn})
    def mean_epistemic(self):   return self._gi.get('epistemic', {n: 0.0 for n in self._sn})
    def mean_effective(self):   return self._gi.get('effective', {n: 1/len(self._sn) for n in self._sn})


# ══════════════════════════════════════════════════════════════════════
# AR-2: Parity Gate
# ══════════════════════════════════════════════════════════════════════

class ParityGate(nn.Module):
    def __init__(self, shared_dim, n_streams, context_dim=128, forensic_names=None):
        super().__init__()
        self.n_streams = n_streams
        self.mlp_f = nn.Sequential(nn.Linear(shared_dim, context_dim, bias=False), nn.GELU(), nn.Linear(context_dim, n_streams))
        self.mlp_s = nn.Sequential(nn.Linear(shared_dim, context_dim, bias=False), nn.GELU(), nn.Linear(context_dim, n_streams))
        self.alpha_raw = nn.Parameter(torch.tensor(0.25))  # sigmoid(0.25)≈0.56 → α≈0.52, forensic-leaning
        nn.init.zeros_(self.mlp_f[-1].weight); nn.init.zeros_(self.mlp_f[-1].bias)
        nn.init.zeros_(self.mlp_s[-1].weight); nn.init.zeros_(self.mlp_s[-1].bias)

    @property
    def alpha(self): return 0.3 + 0.4 * torch.sigmoid(self.alpha_raw)

    def forward(self, vit_eff, forensic_eff, sparse_k=None):
        fc = torch.stack(list(forensic_eff.values()), dim=1).mean(1) if forensic_eff else torch.zeros_like(vit_eff)
        logits = self.alpha * self.mlp_f(fc) + (1-self.alpha) * self.mlp_s(vit_eff)
        if sparse_k and sparse_k < self.n_streams:
            if self.training:
                soft = F.gumbel_softmax(logits, tau=0.5, hard=False)
                _, idx = soft.topk(sparse_k, 1)
                hard = torch.zeros_like(soft).scatter_(1, idx, 1.0)
                return hard - soft.detach() + soft
            else:
                p = torch.softmax(logits, 1)
                _, idx = p.topk(sparse_k, 1)
                g = torch.zeros_like(p).scatter_(1, idx, p.gather(1, idx))
                return g / (g.sum(1, keepdim=True) + 1e-8)
        return torch.softmax(logits, 1)


# ══════════════════════════════════════════════════════════════════════
# AR-4: Forensic Confidence Weighter
# ══════════════════════════════════════════════════════════════════════

class ForensicConfidenceWeighter(nn.Module):
    def __init__(self, dim, n_forensic, init_dampening=0.7):
        super().__init__()
        raw = math.log(init_dampening / (1.0 - init_dampening + 1e-8))
        self.dampening_raw = nn.Parameter(torch.tensor(raw))
        self.mu_proj    = nn.Linear(dim, dim, bias=False)
        self.sigma_proj = nn.Linear(dim, dim, bias=False)
        self.norm_mu    = nn.LayerNorm(dim)
        self.norm_sigma = nn.LayerNorm(dim)
        nn.init.orthogonal_(self.mu_proj.weight); nn.init.eye_(self.sigma_proj.weight)

    @property
    def dampening(self): return 0.1 + 0.9 * torch.sigmoid(self.dampening_raw)

    def forward(self, fs: torch.Tensor):
        mu    = self.norm_mu(self.mu_proj(fs))
        sigma = F.softplus(self.norm_sigma(self.sigma_proj(fs))) + 1e-4
        inv_v = 1.0 / (sigma**2 + 1e-8)
        nw    = inv_v / (inv_v.sum(1, keepdim=True) + 1e-8)
        return mu * nw * fs.shape[1] * self.dampening


# ══════════════════════════════════════════════════════════════════════
# AR-3: Iterative Semantic-Forensic Contradiction Reasoner
# ══════════════════════════════════════════════════════════════════════

class IterativeSFContradictionReasoner(nn.Module):
    def __init__(self, dim, n_forensic, n_iters=2, dropout=0.05):
        super().__init__()
        self.n_iters = n_iters; self.dim = dim
        nh = max(1, dim // 64)
        self.fsa_norm = nn.LayerNorm(dim)
        self.fsa = nn.MultiheadAttention(dim, nh, dropout=dropout, batch_first=True)
        self.csf_nq  = nn.LayerNorm(dim); self.csf_nkv = nn.LayerNorm(dim)
        self.csf = nn.MultiheadAttention(dim, nh, dropout=dropout, batch_first=True)
        self.cfs_nq  = nn.LayerNorm(dim); self.cfs_nkv = nn.LayerNorm(dim)
        self.cfs = nn.MultiheadAttention(dim, nh, dropout=dropout, batch_first=True)
        self.gru  = nn.GRUCell(dim, dim)
        self.gate = nn.Sequential(nn.Linear(dim*2, 1), nn.Sigmoid())
        self.ff_s = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*2, bias=False), nn.GELU(), nn.Linear(dim*2, dim, bias=False))
        self.ff_f = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, dim*2, bias=False), nn.GELU(), nn.Linear(dim*2, dim, bias=False))
        self.drop = nn.Dropout(dropout)

    def forward(self, vit: torch.Tensor, forensic: torch.Tensor):
        B = vit.shape[0]; v = vit; f = forensic  # f: (B, n_forensic, D)
        h = torch.zeros(B, self.dim, device=vit.device, dtype=vit.dtype)
        for _ in range(self.n_iters):
            fn = self.fsa_norm(f); fsa, _ = self.fsa(fn, fn, fn); f = f + self.drop(fsa)
            q = self.csf_nq(v).unsqueeze(1); kv = self.csf_nkv(f)
            ca, _ = self.csf(q, kv, kv); cs = self.drop(ca.squeeze(1))
            h = self.gru(cs, h)
            g = self.gate(torch.cat([v, h], 1)); v = v + g * h
            qf = self.cfs_nq(f); kvf = self.cfs_nkv(v).unsqueeze(1)
            caf, _ = self.cfs(qf, kvf, kvf); f = f + self.drop(caf)
        return v + self.ff_s(v), f + self.ff_f(f)


# ══════════════════════════════════════════════════════════════════════
# AR-2+3+4: Authenticity Reasoning Fusion
# ══════════════════════════════════════════════════════════════════════

class AuthenticityReasoningFusion(nn.Module):
    def __init__(self, stream_dims, shared_dim=256, context_dim=128,
                 n_iters=2, use_isfcr=True, use_fcw=True,
                 sparse_k=None, evidential_coeff=0.01):
        super().__init__()
        assert 'vit' in stream_dims
        self.stream_names   = list(stream_dims.keys())
        self.shared_dim     = shared_dim
        self.out_dim        = shared_dim
        self.use_isfcr      = use_isfcr
        self.use_fcw        = use_fcw
        self.sparse_k       = sparse_k
        self.forensic_names = [n for n in self.stream_names if n != 'vit']
        nf = len(self.forensic_names)

        self.align_mlps = nn.ModuleDict({
            n: nn.Sequential(nn.Linear(d, shared_dim, bias=False), nn.LayerNorm(shared_dim), nn.GELU())
            for n, d in stream_dims.items()})
        self.evid_heads = nn.ModuleDict({
            n: HeteroscedasticEvidentialHead(shared_dim, evidential_coeff)
            for n in self.stream_names})
        if use_isfcr and nf > 0:
            self.isfcr = IterativeSFContradictionReasoner(shared_dim, nf, n_iters)
        if use_fcw and nf > 0:
            self.fcw = ForensicConfidenceWeighter(shared_dim, nf, 0.7)
        self.parity_gate = ParityGate(shared_dim, len(self.stream_names), context_dim, self.forensic_names)
        self.out_norm    = nn.LayerNorm(shared_dim)
        self._ema_contrib = {n: 1.0/len(self.stream_names) for n in self.stream_names}
        self._ema_alpha   = 0.05

    def _update_ema(self, gates):
        mg = gates.detach().mean(0); a = self._ema_alpha
        for i, n in enumerate(self.stream_names):
            self._ema_contrib[n] = (1-a)*self._ema_contrib[n] + a*mg[i].item()

    def forward(self, streams):
        normed  = {n: F.normalize(streams[n].float(), dim=1, eps=1e-8) for n in self.stream_names}
        aligned = {n: self.align_mlps[n](normed[n]) for n in self.stream_names}
        evid    = {n: self.evid_heads[n](aligned[n]) for n in self.stream_names}
        feat_eff = {n: aligned[n] * evid[n].effective for n in self.stream_names}
        vit_eff  = feat_eff['vit']
        if self.use_isfcr and self.forensic_names and hasattr(self, 'isfcr'):
            fs = torch.stack([feat_eff[n] for n in self.forensic_names], 1)
            vit_r, for_r = self.isfcr(vit_eff, fs)
            attended = {'vit': vit_r}
            for i, n in enumerate(self.forensic_names): attended[n] = for_r[:, i, :]
        else:
            attended = feat_eff
        if self.use_fcw and self.forensic_names and hasattr(self, 'fcw'):
            fstack = torch.stack([attended[n] for n in self.forensic_names], 1)
            fw = self.fcw(fstack)
            for i, n in enumerate(self.forensic_names): attended[n] = fw[:, i, :]
        fed = {n: attended[n] for n in self.forensic_names}
        gates = self.parity_gate(attended['vit'], fed, self.sparse_k)
        if self.training: self._update_ema(gates)
        ss = torch.stack([attended[n] for n in self.stream_names], 1).clamp(-10, 10)
        fused = (ss * gates.unsqueeze(2)).sum(1)
        fused = torch.nan_to_num(fused, nan=0.0)
        fused = self.out_norm(fused)
        stats = FusionStats(
            gates       = gates.detach(),
            reliability = {n: evid[n].reliability.squeeze(1).detach() for n in self.stream_names},
            aleatoric   = {n: evid[n].aleatoric.squeeze(1).detach()   for n in self.stream_names},
            epistemic   = {n: evid[n].epistemic.squeeze(1).detach()   for n in self.stream_names},
            calibration = {n: evid[n].calibration.squeeze(1).detach() for n in self.stream_names},
            effective   = {n: evid[n].effective.squeeze(1).detach()   for n in self.stream_names},
            nig_losses  = {n: evid[n].nig_loss.detach()               for n in self.stream_names},
            stream_names= self.stream_names,
        )
        return fused, stats

    def get_gate_weights(self, streams):
        with torch.no_grad():
            _, stats = self.forward(streams)
        return stats.uncertainty_summary()

    def get_low_contribution_streams(self, threshold=0.03):
        return [n for n in self.stream_names if self._ema_contrib[n] < threshold]

    @property
    def use_cross_attn(self): return self.use_isfcr
    @property
    def reliability_heads(self): return self.evid_heads


# ══════════════════════════════════════════════════════════════════════
# Forensic Stream Modules
# ══════════════════════════════════════════════════════════════════════

def _make_srm_kernels_3x3(n=9):
    srm3 = [[[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]],[[-1, 2,-1],[ 2,-4, 2],[-1, 2,-1]],
             [[ 1,-2, 1],[-2, 4,-2],[ 1,-2, 1]],[[ 0, 0, 0],[-1, 2,-1],[ 0, 0, 0]],
             [[ 0,-1, 0],[ 0, 2, 0],[ 0,-1, 0]],[[-1, 0, 1],[ 0, 0, 0],[ 1, 0,-1]],
             [[ 1, 0,-1],[ 0, 0, 0],[-1, 0, 1]],[[ 0, 1, 0],[ 1,-4, 1],[ 0, 1, 0]],
             [[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]]]
    w = torch.zeros(n, 3, 3, 3)
    for i, k in enumerate(srm3[:n]):
        kf = torch.tensor(k, dtype=torch.float32) / 4.0
        for c in range(3): w[i, c] = kf
    if n > len(srm3): nn.init.kaiming_normal_(w[len(srm3):], mode='fan_out')
    return w

def _make_srm_kernels_5x5(n=30):
    bank = [[[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]],
            [[0,0,0,0,0],[0,-1,2,-1,0],[0,2,-4,2,0],[0,-1,2,-1,0],[0,0,0,0,0]],
            [[0,0,0,0,0],[0,1,-2,1,0],[0,-2,4,-2,0],[0,1,-2,1,0],[0,0,0,0,0]],
            [[0,0,0,0,0],[0,0,1,0,0],[0,1,-4,1,0],[0,0,1,0,0],[0,0,0,0,0]],
            [[0,0,0,0,0],[0,-1,0,1,0],[0,0,0,0,0],[0,1,0,-1,0],[0,0,0,0,0]],
            [[0,0,0,0,0],[0,1,0,-1,0],[0,0,0,0,0],[0,-1,0,1,0],[0,0,0,0,0]],
            [[0,0,0,0,0],[0,0,1,0,0],[0,-1,0,1,0],[0,0,-1,0,0],[0,0,0,0,0]],
            [[0,0,-1,0,0],[0,0,3,0,0],[-1,3,-8,3,-1],[0,0,3,0,0],[0,0,-1,0,0]],
            [[0,0,0,0,0],[0,0,-1,0,0],[0,-1,4,-1,0],[0,0,-1,0,0],[0,0,0,0,0]]]
    w = torch.zeros(n, 3, 5, 5)
    for i, k in enumerate(bank[:min(len(bank), n)]):
        kf = torch.tensor(k, dtype=torch.float32) / 4.0
        for c in range(3): w[i, c] = kf
    if n > len(bank): nn.init.kaiming_normal_(w[len(bank):], mode='fan_out')
    return w


def _make_srm_kernels_7x7(n_filters: int = 15) -> torch.Tensor:
    w = torch.zeros(n_filters, 3, 7, 7)
    center = torch.zeros(7, 7)
    for r in range(7):
        for c in range(7):
            dist = abs(r - 3) + abs(c - 3)
            if dist == 0:   center[r, c] = 24.0
            elif dist == 1: center[r, c] = -4.0
            elif dist == 2: center[r, c] = -1.0
    center /= 24.0
    for i in range(min(6, n_filters)):
        for ch in range(3): w[i, ch] = center * ((-1) ** i)
    if n_filters > 6:
        nn.init.kaiming_normal_(w[6:], mode='fan_out')
    return w


class MultiScaleSRM(nn.Module):
    def __init__(self, out_ch=64):
        super().__init__()
        self.srm3 = nn.Conv2d(3, 9,  3, padding=1, bias=False)
        self.srm5 = nn.Conv2d(3, 30, 5, padding=2, bias=False)
        self.srm7 = nn.Conv2d(3, 15, 7, padding=3, bias=False)
        with torch.no_grad():
            self.srm3.weight.copy_(_make_srm_kernels_3x3(9))
            self.srm5.weight.copy_(_make_srm_kernels_5x5(30))
            self.srm7.weight.copy_(_make_srm_kernels_7x7(15))
        self.fuse = nn.Sequential(nn.Conv2d(54, out_ch, 1, bias=False), nn.BatchNorm2d(out_ch), nn.GELU())

    def forward(self, x):
        return self.fuse(torch.cat([
            torch.tanh(self.srm3(x)),
            torch.tanh(self.srm5(x)),
            torch.tanh(self.srm7(x))], dim=1))


class PixelFrequencyHead(nn.Module):
    def __init__(self, in_ch=64, out_dim=128):
        super().__init__()
        self._pixel_attn = None
        self.score = nn.Sequential(nn.Conv2d(in_ch, 32, 3, padding=1, bias=False), nn.BatchNorm2d(32), nn.GELU(), nn.Conv2d(32, 1, 1))
        self.refine = nn.Sequential(DepthwiseSepConv(in_ch, in_ch, 3), DepthwiseSepConv(in_ch, out_dim, 3))
        self.norm   = nn.LayerNorm(out_dim)

    def forward(self, x):
        B, C, H, W = x.shape
        sc = self.score(x)
        a  = torch.softmax(sc.view(B, 1, -1), -1).view(B, 1, H, W)
        self._pixel_attn = a.detach()
        return self.norm(self.refine(x * a).mean([2, 3]))


class FrequencyStream(nn.Module):
    _LAP = torch.tensor([[0.,-1.,0.],[-1.,4.,-1.],[0.,-1.,0.]])

    def __init__(self, out_dim=128):
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer('laplacian', self._LAP.view(1,1,3,3).repeat(3,1,1,1))
        self.ms_srm     = MultiScaleSRM(out_ch=64)
        self.pixel_head = PixelFrequencyHead(64, out_dim//2)
        self.layer1 = DepthwiseSepConv(67, 64, 3, 2)
        self.layer2 = DepthwiseSepConv(64, 96, 3, 2)
        self.layer3 = DepthwiseSepConv(96, out_dim//2, 3, 2)
        self.norm_g = nn.LayerNorm(out_dim//2)
        self.fusion = nn.Sequential(nn.Linear(out_dim, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        self._feat_l1 = self._feat_l2 = self._feat_l3 = None

    @property
    def _pixel_attn(self): return self.pixel_head._pixel_attn

    def forward(self, x):
        xr  = (x * self._std + self._mean).clamp(0, 1)
        lap = F.conv2d(xr, self.laplacian, padding=1, groups=3).clamp(-1, 1)
        sf  = self.ms_srm(xr)
        p   = self.pixel_head(sf)
        f   = torch.cat([sf, lap], 1)
        f   = self.layer1(f); self._feat_l1 = f
        f   = self.layer2(f); self._feat_l2 = f
        f   = self.layer3(f); self._feat_l3 = f
        g   = self.norm_g(f.mean([2,3]))
        return self.fusion(torch.cat([p, g], 1))


class MultiScaleDCTBlock(nn.Module):
    _BS = [8, 16, 32]
    def __init__(self, proj_dim=96):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(4032, 192), nn.LayerNorm(192), nn.GELU(),
                                   nn.Linear(192, proj_dim), nn.LayerNorm(proj_dim), nn.GELU())
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        for bs in self._BS: self.register_buffer(f'_dct_{bs}', self._build_basis(bs))

    @staticmethod
    def _build_basis(N):
        n = torch.arange(N, dtype=torch.float32); k = n.clone()
        return torch.cos(math.pi/N * (n.unsqueeze(1)+0.5) * k.unsqueeze(0))

    def _dct2(self, x, bs):
        B, C, H, W = x.shape
        ph = (bs - H%bs)%bs; pw = (bs - W%bs)%bs
        if ph or pw: x = F.pad(x, (0, pw, 0, ph))
        blocks = x.unfold(2, bs, bs).unfold(3, bs, bs).contiguous().view(B, -1, bs, bs)
        basis  = getattr(self, f'_dct_{bs}')
        d2     = (basis.T @ (blocks @ basis).transpose(-2,-1)).transpose(-2,-1)
        return torch.log1p((d2**2).mean(1).view(B, bs*bs))

    def forward(self, x):
        xr = (x * self._std + self._mean).clamp(0, 1)
        r,g,b = xr[:,0:1], xr[:,1:2], xr[:,2:3]
        Y  =  0.299*r+0.587*g+0.114*b
        Cb = -0.1687*r-0.3313*g+0.5*b+0.5
        Cr =  0.5*r-0.4187*g-0.0813*b+0.5
        return self.proj(torch.cat([self._dct2(ch, bs) for ch in [Y,Cb,Cr] for bs in self._BS], 1))


class FFTPhaseStream(nn.Module):
    def __init__(self, out_dim=64, cw=7):
        super().__init__()
        self.net  = nn.Sequential(DepthwiseSepConv(2, 32, 3, 2), DepthwiseSepConv(32, out_dim, 3, 2))
        self.norm = nn.LayerNorm(out_dim)
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        c = torch.arange(cw, dtype=torch.float32) - cw//2; s = cw/3.0
        g = torch.exp(-0.5*(c/s)**2); g /= g.sum()
        self.register_buffer('_gauss', (g.unsqueeze(0)*g.unsqueeze(1)).view(1,1,cw,cw))

    def _coherence(self, ph):
        w = self._gauss.shape[-1]; pad = w//2; g = self._gauss
        sm = F.conv2d(F.pad(torch.sin(ph), [pad]*4, 'reflect'), g, padding=0)
        cm = F.conv2d(F.pad(torch.cos(ph), [pad]*4, 'reflect'), g, padding=0)
        return (sm**2 + cm**2).clamp(0).sqrt()

    def forward(self, x):
        xr   = (x * self._std + self._mean).clamp(0, 1)
        gray = 0.299*xr[:,0:1]+0.587*xr[:,1:2]+0.114*xr[:,2:3]
        fft  = torch.fft.fftshift(torch.fft.fft2(gray.squeeze(1)), dim=[-2,-1])
        rm   = torch.log1p(fft.abs()+1e-8).unsqueeze(1)
        B    = rm.shape[0]
        mag  = (rm - rm.view(B,-1).mean(1).view(B,1,1,1)) / (rm.view(B,-1).std(1).view(B,1,1,1)+1e-6)
        ph   = torch.nan_to_num(fft.angle(), nan=0.0)
        coh  = self._coherence(ph.unsqueeze(1))
        out  = self.norm(self.net(torch.cat([mag, coh], 1).clamp(-5, 5)).mean([2,3]))
        return torch.nan_to_num(out, nan=0.0)


class NoiseConsistencyBlock(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(15, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        c = torch.arange(7, dtype=torch.float32) - 3
        g = torch.exp(-0.5*(c/1.5)**2); g /= g.sum()
        self.register_buffer('_g1d', g)

    def _smooth(self, x):
        C = x.shape[1]; k = self._g1d; ks = k.shape[0]; pad = ks//2
        x = F.conv2d(F.pad(x,[pad,pad,0,0],'reflect'), k.view(1,1,1,ks).expand(C,1,1,ks), groups=C)
        return F.conv2d(F.pad(x,[0,0,pad,pad],'reflect'), k.view(1,1,ks,1).expand(C,1,ks,1), groups=C)

    def forward(self, x):
        xr  = (x * self._std + self._mean).clamp(0, 1)
        res = (xr - self._smooth(xr)).view(xr.shape[0], 3, -1)
        ns  = F.normalize(res, dim=2, eps=1e-6)
        feats = []
        for c in range(3):
            n = ns[:, c:c+1].view(xr.shape[0], 1, xr.shape[2], xr.shape[3])
            nm = n.mean([2,3]); nstd = n.std([2,3])
            feats.extend([
                (n[:,:,:,:-1]*n[:,:,:,1:]).mean([2,3]),
                (n[:,:,:-1,:]*n[:,:,1:,:]).mean([2,3]),
                (n[:,:,:-1,:-1]*n[:,:,1:,1:]).mean([2,3]),
                nstd,
                ((n-nm.unsqueeze(-1).unsqueeze(-1))**3).mean([2,3])/(nstd**3+1e-6).clamp(-10,10),
            ])
        return self.proj(torch.nan_to_num(torch.cat(feats, 1), nan=0.0))


class JPEGAwareBlock(nn.Module):
    def __init__(self, out_dim=32):
        super().__init__()
        self.out_dim = out_dim
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        # Project from raw grid scores (3) + extra stats (9) = 12 → out_dim
        self.proj = nn.Sequential(
            nn.Linear(12, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, out_dim), nn.LayerNorm(out_dim), nn.GELU()
        )

    def _grid(self, ch):
        B, _, H, W = ch.shape
        pw = ch.fft.fft2(ch.squeeze(1)).abs()**2 if False else torch.fft.fft2(ch.squeeze(1)).abs()**2
        t  = pw.mean([1,2]) + 1e-8; g = torch.zeros(B, device=ch.device)
        for hb in [H//8, 2*H//8, 3*H//8]:
            for wb in [W//8, 2*W//8, 3*W//8]:
                if 0<hb<H and 0<wb<W:
                    g += pw[:,hb,wb]+pw[:,H-hb,wb]+pw[:,hb,W-wb]
        return (g/(t*9+1e-8)).clamp(0,1).unsqueeze(1)

    def forward(self, x):
        xr = (x*self._std+self._mean).clamp(0,1)
        r,g_,b = xr[:,0:1],xr[:,1:2],xr[:,2:3]
        Y  =  0.299*r+0.587*g_+0.114*b
        Cb = -0.1687*r-0.3313*g_+0.5*b+0.5
        Cr =  0.5*r-0.4187*g_-0.0813*b+0.5
        gy = self._grid(Y); gcb = self._grid(Cb); gcr = self._grid(Cr)
        # Add channel-level DCT energy stats for richer JPEG fingerprint
        def _chan_stats(ch):
            pw = torch.fft.fft2(ch.squeeze(1)).abs()**2
            return torch.stack([pw.mean([-2,-1]), pw.std([-2,-1]),
                                 pw.max(dim=-1).values.max(dim=-1).values], dim=1)
        sy = _chan_stats(Y); scb = _chan_stats(Cb); scr = _chan_stats(Cr)
        feats = torch.cat([gy, gcb, gcr, sy, scb, scr], 1)  # (B, 12)
        return self.proj(torch.nan_to_num(feats.clamp(-10, 10), nan=0.0))


# ── NEW: Local Texture Coherence Stream ────────────────────────────────────────
# Detects GAN micro-texture field periodicity & conv upsampling tiling artifacts

class LocalTextureCoherenceStream(nn.Module):
    """
    Measures local texture field consistency. Real images have stochastic texture
    variation; GAN conv-upsampling produces periodic tiling and overly smooth
    micro-texture gradients that this stream exposes.
    """
    def __init__(self, out_dim=48):
        super().__init__()
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        # Spectral flatness in local windows
        self.proj = nn.Sequential(
            nn.Linear(32, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def _spectral_flatness(self, x):
        # x: (B, 1, H, W) — compute per-patch spectral flatness ratio
        B, _, H, W = x.shape
        ps = 16; stride = 16
        patches = x.unfold(2, ps, stride).unfold(3, ps, stride)  # (B,1,nh,nw,ps,ps)
        nh, nw  = patches.shape[2], patches.shape[3]
        p_flat = patches.contiguous().view(B, nh * nw, -1)
        # FFT over patch
        pf = torch.fft.rfft(p_flat, dim=-1).abs() + 1e-8
        # Geometric mean / arithmetic mean = flatness ∈ [0,1]
        gm = torch.exp(torch.log(pf).mean(-1))
        am = pf.mean(-1)
        flatness = (gm / (am + 1e-8)).clamp(0, 1)  # (B, nh*nw)
        return flatness

    def _autocorr_periodicity(self, x):
        # Detect periodic patterns via autocorrelation of gradient magnitudes
        B, _, H, W = x.shape
        gray = x.mean(1, keepdim=True)
        # Sobel
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=x.dtype, device=x.device).view(1,1,3,3)
        ky = kx.transpose(-1,-2)
        gx = F.conv2d(F.pad(gray, [1,1,1,1], 'reflect'), kx)
        gy = F.conv2d(F.pad(gray, [1,1,1,1], 'reflect'), ky)
        gm = (gx**2 + gy**2).sqrt()
        # Downsample for efficiency
        gm_small = F.avg_pool2d(gm, 8)  # (B,1,H//8,W//8)
        gm_f = torch.fft.rfft2(gm_small.squeeze(1))
        power = gm_f.abs()**2
        # Ratio of top-10 peaks to total power = periodicity score
        B2, H2, W2_h = power.shape
        flat = power.view(B2, -1)
        topk, _ = flat.topk(min(10, flat.shape[-1]), dim=-1)
        ratio = topk.sum(-1) / (flat.sum(-1) + 1e-8)
        return ratio.unsqueeze(1)  # (B, 1)

    def forward(self, x):
        xr = (x * self._std + self._mean).clamp(0, 1)
        sf = self._spectral_flatness(xr)  # (B, nh*nw)
        # Summary stats of spectral flatness distribution
        sf_mean = sf.mean(-1, keepdim=True)
        sf_std  = sf.std(-1, keepdim=True)
        sf_max  = sf.max(-1, keepdim=True).values
        sf_min  = sf.min(-1, keepdim=True).values
        # Inter-patch std ratio (GAN = low, real = high)
        sf_range = sf_max - sf_min

        per = self._autocorr_periodicity(xr)  # (B, 1)

        # Multi-channel feature
        feats = torch.cat([
            sf_mean, sf_std, sf_max, sf_min, sf_range,
            per,
            # Gradient magnitude spatial entropy
        ], dim=1)  # (B, 6)

        # Pad to 32 dims with patch quantile features
        q25 = sf.quantile(0.25, dim=-1, keepdim=True)
        q75 = sf.quantile(0.75, dim=-1, keepdim=True)
        iqr = q75 - q25
        skew = ((sf - sf_mean)**3).mean(-1, keepdim=True) / (sf_std**3 + 1e-6)
        kurt = ((sf - sf_mean)**4).mean(-1, keepdim=True) / (sf_std**4 + 1e-6) - 3

        feats = torch.cat([feats, q25, q75, iqr, skew, kurt], 1)  # (B, 11)

        # Add per-channel spectral flatness means
        for c in range(3):
            sf_c = self._spectral_flatness(xr[:,c:c+1])
            feats = torch.cat([feats, sf_c.mean(-1, keepdim=True), sf_c.std(-1, keepdim=True)], 1)
        # (B, 11+6 = 17) — pad to 32
        pad = torch.zeros(feats.shape[0], 32-feats.shape[1], device=x.device)
        feats = torch.cat([feats, pad], 1)
        return self.proj(torch.nan_to_num(feats, nan=0.0))


# ── NEW: Chromatic Aberration Stream ──────────────────────────────────────────
# Real camera lenses have chromatic aberration; GANs produce chromatically flat edges

class ChromaticAberrationStream(nn.Module):
    """
    Real lenses always produce lateral chromatic aberration (channel edge offset).
    2025-26 GANs render images without this physical constraint.
    This stream measures channel-edge alignment to expose GAN chromatic flatness.
    """
    def __init__(self, out_dim=32):
        super().__init__()
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        self.proj = nn.Sequential(
            nn.Linear(16, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, out_dim), nn.LayerNorm(out_dim), nn.GELU())
        # Sobel kernels
        kx = torch.tensor([[-1,0,1],[-2,0,2],[-1,0,1]], dtype=torch.float32).view(1,1,3,3)
        ky = kx.transpose(-1,-2)
        self.register_buffer('_kx', kx)
        self.register_buffer('_ky', ky)

    def _channel_edges(self, x_ch):
        gx = F.conv2d(F.pad(x_ch,[1,1,1,1],'reflect'), self._kx)
        gy = F.conv2d(F.pad(x_ch,[1,1,1,1],'reflect'), self._ky)
        return (gx**2 + gy**2).sqrt()

    def forward(self, x):
        xr = (x * self._std + self._mean).clamp(0, 1)
        r,g,b = xr[:,0:1], xr[:,1:2], xr[:,2:3]
        er = self._channel_edges(r)
        eg = self._channel_edges(g)
        eb = self._channel_edges(b)

        # Cross-channel edge correlation (GAN: high, real: lower due to CA)
        rg_corr = (er * eg).mean([2,3]) / (er.std([2,3]) * eg.std([2,3]) + 1e-6)
        rb_corr = (er * eb).mean([2,3]) / (er.std([2,3]) * eb.std([2,3]) + 1e-6)
        gb_corr = (eg * eb).mean([2,3]) / (eg.std([2,3]) * eb.std([2,3]) + 1e-6)

        # Channel edge magnitude ratios
        r_mag = er.mean([2,3]); g_mag = eg.mean([2,3]); b_mag = eb.mean([2,3])
        rg_rat = r_mag / (g_mag + 1e-6); rb_rat = r_mag / (b_mag + 1e-6)
        gb_rat = g_mag / (b_mag + 1e-6)

        # Channel edge peak offset (sub-pixel CA measurement proxy)
        def _peak_loc(e):
            B, _, H, W = e.shape
            flat = e.view(B, -1)
            idx  = flat.argmax(-1).float()
            return torch.stack([idx / W, idx % W], -1) / torch.tensor([H, W], device=e.device, dtype=torch.float32)

        pr = _peak_loc(er); pg = _peak_loc(eg); pb = _peak_loc(eb)
        rg_off = (pr - pg).norm(dim=-1, keepdim=True)
        rb_off = (pr - pb).norm(dim=-1, keepdim=True)
        gb_off = (pg - pb).norm(dim=-1, keepdim=True)

        feats = torch.cat([
            rg_corr, rb_corr, gb_corr,
            rg_rat, rb_rat, gb_rat,
            rg_off, rb_off, gb_off,
            r_mag, g_mag, b_mag,
            (r_mag - b_mag).abs(), (r_mag - g_mag).abs(), (g_mag - b_mag).abs(),
            # CA variance asymmetry: edges in periphery vs center
            (er[:,:,er.shape[2]//4:3*er.shape[2]//4, er.shape[3]//4:3*er.shape[3]//4]).mean([2,3]),
        ], 1)  # (B, 16)

        return self.proj(torch.nan_to_num(feats.clamp(-10, 10), nan=0.0))


# ── NEW: Spectral Flatness & GAN Upsampling Artifact Stream ───────────────────

class SpectralFlatnessStream(nn.Module):
    """
    GANs produce overly smooth frequency spectra (spectral flatness ≈ 1 in mid-freq).
    Also detects bilinear/bicubic upsampling grid artifacts at 1/stride frequency.
    """
    def __init__(self, out_dim=48):
        super().__init__()
        self.register_buffer('_mean', torch.tensor(CLIP_MEAN).view(1,3,1,1))
        self.register_buffer('_std',  torch.tensor(CLIP_STD).view(1,3,1,1))
        self.proj = nn.Sequential(
            nn.Linear(24, 64), nn.LayerNorm(64), nn.GELU(),
            nn.Linear(64, out_dim), nn.LayerNorm(out_dim), nn.GELU())

    def _radial_spectrum(self, gray):
        B, H, W = gray.shape
        fft  = torch.fft.fftshift(torch.fft.fft2(gray), dim=[-2,-1])
        mag  = torch.log1p(fft.abs())
        cy, cx = H//2, W//2
        # Radial bins
        ys, xs = torch.meshgrid(
            torch.arange(H, device=gray.device) - cy,
            torch.arange(W, device=gray.device) - cx, indexing='ij')
        r = (ys**2 + xs**2).float().sqrt()
        max_r = min(cy, cx)
        n_bins = 16
        feats = []
        for i in range(n_bins):
            lo = i * max_r / n_bins; hi = (i+1) * max_r / n_bins
            mask = ((r >= lo) & (r < hi)).float()
            ring_mag = (mag * mask.unsqueeze(0)).sum([-2,-1]) / (mask.sum() + 1e-6)
            feats.append(ring_mag.unsqueeze(-1))
        return torch.cat(feats, -1)  # (B, 16)

    def _upsampling_artifact_score(self, gray):
        B, H, W = gray.shape
        fft  = torch.fft.fft2(gray)
        mag  = fft.abs()
        # Check for grid artifacts at multiples of 1/stride
        score = torch.zeros(B, 4, device=gray.device)
        for i, stride in enumerate([2, 4, 8, 16]):
            if H >= stride*4 and W >= stride*4:
                hi = H // stride; wi = W // stride
                if 0 < hi < H and 0 < wi < W:
                    score[:, i] = mag[:, hi, wi] / (mag.mean([-2,-1]) + 1e-8)
        return score  # (B, 4)

    def forward(self, x):
        xr   = (x * self._std + self._mean).clamp(0, 1)
        gray = (0.299*xr[:,0]+0.587*xr[:,1]+0.114*xr[:,2])  # (B,H,W)
        rs   = self._radial_spectrum(gray)         # (B, 16)
        ua   = self._upsampling_artifact_score(gray)  # (B, 4)
        feats = torch.cat([rs, ua], -1)            # (B, 20)
        # Trim to 24 with padding
        pad   = torch.zeros(feats.shape[0], 4, device=x.device)
        feats = torch.cat([feats, pad], -1)
        return self.proj(torch.nan_to_num(feats.clamp(-20, 20), nan=0.0))


# ── Patch Attention Pool ───────────────────────────────────────────────────────

class PatchAttentionPool(nn.Module):
    def __init__(self, dim, hidden=128):
        super().__init__()
        self.attn = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, hidden), nn.Tanh(), nn.Linear(hidden, 1, bias=False))
    def forward(self, x):
        w = torch.softmax(self.attn(x), 1)
        return (w * x).sum(1), w.squeeze(-1)


class CrossLevelArtifactAttention(nn.Module):
    def __init__(self, dim, num_heads=4, dropout=0.1):
        super().__init__()
        self.nq  = nn.LayerNorm(dim); self.nkv = nn.LayerNorm(dim)
        self.attn= nn.MultiheadAttention(dim, num_heads, dropout=dropout, batch_first=True, add_bias_kv=True)
        self.proj= nn.Linear(dim, dim)
        self.drop= nn.Dropout(dropout)
        self.gmlp= nn.Sequential(nn.Linear(dim,32),nn.GELU(),nn.Linear(32,3))

    def forward(self, cls, ep, mp, fp):
        g  = torch.softmax(self.gmlp(cls), -1)
        kv = torch.cat([g[:,0:1].unsqueeze(2)*ep, g[:,1:2].unsqueeze(2)*mp, g[:,2:3].unsqueeze(2)*fp], 1)
        q  = self.nq(cls).unsqueeze(1)
        o,_= self.attn(q, self.nkv(kv), self.nkv(kv))
        return self.drop(self.proj((o+q).squeeze(1)))


class GeneratorSignatureHead(nn.Module):
    def __init__(self, in_f=256, n=8):
        super().__init__()
        self.head = nn.Sequential(nn.Linear(in_f,128),nn.GELU(),nn.Dropout(0.15),nn.Linear(128,n))
    def forward(self, x): return self.head(x)


class WarmupCosineClassifier(nn.Module):
    def __init__(self, in_f, nc, scale_start=8., scale_end=15., warmup_steps=1000):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(nc, in_f))
        self.scale  = nn.Parameter(torch.tensor(scale_start))
        self.scale.requires_grad = False
        self.fake_bias = nn.Parameter(torch.tensor(0.0))
        self.scale_start = scale_start; self.scale_end = scale_end
        self.warmup_steps = warmup_steps
        self.register_buffer('_step', torch.tensor(0))
        nn.init.orthogonal_(self.weight)

    def step_warmup(self):
        if self._step.item() < self.warmup_steps:
            f = self._step.item() / self.warmup_steps
            self.scale.data.fill_(self.scale_start + f*(self.scale_end-self.scale_start))
            self._step += 1
        else:
            self.scale.requires_grad = True

    def unfreeze_scale(self): self.scale.requires_grad = True

    def adapt_bias(self, delta):
        # Clamped to [-1.0, 1.0] — tighter than [-1.5, 1.5], collapse-safe
        with torch.no_grad():
            self.fake_bias.data.add_(delta).clamp_(-1.0, 1.0)

    def forward(self, x):
        xn = F.normalize(x, 1); wn = F.normalize(self.weight, 1)
        logits = self.scale * (xn @ wn.T)
        bias   = torch.stack([self.fake_bias, torch.zeros(1, device=x.device).squeeze()], 0)
        return torch.clamp(logits + bias.unsqueeze(0), -8., 8.)


# ══════════════════════════════════════════════════════════════════════
# AR-5: Authenticity Consistency Loss
# ══════════════════════════════════════════════════════════════════════

class AuthenticityConsistencyLoss(nn.Module):
    def __init__(self, w_agreement=0.05, w_parity=0.03, w_nig=1.0, w_freq=0.01, parity_threshold=0.3):
        super().__init__()
        self.wa = w_agreement; self.wp = w_parity
        self.wn = w_nig;       self.wf = w_freq
        self.pt = parity_threshold

    def forward(self, logits, labels, fs: FusionStats, pixel_attn):
        dev = logits.device
        eff = torch.stack(list(fs.effective.values()), 1)
        is_fake = (labels==0).float(); is_real = (labels==1).float()
        std = eff.std(1)
        L_agree = (F.relu(std-0.1)*is_real).mean() - (eff.min(1).values*is_fake).mean()
        L_agree = L_agree.clamp(-1.5, 1.5)

        vit_e = fs.effective.get('vit', eff[:,0])
        for_e = torch.stack([v for k,v in fs.effective.items() if k!='vit'], 1).mean(1) if len(fs.stream_names)>1 else vit_e
        L_parity = (F.relu(vit_e - for_e - self.pt) * is_fake).mean()

        L_nig = fs.total_nig_loss()
        try:
            if not L_nig.requires_grad: L_nig = torch.tensor(0., device=dev)
        except: L_nig = torch.tensor(0., device=dev)

        if pixel_attn is not None:
            rc = torch.softmax(logits, 1)[:,1]
            ma = pixel_attn.view(pixel_attn.shape[0],-1).max(1).values
            L_freq = self.wf * (F.relu(ma-0.7)*rc).mean()
        else:
            L_freq = torch.tensor(0., device=dev)

        total = self.wa*L_agree + self.wp*L_parity + self.wn*L_nig + L_freq
        return total, {'L_agree': L_agree.item(), 'L_parity': L_parity.item(),
                       'L_nig': L_nig.item() if hasattr(L_nig,'item') else 0., 'L_freq': L_freq.item()}


FrequencyConsistencyLoss = AuthenticityConsistencyLoss


# ══════════════════════════════════════════════════════════════════════
# AR-6: Deep Postprocessing Invariance Augmentation (17 types)
# New types cover 2025-26 GAN evasion: latent upsampling, perceptual
# hash attack, style injection, content-aware blur, CycleGAN ISP sim
# ══════════════════════════════════════════════════════════════════════

class DeepPostprocessingAug(nn.Module):
    def __init__(self, freq_aug_prob=0.20, jpeg_sim_prob=0.20, blur_prob=0.15,
                 sharpen_prob=0.12, resize_prob=0.15, noise_prob=0.15,
                 chroma_prob=0.12, diffusion_prob=0.12, isp_prob=0.08,
                 thumbnail_prob=0.08, chain_prob=0.15, chain_len=2,
                 noise_scale=0.015, blur_sigma_max=1.5, resize_min_frac=0.6,
                 # 2025-26 aug probs
                 style_jitter_prob=0.08,
                 latent_grid_prob=0.06,
                 local_blur_prob=0.08,
                 tone_map_prob=0.08):
        super().__init__()
        self.probs = dict(
            freq_aug=freq_aug_prob, jpeg_sim=jpeg_sim_prob, blur=blur_prob,
            sharpen=sharpen_prob, resize=resize_prob, noise=noise_prob,
            chroma=chroma_prob, diffusion=diffusion_prob, isp=isp_prob,
            thumbnail=thumbnail_prob, style_jitter=style_jitter_prob,
            latent_grid=latent_grid_prob, local_blur=local_blur_prob,
            tone_map=tone_map_prob)
        self.chain_prob = chain_prob; self.chain_len = chain_len
        self.noise_scale = noise_scale; self.blur_sigma_max = blur_sigma_max
        self.resize_min_frac = resize_min_frac

    def _spectral_noise(self, x):
        n = torch.randn_like(x) * self.noise_scale
        s = F.avg_pool2d(F.pad(n,[1,1,1,1],'reflect'),3,1,0)
        return (x+n-s).clamp(-3,3)

    def _jpeg_simulate(self, x):
        H,W = x.shape[-2:]
        fft = torch.fft.rfft2(x); fh,fw = fft.shape[-2:]
        kh = max(1,int(fh*(0.4+0.5*random.random())))
        kw = max(1,int(fw*(0.4+0.5*random.random())))
        mask = torch.zeros_like(fft.real); mask[...,:kh,:kw]=1.
        return torch.fft.irfft2(fft*mask,s=(H,W)).clamp(-3,3)

    def _gaussian_blur(self, x):
        sigma = random.uniform(0.3, self.blur_sigma_max)
        ks = max(3, int(sigma*3)*2+1); C = x.shape[1]
        c  = torch.arange(ks, dtype=torch.float32, device=x.device) - ks//2
        g1 = torch.exp(-0.5*(c/sigma)**2); g1 /= g1.sum()
        kh = g1.view(1,1,1,ks).expand(C,1,1,ks)
        x  = F.conv2d(F.pad(x,[ks//2,ks//2,0,0],'reflect'), kh, groups=C)
        kv = g1.view(1,1,ks,1).expand(C,1,ks,1)
        return F.conv2d(F.pad(x,[0,0,ks//2,ks//2],'reflect'), kv, groups=C).clamp(-3,3)

    def _unsharp_sharpen(self, x): return (x+random.uniform(0.3,1.2)*(x-self._gaussian_blur(x))).clamp(-3,3)

    def _resize_perturb(self, x):
        H,W = x.shape[-2:]; frac=random.uniform(self.resize_min_frac,0.95)
        H2,W2=max(32,int(H*frac)),max(32,int(W*frac))
        m1=random.choice(['bilinear','bicubic','nearest-exact'])
        kw1={} if m1=='nearest-exact' else {'align_corners':False}
        m2=random.choice(['bilinear','bicubic'])
        return F.interpolate(F.interpolate(x,(H2,W2),mode=m1,**kw1),(H,W),mode=m2,align_corners=False).clamp(-3,3)

    def _gaussian_noise(self, x): return (x+torch.randn_like(x)*random.uniform(0.005,0.04)).clamp(-3,3)

    def _chroma_shift(self, x):
        s = (torch.rand(1,x.shape[1],1,1,device=x.device)-0.5)*0.12
        return (x+s).clamp(-3,3)

    def _diffusion_patch_perturb(self, x):
        H,W=x.shape[-2:]; sc=random.uniform(0.01,0.04)
        ph,pw=max(1,H//8),max(1,W//8)
        n=torch.randn(1,x.shape[1],ph,pw,device=x.device)*sc
        return (x+F.interpolate(n,(H,W),mode='bilinear',align_corners=False)).clamp(-3,3)

    def _isp_simulate(self, x):
        gain = 1.+(torch.rand(1,x.shape[1],1,1,device=x.device)-0.5)*0.15

        dm = F.interpolate(
            torch.randn(
                1,
                x.shape[1],
                x.shape[2]//4,
                x.shape[3]//4,
                device=x.device
        ) * 0.01,

            size=x.shape[-2:],
            mode='nearest'
        )

        return (x * gain + dm).clamp(-3, 3)

    def _thumbnail_crop_pad(self, x):
        H,W=x.shape[-2:]; frac=random.uniform(0.7,0.95)
        ch,cw=int(H*frac),int(W*frac); y1=random.randint(0,H-ch); x1=random.randint(0,W-cw)
        return F.interpolate(
            x[:, :, y1:y1+ch, x1:x1+cw],
            size=(H, W),
            mode='bilinear',
            align_corners=False
            ).clamp(-3, 3)

    def _style_jitter(self, x):
        """Simulate style-transfer color/contrast jitter used by modern GANs to match photo look."""
        B,C,H,W = x.shape
        # Random per-image affine colour transform (contrast + brightness per channel)
        alpha = 1.0 + (torch.rand(B,C,1,1,device=x.device)-0.5)*0.3  # contrast
        beta  = (torch.rand(B,C,1,1,device=x.device)-0.5)*0.15        # brightness
        return (alpha*x + beta).clamp(-3,3)

    def _latent_grid_artifact(self, x):
        """
        Simulate the checkerboard/grid artifacts left by deconvolution/bilinear upsampling
        in latent diffusion models. Creates low-amplitude grid noise at stride-4 frequency.
        """
        H,W = x.shape[-2:]; stride=random.choice([4,8,16])
        grid = torch.zeros_like(x)
        amp  = random.uniform(0.005, 0.02)
        grid[:,:,::stride,::stride] = amp
        grid[:,:,::stride,1::stride] = -amp/2
        grid[:,:,1::stride,::stride] = -amp/2
        return (x + grid).clamp(-3,3)

    def _local_content_blur(self, x):
        """Simulate GAN face/object-aware region sharpening — blur background, keep subject sharp."""
        H,W = x.shape[-2:]
        # Blur a random rectangular region (simulating GAN depth-of-field synthesis)
        blurred = self._gaussian_blur(x)
        mask = torch.zeros(1,1,H,W,device=x.device)
        y1=random.randint(H//4,H//2); y2=random.randint(H//2,3*H//4)
        x1=random.randint(W//4,W//2); x2=random.randint(W//2,3*W//4)
        mask[:,:,y1:y2,x1:x2] = 1.0
        # Feather mask
        mask = F.avg_pool2d(F.pad(mask,[8,8,8,8],'reflect'),17,1,0)
        return (x*(1-mask) + blurred*mask).clamp(-3,3)

    def _tone_map_simulate(self, x):
        """Simulate HDR tone mapping used by diffusion models to achieve hyper-realistic lighting."""
        # S-curve tone mapping
        gamma = random.uniform(0.7, 1.4)
        xr    = (x + 3.0) / 6.0  # remap to [0,1]
        xr    = xr.clamp(0,1)**gamma
        return (xr*6.0 - 3.0).clamp(-3,3)

    def _get_all_ops(self):
        return [
            ('freq_aug',     self._spectral_noise),
            ('jpeg_sim',     self._jpeg_simulate),
            ('blur',         self._gaussian_blur),
            ('sharpen',      self._unsharp_sharpen),
            ('resize',       self._resize_perturb),
            ('noise',        self._gaussian_noise),
            ('chroma',       self._chroma_shift),
            ('diffusion',    self._diffusion_patch_perturb),
            ('isp',          self._isp_simulate),
            ('thumbnail',    self._thumbnail_crop_pad),
            ('style_jitter', self._style_jitter),
            ('latent_grid',  self._latent_grid_artifact),
            ('local_blur',   self._local_content_blur),
            ('tone_map',     self._tone_map_simulate),
        ]

    def forward(self, x):
        if not self.training: return x
        out = x.clone(); hw = out.shape[-2:]
        all_ops = self._get_all_ops()
        for i in range(x.shape[0]):
            def fix(t, fn):
                a = fn(t)
                if a.shape[-2:] != hw: a = F.interpolate(a, hw, mode='bilinear', align_corners=False)
                return a
            if random.random() < self.chain_prob:
                for _, fn in random.sample(all_ops, min(self.chain_len, len(all_ops))):
                    out[i:i+1] = fix(out[i:i+1], fn)
            else:
                for name, fn in all_ops:
                    if random.random() < self.probs.get(name, 0.0):
                        out[i:i+1] = fix(out[i:i+1], fn)
        return out


RobustAugWrapper = DeepPostprocessingAug
FreqAugmentWrapper = DeepPostprocessingAug


# ══════════════════════════════════════════════════════════════════════
# AR-7: Adaptive Stream Router
# ══════════════════════════════════════════════════════════════════════

class AdaptiveStreamRouter:
    EMA_ALPHA = 0.05

    def __init__(self, stream_names):
        self.stream_names = list(stream_names)
        n = len(stream_names)
        self._ema = {
            'gates':      {s: 1./n for s in stream_names},
            'reliability':{s: 1.0  for s in stream_names},
            'aleatoric':  {s: 0.0  for s in stream_names},
            'epistemic':  {s: 0.0  for s in stream_names},
            'effective':  {s: 1./n for s in stream_names},
        }
        self._var_gates = {s: 0.0 for s in stream_names}
        self._step = 0; self._pruned = set()

    def update(self, stats):
        self._step += 1; a = self.EMA_ALPHA
        g=stats.mean_gates(); r=stats.mean_reliability()
        al=stats.mean_aleatoric(); ep=stats.mean_epistemic(); ef=stats.mean_effective()
        for n in self.stream_names:
            og = self._ema['gates'][n]; ng = g.get(n, og)
            self._ema['gates'][n]       = (1-a)*og + a*ng
            self._ema['reliability'][n] = (1-a)*self._ema['reliability'][n] + a*r.get(n,1.)
            self._ema['aleatoric'][n]   = (1-a)*self._ema['aleatoric'][n]   + a*al.get(n,0.)
            self._ema['epistemic'][n]   = (1-a)*self._ema['epistemic'][n]   + a*ep.get(n,0.)
            self._ema['effective'][n]   = (1-a)*self._ema['effective'][n]   + a*ef.get(n,1./len(self.stream_names))
            self._var_gates[n]          = (1-a)*self._var_gates[n] + a*(ng-self._ema['gates'][n])**2

    def get_low_contribution_streams(self, threshold=0.03):
        return [n for n in self.stream_names if n not in self._pruned and self._ema['effective'][n] < threshold]

    def prune_stream(self, name):
        if name in self.stream_names and name != 'vit':
            self._pruned.add(name); print(f"  [Router] '{name}' pruned.")

    def get_routing_report(self):
        lines = [f"AdaptiveStreamRouter (step {self._step})",
                 f"{'Stream':<14}{'Gate':>8}{'Relbl':>8}{'Aleat':>8}{'Epist':>8}{'Effec':>8}{'Status':>10}",
                 "-"*64]
        for n in self.stream_names:
            st = "PRUNED" if n in self._pruned else ("LOW" if self._ema['effective'][n] < 0.05 else "OK")
            lines.append(f"  {n:<12}{self._ema['gates'][n]:>8.4f}{self._ema['reliability'][n]:>8.4f}"
                         f"{self._ema['aleatoric'][n]:>8.4f}{self._ema['epistemic'][n]:>8.4f}"
                         f"{self._ema['effective'][n]:>8.4f}{st:>10}")
        return "\n".join(lines)

    def get_summary(self):
        return {'step': self._step, 'gate_ema': {n: round(v,4) for n,v in self._ema['gates'].items()},
                'effective': {n: round(v,4) for n,v in self._ema['effective'].items()},
                'pruned': list(self._pruned)}


StreamImportanceTracker = AdaptiveStreamRouter


# ── Error Learning Memory ──────────────────────────────────────────────────────

class ErrorLearningMemory:
    EMA_ALPHA = 0.02  # slow/stable: ~50-step half-life

    def __init__(self, max_records=500):
        self.max_records = max_records
        self.records = []; self.success_records = []
        self.error_stats   = {'real_as_fake':[], 'fake_as_real':[], 'error_patterns':defaultdict(int)}
        self.success_stats = {'real_correct':[], 'fake_correct':[], 'success_patterns':defaultdict(int)}
        self._ema_fake_err = 0.0; self._ema_real_err = 0.0
        self._err_stream_norms: Dict[str,List[float]] = defaultdict(list)
        self._success_stream_norms: Dict[str,List[float]] = defaultdict(list)

    def record(self, true_label, pred_label, confidence, stream_norms=None, image_path=''):
        lm = {0:'FAKE', 1:'REAL'}
        if pred_label == true_label:
            st = 'fake_correct' if true_label==0 else 'real_correct'
            self.success_stats[st].append(confidence)
            if stream_norms:
                valid = {k:v for k,v in stream_norms.items() if isinstance(v,(int,float))}
                if valid:
                    self.success_stats['success_patterns'][f"{lm[true_label]}_ok_{max(valid,key=valid.get)}"] += 1
                    for k, v in valid.items():
                        self._success_stream_norms[k].append(v)
            self.success_records.append({'true':lm[true_label],'confidence':round(confidence,4)})
            if len(self.success_records) > self.max_records: self.success_records.pop(0)
            return
        et = f"{lm[true_label]}_as_{lm[pred_label]}"; key = et.lower().replace('-','_')
        if key in self.error_stats: self.error_stats[key].append(confidence)
        if stream_norms:
            valid = {k:v for k,v in stream_norms.items() if isinstance(v,(int,float))}
            if valid:
                dom = max(valid, key=valid.get)
                self.error_stats['error_patterns'][f"{et}_{dom}"] += 1
                for k,v in valid.items(): self._err_stream_norms[k].append(v)
        ife = (true_label==0 and pred_label==1)
        self._ema_fake_err = (1-self.EMA_ALPHA)*self._ema_fake_err + self.EMA_ALPHA*float(ife)
        self._ema_real_err = (1-self.EMA_ALPHA)*self._ema_real_err + self.EMA_ALPHA*float(not ife)
        self.records.append({'true':lm[true_label],'pred':lm[pred_label],'confidence':round(confidence,4)})
        if len(self.records) > self.max_records: self.records.pop(0)

    def get_loss_weights(self, base_weights):
        w = list(base_weights)
        # Collapse guard: if errors are near-total, trust base weights
        if self._ema_fake_err < 0.8:
            w[0] = min(base_weights[0] * (1. + 0.20 * self._ema_fake_err), base_weights[0] * 1.8)
        if self._ema_real_err < 0.8:
            w[1] = max(base_weights[1], min(base_weights[1] * (1. + 0.40 * self._ema_real_err), base_weights[1] * 2.0))
        return w

    def suggest_bias_adjustment(self):
        fe=self._ema_fake_err; re=self._ema_real_err
        # Don't adjust bias during collapse (when one class error is near-total)
        if fe > 0.85 or re > 0.85: return 0.0
        return float(0.05 * np.clip((fe-re)/(fe+re+1e-8), -1, 1))

    def summary(self):
        rw=self.error_stats['real_as_fake']; fw=self.error_stats['fake_as_real']
        return {'total_errors': len(self.records), 'total_successes': len(self.success_records),
                'real_as_fake': len(rw), 'fake_as_real': len(fw),
                'ema_fake_err_rate': round(self._ema_fake_err,4), 'ema_real_err_rate': round(self._ema_real_err,4),
                'avg_conf_real_err': round(float(sum(rw)/max(len(rw),1)),4),
                'avg_conf_fake_err': round(float(sum(fw)/max(len(fw),1)),4),
                'top_error_patterns': dict(sorted(self.error_stats['error_patterns'].items(),key=lambda x:x[1],reverse=True)[:5])}

    def save(self, path):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({
                'records':         self.records[-100:],
                'success_records': self.success_records[-100:],
                'summary':         self.summary()
            }, f, indent=2)

    def load(self, path):
        if not os.path.exists(path): return
        with open(path) as f: d = json.load(f)
        self.records         = d.get('records', [])
        self.success_records = d.get('success_records', [])


class ConfidenceHistoryTracker:
    def __init__(self, window_size=500, shift_threshold=0.22):
        self.ws=window_size; self.thr=shift_threshold
        self._hist: deque = deque(maxlen=window_size)
        self._baseline = None; self._shift_cnt = 0

    def update(self, conf): self._hist.append(conf)
    def check_shift(self):
        if len(self._hist) < 50: return {'shift_detected':False}
        recent = float(np.mean(list(self._hist)[-50:]))
        if len(self._hist) == self.ws and self._baseline is None: self._baseline = float(np.mean(self._hist))
        if self._baseline is None: return {'shift_detected':False,'recent_mean':round(recent,4)}
        d = abs(recent - self._baseline); sd = d > self.thr
        if sd:
            self._shift_cnt += 1
            if self._shift_cnt >= 3: self._baseline = recent; self._shift_cnt = 0
        else: self._shift_cnt = 0
        return {'shift_detected':sd,'baseline':round(self._baseline,4),'recent_mean':round(recent,4),'delta':round(d,4),
                'alert':'Distribution shift detected' if sd else 'Stable'}
    def reset_baseline(self): self._baseline = float(np.mean(self._hist)) if self._hist else None


# ── Backbone loader ────────────────────────────────────────────────────────────

def _load_backbone():
    vram = _vram_gb()
    if vram >= 10: cands=[('ViT-L-14',1024),('ViT-B-16',512)]
    elif vram >= 3.5: cands=[('ViT-B-16',512),('ViT-B-32',512)]
    else: cands=[('ViT-B-32',512)]
    print(f"  VRAM {vram:.1f}GB → candidates: {[c[0] for c in cands]}")
    try:
        import open_clip
        for mid,edim in cands:
            try:
                m,_,_=open_clip.create_model_and_transforms(mid,pretrained='openai')
                print(f"  Loaded {mid} open_clip (embed={edim})")
                return m, edim, f'openclip_{mid}'
            except Exception as e: print(f"  {mid}: {e}")
    except ImportError: pass
    try:
        import clip as oc
        nm={'ViT-B-32':'ViT-B/32','ViT-B-16':'ViT-B/16','ViT-L-14':'ViT-L/14'}
        for mid,edim in cands:
            try:
                m,_=oc.load(nm.get(mid,'ViT-B/32'),device='cpu')
                print(f"  Loaded {nm[mid]} openai/clip (embed={edim})")
                return m, edim, f'openai_{mid}'
            except: pass
    except ImportError: pass
    from torchvision.models import vit_b_16, ViT_B_16_Weights
    m = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    print("  Loaded torchvision ViT-B/16 fallback (embed=768)")
    return m, 768, 'torchvision_vitb16'


class MultiScaleCLIPExtractor(nn.Module):
    def __init__(self, model, embed_dim, backbone_type):
        super().__init__()
        self.model=model; self.embed_dim=embed_dim; self.backbone_type=backbone_type
        self._early=self._mid=self._final=None; self._handles=[]
        self._register_hooks()

    def _mkhook(self, attr):
        def h(m,i,o):
            t=o[0] if isinstance(o,tuple) else o
            if t.dim()==3 and t.shape[0]>t.shape[1]: t=t.permute(1,0,2).contiguous()
            setattr(self, attr, t)
        return h

    def _register_hooks(self):
        try:
            if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
                blocks=list(self.model.visual.transformer.resblocks.children())
            elif self.backbone_type=='torchvision_vitb16':
                blocks=list(self.model.encoder.layers.children())
            else: return
            n=len(blocks); ei=min(3,n-3); mi=min(8,n-2); fi=n-1
            self._handles=[blocks[ei].register_forward_hook(self._mkhook('_early')),
                           blocks[mi].register_forward_hook(self._mkhook('_mid')),
                           blocks[fi].register_forward_hook(self._mkhook('_final'))]
            print(f"  Multi-scale hooks: block {ei}+{mi}+{fi}")
        except Exception as e: print(f"  Hook failed ({e})")

    def forward(self, x):
        self._early=self._mid=self._final=None
        if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
            cls = self.model.encode_image(x)
        elif self.backbone_type=='torchvision_vitb16':
            x2=self.model._process_input(x)
            ct=self.model.class_token.expand(x2.shape[0],-1,-1)
            x2=torch.cat([ct,x2],1)+self.model.encoder.pos_embedding
            x2=self.model.encoder.ln(self.model.encoder.layers(self.model.encoder.dropout(x2)))
            cls=x2[:,0]
        else:
            o=self.model(x); cls=o[0] if isinstance(o,tuple) else o
            if cls.dim()>2: cls=cls[:,0]
        if cls.dim()==1: cls=cls.unsqueeze(0)
        cls=cls[:,:self.embed_dim]
        def _safe(h):
            if h is None: return cls.unsqueeze(1)
            p=h[:,1:]; return p[...,:self.embed_dim] if p.shape[-1]!=self.embed_dim else p
        return cls, _safe(self._early), _safe(self._mid), _safe(self._final)

    def __del__(self):
        for h in self._handles:
            try: h.remove()
            except: pass


# ══════════════════════════════════════════════════════════════════════
# Loss Functions
# ══════════════════════════════════════════════════════════════════════

class AsymmetricFocalLoss(nn.Module):
    """
    Focal loss with per-class gamma + REAL-class floor gradient.
    gamma_fake=2.0, gamma_real=2.5 — harder push on missed REALs.
    real_floor_weight: constant BCE floor for every REAL sample.
    """
    def __init__(self, gamma_fake=2.0, gamma_real=2.5, weight=None,
                 label_smoothing=0.03, real_floor_weight=0.05):
        super().__init__()
        self.gf = gamma_fake; self.gr = gamma_real
        self.weight = weight; self.ls = label_smoothing
        self.rfw = real_floor_weight

    def forward(self, logits, targets):
        ce  = F.cross_entropy(logits, targets, weight=self.weight, label_smoothing=self.ls, reduction='none')
        pt  = torch.exp(-ce.clamp(max=80.))
        g   = torch.where(targets==0, torch.full_like(ce, self.gf), torch.full_like(ce, self.gr))
        foc = (1.-pt)**g * ce
        # ForensicsAI: unconditional floor BCE on all REAL samples
        # This is the key — no underconfidence threshold, constant gradient always flows
        if self.rfw > 0:
            real_mask  = (targets == 1).float()
            real_probs = torch.softmax(logits, dim=1)[:, 1].clamp(1e-6, 1.0 - 1e-6)
            floor_bce  = -torch.log(real_probs) * real_mask
            foc = foc + self.rfw * floor_bce
        return foc.mean()

    def update_weight(self, w):
        dev = self.weight.device if self.weight is not None else torch.device('cpu')
        self.weight = w.to(dev)


class FocalLoss(AsymmetricFocalLoss):
    def __init__(self, gamma=2., weight=None, label_smoothing=0.03):
        super().__init__(gamma_fake=gamma, gamma_real=gamma, weight=weight, label_smoothing=label_smoothing)


# ── Adversarial Utilities ─────────────────────────────────────────────────────

class AdversarialTester:
    def __init__(self, model, device): self.model=model; self.device=device

    @torch.enable_grad()
    def fgsm(self, images, labels, eps=8/255):
        images=images.clone().detach().requires_grad_(True).to(self.device)
        labels=labels.to(self.device); self.model.eval()
        F.cross_entropy(self.model(images),labels).backward()
        with torch.no_grad(): adv=(images+eps*images.grad.sign()).clamp(-3,3)
        return adv.detach()

    @torch.enable_grad()
    def pgd(self, images, labels, eps=8/255, alpha=2/255, steps=7):
        images=images.to(self.device); labels=labels.to(self.device)
        adv=(images.clone().detach()+torch.empty_like(images).uniform_(-eps,eps)).clamp(-3,3)
        self.model.eval()
        for _ in range(steps):
            adv=adv.detach().requires_grad_(True)
            F.cross_entropy(self.model(adv),labels).backward()
            with torch.no_grad():
                adv=torch.min(torch.max(adv+alpha*adv.grad.sign(),images-eps),images+eps).clamp(-3,3)
        return adv.detach()

    @torch.enable_grad()
    def cw_l2(self, images, labels, c=1e-3, steps=20, lr=0.01):
        """Carlini-Wagner L2 attack (light version for eval)."""
        images=images.to(self.device); labels=labels.to(self.device)
        delta=torch.zeros_like(images, requires_grad=True)
        opt  =torch.optim.Adam([delta], lr=lr)
        self.model.eval()
        for _ in range(steps):
            adv = (images + delta).clamp(-3,3)
            logits = self.model(adv)
            target_logits = logits.gather(1, labels.unsqueeze(1)).squeeze(1)
            other_logits  = (logits - 1e9*F.one_hot(labels, logits.shape[1])).max(1).values
            loss_adv = F.relu(other_logits - target_logits + 0.01).mean()
            loss_reg = c * delta.norm(p=2)
            (loss_adv + loss_reg).backward()
            opt.step(); opt.zero_grad()
            delta.data.clamp_(-0.1, 0.1)
        return (images + delta).clamp(-3,3).detach()

    def evaluate(self, val_loader, attack='pgd', eps=8/255, max_batches=10):
        self.model.eval()
        all_preds=[]; all_labels=[]; all_orig=[]
        for i,(imgs,labels,*_) in enumerate(val_loader):
            if i >= max_batches: break
            imgs=imgs.to(self.device); labels=labels.to(self.device)
            if attack=='fgsm': adv=self.fgsm(imgs,labels,eps)
            elif attack=='pgd': adv=self.pgd(imgs,labels,eps)
            elif attack=='cw':  adv=self.cw_l2(imgs,labels)
            else: adv=self.fgsm(imgs,labels,eps)
            with torch.no_grad():
                ap=self.model(adv).argmax(1); op=self.model(imgs).argmax(1)
            all_preds.extend(ap.cpu()); all_labels.extend(labels.cpu()); all_orig.extend(op.cpu())
        ap=np.array(all_preds); al=np.array(all_labels); ao=np.array(all_orig)
        adv_acc  = (ap==al).mean()*100
        orig_acc = (ao==al).mean()*100
        bal_drop = orig_acc - adv_acc
        return {'adv_acc':adv_acc,'orig_acc':orig_acc,'bal_drop':bal_drop,
                'verdict':'ROBUST' if bal_drop<10 else 'VULNERABLE' if bal_drop>25 else 'MODERATE'}


# ── MC-Dropout Uncertainty ─────────────────────────────────────────────────────

def predict_with_uncertainty(model, image, n_passes=10):
    was = model.training; model.train()
    with torch.no_grad():
        ll = [model(image) for _ in range(n_passes)]
    model.train(was)
    stacked=torch.stack(ll); probs=torch.softmax(stacked,-1)
    mp=probs.mean(0); sp=probs.std(0)
    return {'prediction':mp.argmax(1),'confidence':mp.max(1).values,
            'uncertainty':sp.max(1).values,'mean_probs':mp}


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(state, path):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp=path+'.tmp'; torch.save(state,tmp); os.replace(tmp,path)

def load_checkpoint(path, map_location='cpu'):
    if not os.path.exists(path): return {}
    try: return torch.load(path,map_location=map_location,weights_only=False)
    except Exception as e: print(f"  [WARN] Could not load {path}: {e}"); return {}


# ── MixUp / CutMix ────────────────────────────────────────────────────────────

def mixup_data(x, y, alpha=0.2):
    if alpha<=0: return x,y,y,1.0
    lam=np.random.beta(alpha,alpha); idx=torch.randperm(x.size(0),device=x.device)
    return lam*x+(1-lam)*x[idx], y, y[idx], lam

def cutmix_data(x, y, alpha=0.5):
    if alpha<=0: return x,y,y,1.0
    lam=np.random.beta(alpha,alpha); B,C,H,W=x.shape
    idx=torch.randperm(B,device=x.device); cr=math.sqrt(1-lam)
    cw,ch=int(W*cr),int(H*cr); cx,cy=random.randint(0,W),random.randint(0,H)
    x1,x2=max(cx-cw//2,0),min(cx+cw//2,W); y1,y2=max(cy-ch//2,0),min(cy+ch//2,H)
    mix=x.clone(); mix[:,:,y1:y2,x1:x2]=x[idx,:,y1:y2,x1:x2]
    return mix, y, y[idx], 1-(x2-x1)*(y2-y1)/(W*H)

def mixup_criterion(criterion, logits, ya, yb, lam):
    return lam*criterion(logits,ya)+(1-lam)*criterion(logits,yb)


# ══════════════════════════════════════════════════════════════════════
# GradCAM
# ══════════════════════════════════════════════════════════════════════

class GradCAM:
    def __init__(self, model):
        self.model=model
        self._g1=self._a1=self._g2=self._a2=self._g3=self._a3=None
        self._handles=[]

    def _register(self):
        for ln,g,a in [('layer1','_g1','_a1'),('layer2','_g2','_a2'),('layer3','_g3','_a3')]:
            tgt=getattr(self.model.freq_stream,ln)
            def mh(g_,a_):
                def fh(m,i,o): setattr(self,a_,o.detach())
                def bh(m,gi,go):
                    if go[0] is not None: setattr(self,g_,go[0].detach())
                return fh,bh
            fh,bh=mh(g,a)
            self._handles+=[tgt.register_forward_hook(fh),tgt.register_full_backward_hook(bh)]

    def _remove(self):
        for h in self._handles:
            try: h.remove()
            except: pass
        self._handles.clear()

    @staticmethod
    def _norm(c): c=c.clamp(0); return (c-c.min())/(c.max()-c.min()+1e-8)

    def _lcam(self,g,a,sz):
        if g is None or a is None: return None
        w=g.mean([2,3],keepdim=True); cam=(w*a).sum(1)[0]
        return F.interpolate(self._norm(cam).unsqueeze(0).unsqueeze(0),(sz,sz),'bilinear',align_corners=False).squeeze().cpu()

    @torch.enable_grad()
    def generate(self, image, class_idx=1, size=224):
        self.model.eval(); self._g1=self._a1=self._g2=self._a2=self._g3=self._a3=None
        self._register()
        try:
            logits=self.model(image.clone()); self.model.zero_grad()
            logits[0,class_idx].backward()
            cams=[(self._lcam(self._g1,self._a1,size),0.5),(self._lcam(self._g2,self._a2,size),0.3),(self._lcam(self._g3,self._a3,size),0.2)]
            valid=[(c,w) for c,w in cams if c is not None]
            if not valid: return torch.ones(size,size)*0.5
            tw=sum(w for _,w in valid)
            return self._norm(sum(c*(w/tw) for c,w in valid))
        finally: self._remove()

    def generate_pixel_attn_cam(self, image: torch.Tensor, size: int = 224) -> torch.Tensor:
        self.model.eval()
        with torch.no_grad():
            _ = self.model(image)
        attn = self.model.freq_stream.pixel_head._pixel_attn
        if attn is None:
            return torch.ones(size, size) * 0.5
        return self._norm(F.interpolate(
            attn[0:1], size=(size, size),
            mode='bilinear', align_corners=False).squeeze()).cpu()

    @torch.enable_grad()
    def generate_overlay(self, image, class_idx=1, size=224,
                         freq_weight=0.6, pixel_weight=0.4) -> dict:
        fc = self.generate(image, class_idx, size)
        pc = self.generate_pixel_attn_cam(image, size)
        fused = freq_weight * fc + pixel_weight * pc
        return {
            'freq_cam':  fc,
            'pixel_cam': pc,
            'fused_cam': self._norm(fused)
        }


# ══════════════════════════════════════════════════════════════════════
# MAIN MODEL — ForensicEngine (SpatialRobustModel v3)
# ══════════════════════════════════════════════════════════════════════

class ForensicEngine(nn.Module):
    """
    ForensicEngine v3 — Production AI Image Authenticity Detection.

    New in v3 vs v2:
    - LocalTextureCoherenceStream: GAN upsampling tiling/periodicity detection
    - ChromaticAberrationStream: lens CA vs GAN chromatic flatness
    - SpectralFlatnessStream: GAN over-smooth spectra + upsampling grid artifacts
    - PGD + CW adversarial training (multi-attack hardening)
    - Improved 17-type augmentation covering 2025-26 GAN evasion techniques
    - Stabilized EMA-based adaptive routing and class weighting
    - REAL-class floor gradient + adaptive floor boosting
    """

    def __init__(self, num_classes=2, dropout=0.25, drop_path_rate=0.08,
                 temperature=1.0, freeze_backbone=True, unfreeze_last_n=0,
                 use_grad_checkpoint=True, input_size=224,
                 freq_dim=128, dct_dim=96, fft_dim=64, noise_dim=32,
                 use_generator_head=True,
                 use_srm=True, use_fft=True, use_dct=True, use_noise=True,
                 use_ltc=True, use_ca=True, use_sfs=True,
                 fusion_shared_dim=256, fusion_context_dim=128,
                 use_isfcr=True, use_fcw=True, n_iters=2,
                 sparse_k=None, evidential_coeff=0.01):
        super().__init__()
        self.input_size=input_size; self.unfrozen_count=0
        self.freq_dim=freq_dim; self.dct_dim=dct_dim
        self.fft_dim=fft_dim; self.noise_dim=noise_dim
        self.use_srm=use_srm; self.use_fft=use_fft; self.use_dct=use_dct
        self.use_noise=use_noise; self.use_ltc=use_ltc; self.use_ca=use_ca; self.use_sfs=use_sfs
        self.use_fcw=use_fcw; self.use_generator_head=use_generator_head
        self.register_buffer('temperature', torch.tensor(temperature, dtype=torch.float32))

        raw, embed_dim, btype = _load_backbone()
        self.embed_dim=embed_dim; self.backbone_type=btype
        self.extractor = MultiScaleCLIPExtractor(raw, embed_dim, btype)

        if freeze_backbone:
            for p in self.extractor.model.parameters(): p.requires_grad=False
            if unfreeze_last_n > 0: self._unfreeze_last_n(unfreeze_last_n)
        if use_grad_checkpoint: self._enable_grad_checkpoint()

        self.early_pool = PatchAttentionPool(embed_dim, 128)
        self.mid_pool   = PatchAttentionPool(embed_dim, 128)
        self.final_pool = PatchAttentionPool(embed_dim, 128)
        self.cross_attn = CrossLevelArtifactAttention(embed_dim, 4, dropout/3)
        self.drop_path  = DropPath(drop_path_rate)

        # Forensic streams
        if use_srm:   self.freq_stream  = FrequencyStream(freq_dim)
        if use_dct:   self.dct_block    = MultiScaleDCTBlock(dct_dim)
        if use_fft:   self.fft_stream   = FFTPhaseStream(fft_dim)
        if use_noise: self.noise_block  = NoiseConsistencyBlock(noise_dim)
        self.jpeg_block = JPEGAwareBlock()
        # New v3 streams
        if use_ltc:   self.ltc_stream   = LocalTextureCoherenceStream(out_dim=48)
        if use_ca:    self.ca_stream    = ChromaticAberrationStream(out_dim=32)
        if use_sfs:   self.sfs_stream   = SpectralFlatnessStream(out_dim=48)

        vit_total = embed_dim * 4
        stream_dims = {'vit': vit_total}
        if use_srm:   stream_dims['srm']   = freq_dim
        if use_dct:   stream_dims['dct']   = dct_dim
        if use_fft:   stream_dims['fft']   = fft_dim
        if use_noise: stream_dims['noise'] = noise_dim
        stream_dims['jpeg'] = 32
        if use_ltc:   stream_dims['ltc']  = 48
        if use_ca:    stream_dims['ca']   = 32
        if use_sfs:   stream_dims['sfs']  = 48

        self.stream_fusion = AuthenticityReasoningFusion(
            stream_dims, fusion_shared_dim, fusion_context_dim,
            n_iters, use_isfcr, use_fcw, sparse_k, evidential_coeff)

        fd = self.stream_fusion.out_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(fd), nn.Linear(fd, 512), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(512, 256), nn.GELU(), nn.Dropout(dropout/2))
        self.classifier = WarmupCosineClassifier(256, num_classes, 8., 15., 1000)

        if use_generator_head:
            self.generator_head = GeneratorSignatureHead(256, len(GENERATOR_LABELS))

        self.authenticity_loss = AuthenticityConsistencyLoss(
            w_agreement=0.02, w_parity=0.02, w_nig=1.0, w_freq=0.01)
        self._init_weights()
        self.error_memory       = ErrorLearningMemory(500)
        self.confidence_tracker = ConfidenceHistoryTracker(200)
        self.stream_router      = AdaptiveStreamRouter(list(stream_dims.keys()))

        tr = sum(p.numel() for p in self.parameters() if p.requires_grad)
        tt = sum(p.numel() for p in self.parameters())
        print(f"  ForensicEngine v3 — {tt:,} total | {tr:,} trainable ({100*tr/tt:.1f}%)")
        print(f"  Streams: {list(stream_dims.keys())}")

    def _unfreeze_last_n(self, n):
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
        except AttributeError as e: print(f"  Unfreeze failed: {e}")

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
            print(f"  Grad checkpoint: {cnt} blocks")
        except AttributeError as e: print(f"  Grad checkpoint skipped: {e}")

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None: nn.init.zeros_(m.bias)

    def _resize(self, x):
        if x.shape[-2]!=self.input_size or x.shape[-1]!=self.input_size:
            x=F.interpolate(x,(self.input_size,self.input_size),mode='bilinear',align_corners=False)
        return x

    def _build_streams(self, x, ep, mp, fp, co):
        s = {'vit': torch.cat([ep, mp, fp, co], 1)}
        if self.use_srm:   s['srm']   = self.freq_stream(x)
        if self.use_dct:   s['dct']   = self.dct_block(x)
        if self.use_fft:   s['fft']   = self.fft_stream(x)
        if self.use_noise: s['noise'] = self.noise_block(x)
        s['jpeg'] = self.jpeg_block(x)
        if self.use_ltc:   s['ltc']  = self.ltc_stream(x)
        if self.use_ca:    s['ca']   = self.ca_stream(x)
        if self.use_sfs:   s['sfs']  = self.sfs_stream(x)
        return s

    def _forward_core(self, x):
        x = self._resize(x)
        if not torch.isfinite(x).all(): x=torch.nan_to_num(x,nan=0.,posinf=3.,neginf=-3.)
        cls, early_p, mid_p, final_p = self.extractor(x)
        ep,ew = self.early_pool(early_p)
        mp,mw = self.mid_pool(mid_p)
        fp,fw = self.final_pool(final_p)
        vc    = self.drop_path(torch.cat([ep,mp,fp],1))
        ep=vc[:,:self.embed_dim]; mp=vc[:,self.embed_dim:2*self.embed_dim]; fp=vc[:,2*self.embed_dim:]
        co    = self.cross_attn(cls, early_p, mid_p, final_p)
        st    = self._build_streams(x, ep, mp, fp, co)
        fused, stats = self.stream_fusion(st)
        feat  = self.mlp(torch.nan_to_num(fused, nan=0.))
        return feat, st, stats, (ew, mw, fw)

    def forward(self, x):
        feat,_,_,_ = self._forward_core(x)
        return self.classifier(feat) / self.temperature

    def forward_with_entropy(self, x, labels=None):
        feat, streams, stats, (ew,mw,fw) = self._forward_core(x)
        logits = self.classifier(feat) / self.temperature
        e_ent = -(ew*(ew+1e-8).log()).sum(1).mean()
        m_ent = -(mw*(mw+1e-8).log()).sum(1).mean()
        f_ent = -(fw*(fw+1e-8).log()).sum(1).mean()
        pa    = self.freq_stream._pixel_attn if self.use_srm else None
        if labels is not None:
            aux, comp = self.authenticity_loss(logits, labels, stats, pa)
        else:
            nig=stats.total_nig_loss(); aux=nig
            comp={'L_nig': nig.item() if hasattr(nig,'item') else 0.}
        comp.update({'e_ent':e_ent.item(),'m_ent':m_ent.item(),'f_ent':f_ent.item()})
        return logits, aux, comp

    def forward_with_streams(self, x):
        feat, streams, stats, (ew,mw,fw) = self._forward_core(x)
        logits = self.classifier(feat) / self.temperature
        # Always update router for monitoring; it has no effect on model parameters
        self.stream_router.update(stats)
        info={k: v.norm(1).mean().item() for k,v in streams.items() if k!='jpeg'}
        info.update({
            'jpeg_strength_mean': streams['jpeg'].mean().item(),
            'early_attn_entropy': -(ew*(ew+1e-8).log()).sum(1).mean().item(),
            'mid_attn_entropy':   -(mw*(mw+1e-8).log()).sum(1).mean().item(),
            'final_attn_entropy': -(fw*(fw+1e-8).log()).sum(1).mean().item(),
            '_gate_weights': stats.mean_gates(), '_reliability': stats.mean_reliability(),
            '_aleatoric': stats.mean_aleatoric(), '_epistemic': stats.mean_epistemic(),
            '_calibration': stats.mean_calibration(), '_effective': stats.mean_effective(),
            '_parity_alpha': self.stream_fusion.parity_gate.alpha.item(),
        })
        if self.use_fcw and hasattr(self.stream_fusion,'fcw'):
            info['_forensic_dampening'] = self.stream_fusion.fcw.dampening.item()
        for old,new in [('srm','freq_out'),('fft','fft_out'),('dct','dct_out'),('noise','noise_out'),('vit','cross_out')]:
            if old in info: info[new]=info.pop(old)
        return logits, info

    def forward_with_generator(self, x):
        feat,_,_,_ = self._forward_core(x)
        bl = self.classifier(feat)/self.temperature
        gl = self.generator_head(feat) if self.use_generator_head else None
        return bl, gl

    def predict(self, x):
        self.eval()
        with torch.no_grad():
            feat, streams, stats, _ = self._forward_core(x)
            logits = self.classifier(feat)/self.temperature
            probs  = torch.softmax(logits, 1)
            ep     = list(stats.mean_epistemic().values())
            eff    = list(stats.mean_effective().values())
            et     = torch.tensor(eff)
        return {'prediction': probs.argmax(1), 'confidence': probs.max(1).values,
                'uncertainty': float(sum(ep)/max(len(ep),1)),
                'stream_importance': stats.mean_gates(), 'stream_reliability': stats.mean_reliability(),
                'contradiction_strength': float(et.std()) if len(eff)>1 else 0.}

    def get_explainability_report(self, x):
        self.eval()
        with torch.no_grad():
            feat, streams, stats, _ = self._forward_core(x)
            logits = self.classifier(feat)/self.temperature
        return {'stream_contribution': stats.mean_gates(), 'stream_reliability': stats.mean_reliability(),
                'stream_uncertainty': stats.mean_epistemic(), 'stream_effective': stats.mean_effective(),
                'parity_alpha': self.stream_fusion.parity_gate.alpha.item(),
                'contradiction_strength': float(torch.tensor(list(stats.mean_effective().values())).std()),
                'routing_report': self.stream_router.get_routing_report()}

    def set_temperature(self, t): self.temperature.fill_(t)

    def calibrate(self, val_loader, device, temp_range=(0.5,3.0)):
        try: from scipy.optimize import minimize_scalar
        except ImportError: print("scipy missing — skip calibration"); return 1.0
        self.eval(); self.set_temperature(1.0)
        ll,labs=[],[]
        with torch.no_grad():
            for batch in val_loader:
                ll.append(self.forward(batch[0].to(device)).cpu())
                labs.append(batch[1])
        al=torch.cat(ll); ab=torch.cat(labs)
        res=minimize_scalar(lambda T:F.cross_entropy(al/float(T),ab).item(),bounds=temp_range,method='bounded')
        T=float(np.clip(res.x,*temp_range)); self.set_temperature(T)
        print(f"  Calibrated T={T:.4f}"); return T

    def step_error_memory(self, true_labels, pred_labels, confs, stream_norms):
        for tl,pl,c in zip(true_labels,pred_labels,confs):
            self.error_memory.record(int(tl),int(pl),float(c),stream_norms)
            self.confidence_tracker.update(float(c))
        delta=self.error_memory.suggest_bias_adjustment()
        if abs(delta)>0.015: self.classifier.adapt_bias(delta)

    def training_step_hook(self): self.classifier.step_warmup()

    def get_stream_importance(self, x):
        with torch.no_grad():
            x=self._resize(x)
            cls,ep,mp,fp=self.extractor(x)
            ep_p,_=self.early_pool(ep); mp_p,_=self.mid_pool(mp); fp_p,_=self.final_pool(fp)
            co=self.cross_attn(cls,ep,mp,fp)
            st=self._build_streams(x,ep_p,mp_p,fp_p,co)
            gi=self.stream_fusion.get_gate_weights(st)
        self.stream_router.update(_MinimalFusionStats(gi, self.stream_fusion.stream_names))
        return {**gi, 'routing_summary':self.stream_router.get_summary()}

    def get_routing_report(self):   return self.stream_router.get_routing_report()
    def get_low_importance_streams(self, threshold=0.03): return self.stream_router.get_low_contribution_streams(threshold)
    def prune_stream(self, name):   self.stream_router.prune_stream(name)


# Backward-compat aliases
SpatialRobustModel = ForensicEngine


def get_spatial_robust_model(**kwargs) -> ForensicEngine:
    return ForensicEngine(**kwargs)

def get_forensic_engine(**kwargs) -> ForensicEngine:
    return ForensicEngine(**kwargs)

def get_baseline_model(freeze_backbone=True, **kwargs) -> ForensicEngine:
    return ForensicEngine(freeze_backbone=freeze_backbone, use_dct=False, use_fft=False,
                          use_noise=False, use_ltc=False, use_ca=False, use_sfs=False,
                          use_generator_head=False, **kwargs)
