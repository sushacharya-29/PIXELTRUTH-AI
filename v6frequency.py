import os
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple

# ── DCT basis (precomputed, no grad) ──────────────────────────────────────────

def _build_dct_basis(N: int) -> torch.Tensor:
    n = torch.arange(N, dtype=torch.float32)
    k = torch.arange(N, dtype=torch.float32)
    basis = torch.cos(math.pi / N * (n.unsqueeze(1) + 0.5) * k.unsqueeze(0))
    basis[:, 0] *= math.sqrt(1.0 / N)
    basis[:, 1:] *= math.sqrt(2.0 / N)
    return basis  # (N, N)


# ── SRM fixed filter bank (30 high-pass kernels, 5×5) ─────────────────────────
# Classic steganalysis kernels — known to expose GAN/diffusion artifacts

def _srm_kernels() -> torch.Tensor:
    K = torch.zeros(30, 1, 5, 5)

    # 3×3 difference filters (center-surround)
    spd3 = [
        [[ 0,-1, 0],[-1, 4,-1],[ 0,-1, 0]],  # Laplacian
        [[-1,-1,-1],[-1, 8,-1],[-1,-1,-1]],  # 8-neighbor
        [[ 1, 0,-1],[ 0, 0, 0],[-1, 0, 1]],  # diagonal
        [[ 0, 1, 0],[ 1,-4, 1],[ 0, 1, 0]],  # neg Laplacian
        [[ 1,-2, 1],[-2, 4,-2],[ 1,-2, 1]],  # 2nd order
    ]
    for i, k3 in enumerate(spd3):
        t = torch.tensor(k3, dtype=torch.float32)
        K[i, 0, 1:4, 1:4] = t

    # 5×5 high-pass kernels
    h5 = torch.tensor([
        [-1, 2,-2, 2,-1],
        [ 2,-6, 8,-6, 2],
        [-2, 8,-12,8,-2],
        [ 2,-6, 8,-6, 2],
        [-1, 2,-2, 2,-1],
    ], dtype=torch.float32)
    K[5, 0] = h5 / 12.0

    # Horizontal/vertical edge detectors
    for i, (dx, dy) in enumerate([(1,0),(0,1),(1,1),(1,-1)]):
        m = torch.zeros(5,5)
        for r in range(5):
            for c in range(5):
                d = abs(r*dy + c*dx)
                m[r,c] = 1.0 if d == 0 else (-1.0 / max(d, 1))
        m -= m.mean()
        K[6+i, 0] = m / (m.abs().sum() + 1e-8)

    # Gradient-difference filters
    for i in range(10):
        f = torch.randn(5, 5)
        f -= f.mean()
        K[10+i, 0] = f / (f.abs().sum() + 1e-8)

    # Wavelet-style QMF filters
    h = torch.tensor([-1, 2, 6, 2, -1], dtype=torch.float32) / 10.0
    for i in range(10):
        angle = math.pi * i / 10
        c, s = math.cos(angle), math.sin(angle)
        fk = h.unsqueeze(0) * math.cos(angle) + h.unsqueeze(1) * math.sin(angle)
        fk = fk[:5, :5] if fk.shape == (5,5) else F.interpolate(
            fk.unsqueeze(0).unsqueeze(0), (5,5)).squeeze()
        fk -= fk.mean()
        K[20+i, 0] = fk / (fk.abs().sum() + 1e-8)

    return K  # (30, 1, 5, 5)


# ── 2D DCT spectral features ──────────────────────────────────────────────────

class DCTSpectralBlock(nn.Module):
    def __init__(self, block_size: int = 8, proj_dim: int = 128):
        super().__init__()
        self.block_size = block_size
        basis = _build_dct_basis(block_size)
        self.register_buffer('basis', basis)

        # Multi-resolution: 8×8 (64) + 16×16 (256) → project to proj_dim
        self.proj = nn.Sequential(
            nn.Linear(64 + 64, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
            nn.Linear(proj_dim, proj_dim),
            nn.LayerNorm(proj_dim),
        )

        self.proj16 = nn.Sequential(
            nn.Linear(256, 64),
            nn.GELU(),
        )

    def _dct2d(self, x: torch.Tensor, bs: int) -> torch.Tensor:
        B, _, H, W = x.shape
        ph = (bs - H % bs) % bs
        pw = (bs - W % bs) % bs
        if ph or pw:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, H2, W2 = x.shape
        blocks = x.unfold(2, bs, bs).unfold(3, bs, bs)
        nH, nW = blocks.shape[2], blocks.shape[3]
        blocks = blocks.contiguous().view(B, nH * nW, bs, bs)

        b = self.basis if bs == 8 else _build_dct_basis(bs).to(x.device)
        dct = torch.einsum('bnik,kj->bnij', blocks, b)
        dct = torch.einsum('ki,bnij->bnkj', b, dct)

        energy = (dct ** 2).mean(dim=1)
        return torch.log1p(energy).view(B, bs * bs)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        gray = 0.299 * x[:, 0:1] + 0.587 * x[:, 1:2] + 0.114 * x[:, 2:3]
        f8   = self._dct2d(gray, 8)    # (B, 64)
        f16  = self._dct2d(gray, 16)   # (B, 256)
        f16  = self.proj16(f16)        # (B, 64)
        return self.proj(torch.cat([f8, f16], dim=1))  # (B, proj_dim)


# ── Wavelet decomposition feature extractor ───────────────────────────────────
# Haar DWT captures checkerboard artifacts (GAN upconv) across frequency bands

class HaarWaveletBlock(nn.Module):
    def __init__(self, in_ch: int = 3, proj_dim: int = 64):
        super().__init__()
        # Haar low/high pass
        ll = torch.tensor([[ 1, 1],[ 1, 1]], dtype=torch.float32) * 0.5
        lh = torch.tensor([[ 1, 1],[-1,-1]], dtype=torch.float32) * 0.5
        hl = torch.tensor([[ 1,-1],[ 1,-1]], dtype=torch.float32) * 0.5
        hh = torch.tensor([[ 1,-1],[-1, 1]], dtype=torch.float32) * 0.5

        kernels = torch.stack([ll, lh, hl, hh], dim=0)  # (4,2,2)
        kernels = kernels.unsqueeze(1).repeat(in_ch, 1, 1, 1)  # (4*in_ch, 1, 2, 2) via groups
        # actually: (4, 1, 2, 2) applied per-channel via groups
        k_full = kernels[:4].unsqueeze(0)  # reuse single channel, expand later
        self.register_buffer('haar_k', torch.stack([ll, lh, hl, hh]).unsqueeze(1))

        self.in_ch = in_ch
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch * 4 * 3, 128, 1, bias=False),  # fuse all subbands
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 64, 3, padding=1, groups=64, bias=False),
            nn.Conv2d(64, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )

    def _dwt(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        subbands = []
        for c in range(C):
            ch = x[:, c:c+1]
            for k in range(4):
                sb = F.conv2d(ch, self.haar_k[k:k+1], stride=2, padding=0)
                subbands.append(sb)
        return torch.cat(subbands, dim=1)  # (B, C*4, H/2, W/2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # 3 levels of DWT for multi-scale frequency analysis
        w1 = self._dwt(x)                # level 1  (B, 12, H/2, W/2)
        x2 = x[:, :, ::2, ::2]
        w2 = self._dwt(x2)               # level 2
        x3 = x[:, :, ::4, ::4]
        w3 = self._dwt(x3)               # level 3

        # Resize to same spatial size (smallest)
        H3, W3 = w3.shape[2], w3.shape[3]
        w1 = F.adaptive_avg_pool2d(w1, (H3, W3))
        w2 = F.adaptive_avg_pool2d(w2, (H3, W3))

        feat = torch.cat([w1, w2, w3], dim=1)  # (B, 36, H3, W3)
        feat = self.conv(feat)                   # (B, proj_dim, H3, W3)
        return feat.mean(dim=[2, 3])             # (B, proj_dim)


# ── SRM + CNN residual extractor ──────────────────────────────────────────────

class SRMResidualExtractor(nn.Module):
    def __init__(self, proj_dim: int = 128):
        super().__init__()
        srm = _srm_kernels()  # (30, 1, 5, 5)
        # Expand to 3-channel (grouped)
        srm3 = srm.repeat(3, 1, 1, 1)  # (90, 1, 5, 5)
        self.register_buffer('srm_weight', srm3)

        # Learnable filter bank on top of SRM
        self.learnable = nn.Conv2d(3, 32, 5, padding=2, bias=False)
        nn.init.kaiming_normal_(self.learnable.weight)

        # Fixed Laplacian
        lap = torch.tensor([[0,-1,0],[-1,4,-1],[0,-1,0]], dtype=torch.float32)
        self.register_buffer('lap', lap.view(1,1,3,3).repeat(3,1,1,1))

        in_ch = 90 + 32 + 3  # srm + learnable + lap

        self.net = nn.Sequential(
            nn.Conv2d(in_ch, 128, 3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(128),
            nn.GELU(),
            nn.Conv2d(128, 128, 3, stride=2, padding=1, groups=128, bias=False),
            nn.Conv2d(128, 192, 1, bias=False),
            nn.BatchNorm2d(192),
            nn.GELU(),
            nn.Conv2d(192, 192, 3, stride=2, padding=1, groups=192, bias=False),
            nn.Conv2d(192, proj_dim, 1, bias=False),
            nn.BatchNorm2d(proj_dim),
            nn.GELU(),
        )
        self.norm = nn.LayerNorm(proj_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: raw [0,1] RGB
        srm_out = F.conv2d(x, self.srm_weight, padding=2, groups=3)
        srm_out = torch.tanh(srm_out)

        learnable_out = torch.tanh(self.learnable(x))

        lap_out = F.conv2d(x, self.lap, padding=1, groups=3).clamp(-1, 1)

        feat = torch.cat([srm_out, learnable_out, lap_out], dim=1)
        feat = self.net(feat).mean(dim=[2, 3])
        return self.norm(feat)


# ── Frequency Attention Module ─────────────────────────────────────────────────
# Cross-attention between DCT, SRM, and Wavelet embeddings

class FreqCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=0.1, batch_first=True)
        self.ff   = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, dim * 2),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(dim * 2, dim),
        )

    def forward(self, streams: torch.Tensor) -> torch.Tensor:
        # streams: (B, N_streams, dim)
        normed = self.norm(streams)
        out, _ = self.attn(normed, normed, normed)
        out    = streams + out
        out    = out + self.ff(out)
        return out.mean(dim=1)  # (B, dim)


# ── Squeeze-Excitation for frequency channels ─────────────────────────────────

class FreqSE(nn.Module):
    def __init__(self, dim: int, reduction: int = 4):
        super().__init__()
        self.fc = nn.Sequential(
            nn.Linear(dim, dim // reduction),
            nn.GELU(),
            nn.Linear(dim // reduction, dim),
            nn.Sigmoid(),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.fc(x)


# ── Main Frequency Model ───────────────────────────────────────────────────────

class FrequencyModelV6(nn.Module):
    """
    Pure frequency-domain deepfake detector.

    3 parallel streams:
      1. SRM residual extractor    (pixel noise fingerprint)
      2. DCT spectral block        (JPEG/GAN spectral peaks)
      3. Haar wavelet block        (checkerboard / upconv artifacts)

    Streams fused via cross-attention → classifier
    """

    def __init__(self, num_classes: int = 2, image_size: int = 128,
                 dropout: float = 0.35, proj_dim: int = 192):
        super().__init__()
        self.image_size = image_size

        self.srm_stream  = SRMResidualExtractor(proj_dim=proj_dim)
        self.dct_stream  = DCTSpectralBlock(proj_dim=proj_dim)
        self.haar_stream = HaarWaveletBlock(in_ch=3, proj_dim=proj_dim)

        self.stream_se = nn.ModuleList([FreqSE(proj_dim) for _ in range(3)])

        self.cross_attn = FreqCrossAttention(proj_dim, num_heads=4)

        self.classifier = nn.Sequential(
            nn.LayerNorm(proj_dim),
            nn.Dropout(dropout),
            nn.Linear(proj_dim, proj_dim // 2),
            nn.GELU(),
            nn.Dropout(dropout * 0.5),
            nn.Linear(proj_dim // 2, num_classes),
        )

        # Learnable temperature for calibration
        self.temperature = nn.Parameter(torch.ones(1), requires_grad=False)

        # Per-stream gate (learn which stream is most reliable per sample)
        self.stream_gate = nn.Sequential(
            nn.Linear(proj_dim * 3, 3),
            nn.Softmax(dim=-1),
        )

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, std=0.02)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, (nn.BatchNorm2d, nn.LayerNorm)):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, 3, H, W) in [0, 1] (no ImageNet normalization)
        s1 = self.srm_stream(x)
        s2 = self.dct_stream(x)
        s3 = self.haar_stream(x)

        # SE gating per stream
        s1 = self.stream_se[0](s1)
        s2 = self.stream_se[1](s2)
        s3 = self.stream_se[2](s3)

        # Adaptive stream weighting
        gate = self.stream_gate(torch.cat([s1, s2, s3], dim=1))  # (B, 3)
        fused_gated = gate[:, 0:1] * s1 + gate[:, 1:2] * s2 + gate[:, 2:3] * s3

        # Cross-stream attention fusion
        streams = torch.stack([s1, s2, s3], dim=1)  # (B, 3, D)
        fused_attn = self.cross_attn(streams)         # (B, D)

        # Residual combine
        fused = fused_gated + fused_attn

        logits = self.classifier(fused) / self.temperature
        return logits

    def forward_with_streams(self, x: torch.Tensor):
        s1 = self.srm_stream(x)
        s2 = self.dct_stream(x)
        s3 = self.haar_stream(x)
        s1 = self.stream_se[0](s1)
        s2 = self.stream_se[1](s2)
        s3 = self.stream_se[2](s3)
        gate = self.stream_gate(torch.cat([s1, s2, s3], dim=1))
        fused_gated = gate[:, 0:1] * s1 + gate[:, 1:2] * s2 + gate[:, 2:3] * s3
        streams = torch.stack([s1, s2, s3], dim=1)
        fused_attn = self.cross_attn(streams)
        fused = fused_gated + fused_attn
        logits = self.classifier(fused) / self.temperature
        norms = {
            'srm_norm':  s1.norm(dim=1).mean().item(),
            'dct_norm':  s2.norm(dim=1).mean().item(),
            'haar_norm': s3.norm(dim=1).mean().item(),
            'gate_srm':  gate[:, 0].mean().item(),
            'gate_dct':  gate[:, 1].mean().item(),
            'gate_haar': gate[:, 2].mean().item(),
        }
        return logits, norms

    def set_temperature(self, t: float):
        self.temperature.fill_(t)

    def calibrate(self, val_loader, device: torch.device,
                  temp_range: Tuple[float, float] = (0.5, 3.0)) -> float:
        try:
            from scipy.optimize import minimize_scalar
        except ImportError:
            print("scipy not found — skipping calibration")
            return 1.0
        self.eval()
        logits_list, labels_list = [], []
        with torch.no_grad():
            for batch in val_loader:
                imgs   = batch[0].to(device)
                labels = batch[1]
                logits_list.append(self.forward(imgs).cpu())
                labels_list.append(labels)
        all_logits = torch.cat(logits_list)
        all_labels = torch.cat(labels_list)

        def nll(T):
            return F.cross_entropy(all_logits / float(T), all_labels).item()

        res   = minimize_scalar(nll, bounds=temp_range, method='bounded')
        T_opt = float(np.clip(res.x, temp_range[0], temp_range[1]))
        self.set_temperature(T_opt)
        print(f"  Calibrated temperature: {T_opt:.4f}")
        return T_opt


# ── Factory ────────────────────────────────────────────────────────────────────

def get_frequency_model_v6(
    num_classes: int = 2,
    image_size:  int = 128,
    dropout:     float = 0.35,
    proj_dim:    int = 192,
    weights_path: Optional[str] = None,
) -> FrequencyModelV6:
    model = FrequencyModelV6(num_classes, image_size, dropout, proj_dim)
    if weights_path and os.path.exists(weights_path):
        try:
            ckpt  = torch.load(weights_path, map_location='cpu', weights_only=False)
            state = ckpt.get('model_state_dict', ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            print(f"  Loaded: {weights_path}")
        except Exception as e:
            print(f"  Could not load weights: {e}")
    return model

# Alias used by v5_2trainFrequency.py
get_frequency_model_v5 = get_frequency_model_v6


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("FrequencyModelV6 Self-Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = get_frequency_model_v6().to(device)
    model.eval()

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Parameters: {total_params:,}")

    for B, H in [(1, 128), (4, 128), (2, 256)]:
        x = torch.rand(B, 3, H, H).to(device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 2), f"Expected ({B},2), got {out.shape}"
        probs = torch.softmax(out, dim=1)
        print(f"  ({B},3,{H},{H}) -> {out.shape}  FAKE={probs[0,0]:.3f}  REAL={probs[0,1]:.3f}  OK")

    x = torch.rand(2, 3, 128, 128).to(device)
    with torch.no_grad():
        logits, norms = model.forward_with_streams(x)
    print(f"  Streams: { {k: round(v,3) for k,v in norms.items()} }")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e6
        print(f"  Peak VRAM: {peak:.1f} MB")

    print("\n✅ All FrequencyModelV6 tests passed!")