"""
SpatialModelV6.4 — AI Image Forensics
======================================
Inherits all v6.3 fixes. Targeted improvements for GAN/diffusion detection
(DALL-E, Pixabay, Stable Diffusion, Midjourney) WITHOUT degrading real-image accuracy.

Key upgrades vs v6.3:
  ✅ FRQ-1  Frequency stream: SRM-style high-pass residual extractor (30 learnable
             filter banks) captures ringing/spectral fingerprints invisible to ViT.
  ✅ FRQ-2  DCT spectral feature block: 2D DCT log-energy histogram appended to
             fused features. GAN/diffusion images have characteristic spectral peaks.
  ✅ FRQ-3  JPEG artifact simulation in training augmentation (compression q=40-95)
             matches real-world online image distribution (DALL-E, Pixabay output).
  ✅ FRQ-4  Noise pattern augmentation: Gaussian + Poisson noise, sharpening, slight
             downscale-upscale to simulate social-media re-encoding.
  ✅ CLF-1  CosineClassifier clamp widened: ±5 → ±8. Near-boundary GAN images need
             more logit range. Hard clamp was squashing genuine fake confidence.
  ✅ CLF-2  Fake-class logit bias: learnable scalar bias on FAKE logit only.
             Corrects systematic under-confidence on GAN/diffusion class.
  ✅ CLF-3  AsymmetricFocalLoss replaces FocalLoss: separate gamma_fake/gamma_real
             so hard fake examples get extra focus (gamma_fake=3.0) while real
             examples are not over-penalised (gamma_real=2.0).
  ✅ MEM-1  ErrorLearningMemory fake boost cap raised 2.5→4.0, boost factor 0.30→0.55
             so error-adaptive weights respond faster to fake misclassification runs.
  ✅ AUG-1  MixUp alpha reduced 0.4→0.2: avoids blending real+fake into the exact
             decision-boundary region where GAN images sit.
  ✅ AUG-2  CutMix added alongside MixUp (50/50 chance). CutMix forces model to
             inspect local patches — precisely what exposes GAN artifacts.
  ✅ ATT-1  Entropy weight 0.01→0.03: stronger pressure to spread attention across
             patches, preventing attention collapse to a single region.
  ✅ OPT-1  Separate learning-rate group for frequency stream (2× head LR) so the
             new frequency pathway learns faster than the pretrained ViT head.
  ✅ CFG-1  Default unfreeze_last_n=3 (was 2): more ViT fine-tuning for subtle artifacts.
  ✅ CFG-2  Focal gamma default 2.5 (was 2.0) in CLI.
  ✅ VRAM   Frequency CNN uses depthwise separable convolutions — < 2MB extra VRAM.
             Safe for RTX 2050 4GB with batch_size=8.
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
from typing import Optional, Tuple, List, Dict
from collections import defaultdict
from datetime import datetime

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]
_CKPT_SECRET = b'spatial_v6_signing_key_change_me'


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


# ── SRM-style Frequency Stream (NEW v6.4) ──────────────────────────────────────
#
# Spatial Rich Model (SRM) filters are high-pass filters designed for steganalysis.
# GAN and diffusion models leave characteristic residual patterns in frequency space:
#   - GAN checkerboard artifacts (upconv stride artefacts)
#   - Diffusion model ringing/spectral peaks from DDPM denoising steps
#   - DALL-E clip-guided composition boundaries
# These are largely invisible to a frozen CLIP ViT which is trained to be
# robust to such low-level variations. A dedicated frequency path captures them.
#
# Architecture: DepthwiseSeparable CNN (low VRAM) → GAP → 128-d embedding
# Input: un-normalized RGB (we de-normalize then apply SRM-style kernels)

class DepthwiseSepConv(nn.Module):
    """Depthwise separable conv to keep VRAM usage minimal."""
    def __init__(self, in_ch: int, out_ch: int, kernel: int = 3, stride: int = 1):
        super().__init__()
        self.dw = nn.Conv2d(in_ch, in_ch, kernel, stride=stride,
                            padding=kernel // 2, groups=in_ch, bias=False)
        self.pw = nn.Conv2d(in_ch, out_ch, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_ch)

    def forward(self, x):
        return F.gelu(self.bn(self.pw(self.dw(x))))


class FrequencyStream(nn.Module):
    """
    High-frequency residual extractor for GAN/diffusion artifact detection.

    Pipeline:
      1. De-normalise image back to [0,1]
      2. Apply 3x3 Laplacian high-pass (fixed, not learned) to isolate residual
      3. Pass through 3-layer depthwise-separable CNN
      4. Global average pooling → 128-d feature vector

    The Laplacian pre-filter is frozen; only the CNN weights are trained.
    This gives the frequency path a head-start without requiring large conv filters.
    """

    # Fixed 3×3 Laplacian kernel (high-pass residual)
    _LAPLACIAN = torch.tensor([
        [ 0., -1.,  0.],
        [-1.,  4., -1.],
        [ 0., -1.,  0.],
    ], dtype=torch.float32)

    def __init__(self, out_dim: int = 128):
        super().__init__()
        self.out_dim = out_dim

        # Fixed 3-channel Laplacian (applied per-channel, not learned)
        kernel = self._LAPLACIAN.view(1, 1, 3, 3).repeat(3, 1, 1, 1)  # (3,1,3,3)
        self.register_buffer('laplacian', kernel)

        # Learnable SRM-style filter bank (30 filters, 3 channels in)
        self.srm_conv = nn.Conv2d(3, 30, kernel_size=5, padding=2,
                                  groups=1, bias=False)
        nn.init.kaiming_normal_(self.srm_conv.weight, mode='fan_out')

        # 3-layer depthwise-sep CNN
        self.net = nn.Sequential(
            DepthwiseSepConv(33, 64, kernel=3, stride=2),   # 33 = 3 lap + 30 srm
            DepthwiseSepConv(64, 96, kernel=3, stride=2),
            DepthwiseSepConv(96, out_dim, kernel=3, stride=2),
        )

        self.norm = nn.LayerNorm(out_dim)

        # De-normalise buffers (CLIP stats)
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer('_mean', mean)
        self.register_buffer('_std',  std)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # De-normalise: x_raw ∈ [0, 1] approx
        x_raw = x * self._std + self._mean
        x_raw = x_raw.clamp(0.0, 1.0)

        # Fixed Laplacian high-pass: (B, 3, H, W)
        lap = F.conv2d(x_raw, self.laplacian, padding=1, groups=3)
        lap = lap.clamp(-1.0, 1.0)

        # Learned SRM-style filters
        srm = self.srm_conv(x_raw)
        srm = torch.tanh(srm)

        # Concatenate: (B, 33, H, W)
        feat = torch.cat([lap, srm], dim=1)

        # CNN → GAP → norm
        feat = self.net(feat)                           # (B, out_dim, H', W')
        feat = feat.mean(dim=[2, 3])                    # (B, out_dim)
        return self.norm(feat)


# ── DCT Spectral Feature Block (NEW v6.4) ──────────────────────────────────────
#
# GAN and diffusion models leave spectral fingerprints (periodic artefacts) that
# show up as peaks in the 2D DCT log-energy spectrum.  We compute a compact
# log-energy histogram over the 8x8 DCT blocks and embed it.
# This is very cheap computationally (no learned params in DCT itself).

class DCTSpectralBlock(nn.Module):
    """
    Computes 2D DCT log-energy features over 8×8 blocks on grayscale image.
    Returns a 64-d embedding (one value per DCT frequency coefficient position).
    Then projects to proj_dim.
    """

    def __init__(self, proj_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(64, proj_dim),
            nn.LayerNorm(proj_dim),
            nn.GELU(),
        )
        # De-normalise buffers
        mean = torch.tensor(CLIP_MEAN).view(1, 3, 1, 1)
        std  = torch.tensor(CLIP_STD).view(1, 3, 1, 1)
        self.register_buffer('_mean', mean)
        self.register_buffer('_std',  std)

    @staticmethod
    def _dct2_blocks(x: torch.Tensor) -> torch.Tensor:
        """
        Compute 2D DCT log-energy over non-overlapping 8×8 blocks of a
        grayscale image tensor (B, 1, H, W).  Returns (B, 64) mean log-energy.
        """
        B, C, H, W = x.shape
        # Pad to multiple of 8
        ph = (8 - H % 8) % 8
        pw = (8 - W % 8) % 8
        if ph > 0 or pw > 0:
            x = F.pad(x, (0, pw, 0, ph))
        _, _, H2, W2 = x.shape

        # Unfold into 8×8 blocks: (B, 64, n_blocks)
        blocks = x.unfold(2, 8, 8).unfold(3, 8, 8)  # (B,1,nH,nW,8,8)
        blocks = blocks.contiguous().view(B, -1, 8, 8)  # (B, n_blocks, 8, 8)

        # 2D DCT via two 1D DCTs (separable)
        def dct1d(v):
            N = v.shape[-1]
            n = torch.arange(N, device=v.device, dtype=v.dtype)
            k = torch.arange(N, device=v.device, dtype=v.dtype)
            cos_mat = torch.cos(math.pi / N * (n.unsqueeze(1) + 0.5) * k.unsqueeze(0))
            return v @ cos_mat  # (..., N)

        dct_h = dct1d(blocks)          # (B, n_blocks, 8, 8) — DCT over last dim
        dct_2d = dct1d(dct_h.transpose(-1, -2)).transpose(-1, -2)  # (B, n_blocks, 8, 8)

        # Log energy per coefficient, averaged over blocks: (B, 64)
        energy = (dct_2d ** 2).mean(dim=1).view(B, 64)
        log_e  = torch.log1p(energy)
        return log_e

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # De-normalise
        x_raw = (x * self._std + self._mean).clamp(0.0, 1.0)
        # Grayscale
        gray = 0.299 * x_raw[:, 0:1] + 0.587 * x_raw[:, 1:2] + 0.114 * x_raw[:, 2:3]
        log_e = self._dct2_blocks(gray)          # (B, 64)
        return self.proj(log_e)                  # (B, proj_dim)


# ── Patch Attention Pool ───────────────────────────────────────────────────────

class PatchAttentionPool(nn.Module):
    def __init__(self, dim: int, hidden: int = 128):
        super().__init__()
        self.attn = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden),
            nn.Tanh(),
            nn.Linear(hidden, 1, bias=False),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        w      = self.attn(x)
        w      = torch.softmax(w, dim=1)
        pooled = (w * x).sum(dim=1)
        return pooled, w.squeeze(-1)


# ── Artifact Cross-Attention ───────────────────────────────────────────────────

class ArtifactCrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.norm_q  = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.attn    = nn.MultiheadAttention(
            embed_dim=dim, num_heads=num_heads,
            dropout=dropout, batch_first=True,
            add_bias_kv=True,
        )
        self.proj = nn.Linear(dim, dim)
        self.drop = nn.Dropout(dropout)

    def forward(self, cls: torch.Tensor, patches: torch.Tensor) -> torch.Tensor:
        q_norm = self.norm_q(cls).unsqueeze(1)
        kv     = self.norm_kv(patches)
        out, _ = self.attn(q_norm, kv, kv)
        out    = out + q_norm
        return self.drop(self.proj(out.squeeze(1)))


# ── Cosine Classifier (v6.4: wider clamp ±8, per-class bias for FAKE) ─────────

class CosineClassifier(nn.Module):
    """
    FAKE=0, REAL=1.
    v6.4 changes:
      - logit clamp widened: ±5 → ±8  (GAN images near boundary need more range)
      - fake_bias: learnable scalar added to FAKE logit only. Initialised to +0.3
        to counteract systematic under-confidence on diffusion/GAN images.
    """
    def __init__(self, in_features: int, num_classes: int, initial_scale: float = 15.0):
        super().__init__()
        self.weight    = nn.Parameter(torch.empty(num_classes, in_features))
        self.scale     = nn.Parameter(torch.tensor(initial_scale))
        self.scale.requires_grad = False  # frozen until warmup done
        # Separate bias for FAKE class only (index 0)
        # Init to +0.3: slight push toward predicting FAKE on uncertain inputs
        self.fake_bias = nn.Parameter(torch.tensor(0.3))
        nn.init.orthogonal_(self.weight)

    def unfreeze_scale(self):
        self.scale.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n    = F.normalize(x, dim=1)
        w_n    = F.normalize(self.weight, dim=1)
        logits = self.scale * (x_n @ w_n.T)          # (B, 2)
        # Add learnable bias to FAKE column (index 0)
        bias   = torch.stack([self.fake_bias,
                               torch.zeros(1, device=x.device).squeeze()], dim=0)
        logits = logits + bias.unsqueeze(0)
        # Wider clamp for near-boundary separation
        return torch.clamp(logits, min=-8.0, max=8.0)


# ── Asymmetric Focal Loss (NEW v6.4) ───────────────────────────────────────────
#
# Standard FocalLoss uses the same gamma for both classes.
# Problem: REAL images are already well-classified (high confidence), so
# focal down-weighting is not hurting them much.  But FAKE images near the
# decision boundary (GAN/diffusion) need even MORE focus.
# Solution: gamma_fake > gamma_real — harder focus on the FAKE class.

class AsymmetricFocalLoss(nn.Module):
    """
    Per-class focal loss with independent gamma values.
      gamma_fake (default 3.0): stronger focus on hard FAKE examples
      gamma_real (default 2.0): standard focus on REAL examples (preserve accuracy)
    Also implements label_smoothing and per-class weights.
    """
    def __init__(self, gamma_fake: float = 3.0, gamma_real: float = 2.0,
                 weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma_fake      = gamma_fake
        self.gamma_real      = gamma_real
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        # Per-sample CE (with label smoothing, no reduction)
        ce = F.cross_entropy(logits, targets, weight=self.weight,
                             label_smoothing=self.label_smoothing, reduction='none')
        pt = torch.exp(-ce)

        # Per-sample gamma selection: FAKE=0 → gamma_fake, REAL=1 → gamma_real
        gamma = torch.where(targets == 0,
                            torch.full_like(ce, self.gamma_fake),
                            torch.full_like(ce, self.gamma_real))
        loss = ((1.0 - pt) ** gamma) * ce
        return loss.mean()

    def update_weight(self, w: torch.Tensor):
        if self.weight is not None:
            dev = self.weight.device
        else:
            dev = torch.device('cpu')
        self.weight = w.to(dev)


# Keep FocalLoss as alias for backward compatibility (checkpoint loading)
class FocalLoss(AsymmetricFocalLoss):
    """Backward-compatible alias — maps gamma → gamma_fake=gamma, gamma_real=gamma."""
    def __init__(self, gamma: float = 2.0, weight=None, label_smoothing: float = 0.05):
        super().__init__(gamma_fake=gamma, gamma_real=gamma,
                         weight=weight, label_smoothing=label_smoothing)


# ── MixUp / CutMix (v6.4: alpha 0.4→0.2, CutMix added) ──────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.2):
    """MixUp. alpha reduced to 0.2 to avoid blending near the real/fake boundary."""
    if alpha <= 0 or not torch.is_tensor(x):
        return x, y, y, 1.0
    lam   = np.random.beta(alpha, alpha)
    idx   = torch.randperm(x.size(0), device=x.device)
    mix_x = lam * x + (1 - lam) * x[idx]
    return mix_x, y, y[idx], lam


def cutmix_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.5):
    """
    CutMix augmentation (NEW v6.4).
    Forces model to inspect local patches — critical for exposing GAN artifacts
    that may be confined to specific spatial regions.
    """
    if alpha <= 0:
        return x, y, y, 1.0
    lam = np.random.beta(alpha, alpha)
    B, C, H, W = x.shape
    idx  = torch.randperm(B, device=x.device)
    # Random box
    cut_rat = math.sqrt(1.0 - lam)
    cut_w   = int(W * cut_rat)
    cut_h   = int(H * cut_rat)
    cx      = random.randint(0, W)
    cy      = random.randint(0, H)
    x1, x2  = max(cx - cut_w // 2, 0), min(cx + cut_w // 2, W)
    y1, y2  = max(cy - cut_h // 2, 0), min(cy + cut_h // 2, H)
    mix_x   = x.clone()
    mix_x[:, :, y1:y2, x1:x2] = x[idx, :, y1:y2, x1:x2]
    lam_adj = 1.0 - (x2 - x1) * (y2 - y1) / (W * H)
    return mix_x, y, y[idx], lam_adj


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ── Backbone loader ────────────────────────────────────────────────────────────

def _load_backbone() -> Tuple[nn.Module, int, str]:
    vram = _vram_gb()
    # v6.4: prefer ViT-B/16 even on 4GB — more patches (196 vs 49), far better
    # for detecting subtle GAN/diffusion artifacts. VRAM managed by grad checkpoint.
    if vram >= 10.0:
        candidates = [('ViT-L-14', 1024), ('ViT-B-16', 512), ('ViT-B-32', 512)]
    elif vram >= 3.5:
        # RTX 2050 4GB: ViT-B/16 fits with grad checkpointing + batch=8
        candidates = [('ViT-B-16', 512), ('ViT-B-32', 512)]
    else:
        candidates = [('ViT-B-32', 512)]

    print(f"  VRAM: {vram:.1f} GB  ->  candidates: {[c[0] for c in candidates]}")

    try:
        import open_clip
        for model_id, embed_dim in candidates:
            try:
                model, _, _ = open_clip.create_model_and_transforms(model_id, pretrained='openai')
                print(f"  Loaded {model_id} via open_clip  (embed_dim={embed_dim})")
                return model, embed_dim, f'openclip_{model_id}'
            except Exception as e:
                print(f"  {model_id} failed: {e}")
    except ImportError:
        print("  open_clip not installed, trying openai/clip")

    try:
        import clip as openai_clip
        name_map = {'ViT-B-32': 'ViT-B/32', 'ViT-B-16': 'ViT-B/16', 'ViT-L-14': 'ViT-L/14'}
        for model_id, embed_dim in candidates:
            slash = name_map.get(model_id, 'ViT-B/32')
            try:
                model, _ = openai_clip.load(slash, device='cpu')
                print(f"  Loaded {slash} via openai/clip  (embed_dim={embed_dim})")
                return model, embed_dim, f'openai_{model_id}'
            except Exception as e:
                print(f"  {slash} failed: {e}")
    except ImportError:
        print("  openai/clip not installed, using torchvision ViT fallback")

    from torchvision.models import vit_b_16, ViT_B_16_Weights
    model = vit_b_16(weights=ViT_B_16_Weights.IMAGENET1K_V1)
    print("  Loaded torchvision ViT-B/16 as fallback  (embed_dim=768)")
    return model, 768, 'torchvision_vitb16'


# ── Multi-scale Feature Extractor ─────────────────────────────────────────────

class MultiScaleCLIPExtractor(nn.Module):
    def __init__(self, model: nn.Module, embed_dim: int, backbone_type: str):
        super().__init__()
        self.model         = model
        self.embed_dim     = embed_dim
        self.backbone_type = backbone_type
        self._mid_out:   Optional[torch.Tensor] = None
        self._final_out: Optional[torch.Tensor] = None
        self._handles: List = []
        self._register_hooks()

    def _make_hook(self, store_attr: str):
        def _hook(module, inp, out):
            t = out[0] if isinstance(out, tuple) else out
            if t.dim() == 3 and t.shape[0] > t.shape[1]:
                t = t.permute(1, 0, 2).contiguous()
            setattr(self, store_attr, t)
        return _hook

    def _register_hooks(self):
        try:
            if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
                blocks = list(self.model.visual.transformer.resblocks.children())
            elif self.backbone_type == 'torchvision_vitb16':
                blocks = list(self.model.encoder.layers.children())
            else:
                print("  Multi-scale hooks: unknown backbone, skipping")
                return
            n       = len(blocks)
            mid_idx = max(0, n // 2 - 1)
            h1 = blocks[mid_idx].register_forward_hook(self._make_hook('_mid_out'))
            h2 = blocks[-1].register_forward_hook(self._make_hook('_final_out'))
            self._handles = [h1, h2]
            print(f"  Multi-scale hooks: block {mid_idx} (mid) + block {n-1} (final)")
        except (AttributeError, IndexError) as e:
            print(f"  Multi-scale hooks failed ({e}) — will use CLS fallback")

    def forward(self, x: torch.Tensor):
        self._mid_out = None
        self._final_out = None

        if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
            cls = self.model.encode_image(x)
        elif self.backbone_type == 'torchvision_vitb16':
            x2  = self.model._process_input(x)
            ct  = self.model.class_token.expand(x2.shape[0], -1, -1)
            x2  = torch.cat([ct, x2], dim=1) + self.model.encoder.pos_embedding
            x2  = self.model.encoder.dropout(x2)
            x2  = self.model.encoder.layers(x2)
            x2  = self.model.encoder.ln(x2)
            cls = x2[:, 0]
        else:
            out = self.model(x)
            cls = out[0] if isinstance(out, tuple) else out
            if cls.dim() > 2:
                cls = cls[:, 0]

        if cls.dim() == 1:
            cls = cls.unsqueeze(0)
        cls = cls[:, :self.embed_dim]

        def _safe_patches(hook_out):
            if hook_out is not None:
                p = hook_out[:, 1:, :]
                if p.shape[-1] != self.embed_dim:
                    p = p[..., :self.embed_dim]
                return p
            return cls.unsqueeze(1)

        return cls, _safe_patches(self._mid_out), _safe_patches(self._final_out)

    def __del__(self):
        for h in self._handles:
            try:
                h.remove()
            except Exception:
                pass


# ── Error Learning Memory (v6.4: higher fake boost cap + faster adaptation) ────

class ErrorLearningMemory:
    """
    v6.4 changes:
      - fake_err_rate boost factor: 0.30 → 0.55 (faster adaptation to fake misses)
      - fake weight cap: 2.5 → 4.0 (allows stronger correction when model is
        systematically missing GAN/diffusion images)
      - real weight cap kept at 2.5 (don't let real weight explode — it's already good)
    """

    def __init__(self, max_records: int = 500):
        self.max_records = max_records
        self.records: List[Dict] = []
        self.error_stats = {
            'real_as_fake': [],
            'fake_as_real': [],
            'error_patterns': defaultdict(int),
        }

    def record(self, true_label: int, pred_label: int, confidence: float,
               stream_norms: Optional[Dict[str, float]] = None, image_path: str = ''):
        if pred_label == true_label:
            return
        label_map  = {0: 'FAKE', 1: 'REAL'}
        error_type = f"{label_map[true_label]}_as_{label_map[pred_label]}"
        key = error_type.lower().replace('-', '_')
        if key in self.error_stats:
            self.error_stats[key].append(confidence)
        dominant_stream = None
        if stream_norms:
            dominant_stream = max(stream_norms, key=stream_norms.get)
            self.error_stats['error_patterns'][f"{error_type}_{dominant_stream}"] += 1
        entry = {
            'ts':              datetime.now().isoformat(),
            'true':            label_map[true_label],
            'pred':            label_map[pred_label],
            'confidence':      round(confidence, 4),
            'dominant_stream': dominant_stream,
            'stream_norms':    stream_norms,
            'image_path':      image_path,
            'cause_hint':      self._cause_hint(error_type, dominant_stream),
        }
        self.records.append(entry)
        if len(self.records) > self.max_records:
            self.records.pop(0)

    def _cause_hint(self, error_type: str, dominant_stream: Optional[str]) -> str:
        hints = {
            'REAL_as_FAKE': {
                'mid_pooled':   'Texture stream over-triggered — real image has unusual noise/JPEG pattern',
                'final_pooled': 'Semantic stream confused — real image has atypical composition',
                'cross_out':    'Anomaly stream false alarm — CLS attended to innocent region',
                'freq_out':     'Frequency stream false alarm — real image has high-freq artifacts (JPEG)',
                None:           'High confidence wrong — investigate image quality or augmentation',
            },
            'FAKE_as_REAL': {
                'mid_pooled':   'Texture artifacts missed — GAN/diffusion smoothed pixel distribution',
                'final_pooled': 'Semantic plausibility too high — generator produces natural-looking structure',
                'cross_out':    'Anomaly stream blind — artifacts in unattended regions',
                'freq_out':     'Frequency stream missed — GAN frequency profile mimics real images',
                None:           'Low discriminability — image likely near decision boundary',
            },
        }
        return hints.get(error_type, {}).get(dominant_stream, 'Unknown pattern')

    def get_loss_weights(self, base_weights: List[float]) -> List[float]:
        rw = len(self.error_stats['real_as_fake'])
        fw = len(self.error_stats['fake_as_real'])
        if rw + fw == 0:
            return base_weights
        real_err_rate = rw / (rw + fw + 1e-8)
        fake_err_rate = fw / (rw + fw + 1e-8)
        w    = list(base_weights)
        # v6.4: stronger FAKE boost (0.30→0.55), higher cap (2.5→4.0)
        w[0] = base_weights[0] * (1.0 + 0.55 * fake_err_rate)
        w[0] = min(w[0], 4.0)
        # Real boost kept conservative to preserve real accuracy
        w[1] = base_weights[1] * (1.0 + 0.25 * real_err_rate)
        w[1] = min(w[1], 2.5)
        return w

    def summary(self) -> Dict:
        rw = self.error_stats['real_as_fake']
        fw = self.error_stats['fake_as_real']
        return {
            'total_errors':      len(self.records),
            'real_as_fake':      len(rw),
            'fake_as_real':      len(fw),
            'avg_conf_real_err': round(float(sum(rw) / max(len(rw), 1)), 4),
            'avg_conf_fake_err': round(float(sum(fw) / max(len(fw), 1)), 4),
            'top_error_patterns': dict(
                sorted(self.error_stats['error_patterns'].items(),
                       key=lambda x: x[1], reverse=True)[:5]
            ),
        }

    def save(self, path: str):
        os.makedirs(os.path.dirname(path) or '.', exist_ok=True)
        with open(path, 'w') as f:
            json.dump({'records': self.records[-100:], 'summary': self.summary()}, f, indent=2)

    def load(self, path: str):
        if not os.path.exists(path):
            return
        with open(path) as f:
            data = json.load(f)
        self.records = data.get('records', [])


# ── GradCAM (unchanged from v6.3, fully functional) ───────────────────────────

class GradCAM:
    def __init__(self, model: 'SpatialModelV64'):
        self.model  = model
        self._grads: Optional[torch.Tensor] = None
        self._acts:  Optional[torch.Tensor] = None
        self._handle_f = None
        self._handle_b = None

    def _register(self):
        target = self.model.final_pool

        def fwd_hook(m, inp, out):
            self._acts = inp[0].detach()

        def bwd_hook(m, grad_in, grad_out):
            if grad_in[0] is not None:
                self._grads = grad_in[0].detach()

        self._handle_f = target.register_forward_hook(fwd_hook)
        self._handle_b = target.register_full_backward_hook(bwd_hook)

    def _remove(self):
        if self._handle_f:
            self._handle_f.remove()
            self._handle_f = None
        if self._handle_b:
            self._handle_b.remove()
            self._handle_b = None

    @torch.enable_grad()
    def generate(self, image: torch.Tensor, class_idx: int = 1) -> torch.Tensor:
        self.model.eval()
        self._grads = None
        self._acts  = None
        self._register()
        try:
            img    = image.clone()
            logits = self.model(img)
            self.model.zero_grad()
            score  = logits[0, class_idx]
            score.backward()

            if self._grads is None or self._acts is None:
                return torch.ones(224, 224) * 0.5

            weights = self._grads.mean(dim=1, keepdim=True)
            cam     = (weights * self._acts).sum(dim=-1)
            cam     = cam.clamp(min=0)
            cam     = cam[0]

            N = cam.shape[0]
            p = int(math.isqrt(N))
            if p * p != N:
                p = max(1, int(N ** 0.5))
                while N % p != 0:
                    p -= 1
                q = N // p
            else:
                q = p

            cam_2d = cam.reshape(1, 1, p, q).float()
            cam_up = F.interpolate(cam_2d, size=(224, 224),
                                   mode='bilinear', align_corners=False)
            cam_up = cam_up.squeeze()
            c_min, c_max = cam_up.min(), cam_up.max()
            if c_max > c_min:
                cam_up = (cam_up - c_min) / (c_max - c_min)
            else:
                cam_up = torch.zeros_like(cam_up)
            return cam_up.cpu()
        finally:
            self._remove()


# ── Adversarial Tester ─────────────────────────────────────────────────────────

class AdversarialTester:
    def __init__(self, model: 'SpatialModelV64', device: torch.device):
        self.model  = model
        self.device = device

    @torch.enable_grad()
    def fgsm(self, images, labels, eps=8/255):
        images = images.clone().detach().requires_grad_(True).to(self.device)
        labels = labels.to(self.device)
        self.model.eval()
        logits = self.model(images)
        loss   = F.cross_entropy(logits, labels)
        loss.backward()
        with torch.no_grad():
            adv = (images + eps * images.grad.sign()).clamp(-3.0, 3.0)
        return adv.detach()

    @torch.enable_grad()
    def pgd(self, images, labels, eps=8/255, alpha=2/255, steps=7):
        images = images.to(self.device)
        labels = labels.to(self.device)
        adv    = images.clone().detach() + torch.empty_like(images).uniform_(-eps, eps)
        adv    = adv.clamp(-3.0, 3.0)
        self.model.eval()
        for _ in range(steps):
            adv = adv.detach().requires_grad_(True)
            logits = self.model(adv)
            loss   = F.cross_entropy(logits, labels)
            loss.backward()
            with torch.no_grad():
                adv = adv + alpha * adv.grad.sign()
                adv = torch.min(torch.max(adv, images - eps), images + eps).clamp(-3.0, 3.0)
        return adv.detach()

    def evaluate(self, loader, attack='fgsm', eps=8/255, max_batches=20) -> Dict:
        self.model.eval()
        clean_correct = adv_correct = total = 0
        class_clean = defaultdict(lambda: [0, 0])
        class_adv   = defaultdict(lambda: [0, 0])

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            with torch.no_grad():
                clean_preds   = self.model(images).argmax(1)
                clean_correct += (clean_preds == labels).sum().item()
                for c in labels.unique():
                    mask = labels == c
                    class_clean[c.item()][0] += (clean_preds[mask] == labels[mask]).sum().item()
                    class_clean[c.item()][1] += mask.sum().item()

            adv_imgs = self.fgsm(images, labels, eps) if attack == 'fgsm' \
                       else self.pgd(images, labels, eps)

            with torch.no_grad():
                adv_preds   = self.model(adv_imgs).argmax(1)
                adv_correct += (adv_preds == labels).sum().item()
                for c in labels.unique():
                    mask = labels == c
                    class_adv[c.item()][0] += (adv_preds[mask] == labels[mask]).sum().item()
                    class_adv[c.item()][1] += mask.sum().item()
                total += labels.size(0)

        clean_acc = 100.0 * clean_correct / max(total, 1)
        adv_acc   = 100.0 * adv_correct   / max(total, 1)
        drop      = clean_acc - adv_acc
        bal_clean = np.mean([class_clean[c][0] / max(class_clean[c][1], 1)
                             for c in class_clean]) * 100
        bal_adv   = np.mean([class_adv[c][0] / max(class_adv[c][1], 1)
                             for c in class_adv]) * 100
        bal_drop  = bal_clean - bal_adv

        verdict = ('✅ Learning real patterns' if drop < 15
                   else '⚠️ Borderline' if drop < 40
                   else '❌ Memorising/overfitting')

        result = {
            'attack': attack, 'eps': eps,
            'clean_acc': round(clean_acc, 2), 'adv_acc': round(adv_acc, 2),
            'drop': round(drop, 2),
            'bal_clean': round(bal_clean, 2), 'bal_adv': round(bal_adv, 2),
            'bal_drop': round(bal_drop, 2), 'verdict': verdict,
        }
        print(f"\n[Adversarial] {attack.upper()} ε={eps:.3f}")
        print(f"  Clean: {clean_acc:.2f}%  |  Adv: {adv_acc:.2f}%  |  Drop: {drop:.2f}%")
        print(f"  Balanced — Clean: {bal_clean:.2f}%  Adv: {bal_adv:.2f}%  Drop: {bal_drop:.2f}%")
        print(f"  {verdict}")
        return result


# ── MC-Dropout Uncertainty ─────────────────────────────────────────────────────

def predict_with_uncertainty(model: 'SpatialModelV64', image: torch.Tensor,
                              n_passes: int = 10) -> Dict:
    model.train()
    logits_list = []
    with torch.no_grad():
        for _ in range(n_passes):
            logits_list.append(model(image))
    model.eval()
    stacked    = torch.stack(logits_list)
    probs      = torch.softmax(stacked, dim=-1)
    mean_probs = probs.mean(0)
    std_probs  = probs.std(0)
    return {
        'prediction':  mean_probs.argmax(1),
        'confidence':  mean_probs.max(1).values,
        'uncertainty': std_probs.max(1).values,
        'mean_probs':  mean_probs,
    }


# ── Checkpoint helpers ─────────────────────────────────────────────────────────

def save_checkpoint(state: dict, path: str):
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)


def load_checkpoint(path: str, map_location='cpu') -> dict:
    if not os.path.exists(path):
        return {}
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        return ckpt
    except Exception as e:
        print(f"  [WARN] Could not load checkpoint {path}: {e}")
        return {}


def save_signed(checkpoint: dict, path: str, secret: bytes = _CKPT_SECRET):
    data = pickle.dumps(checkpoint)
    sig  = hmac.new(secret, data, hashlib.sha256).hexdigest()
    torch.save({'payload': checkpoint, 'signature': sig}, path)


def load_signed(path: str, secret: bytes = _CKPT_SECRET, strict: bool = True) -> dict:
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'signature' not in saved:
        if strict:
            raise ValueError(f"No signature in {path}")
        return saved
    data     = pickle.dumps(saved['payload'])
    expected = hmac.new(secret, data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, saved['signature']):
        raise ValueError(f"Signature mismatch in {path}")
    return saved['payload']


# ── Main Model V6.4 ────────────────────────────────────────────────────────────

class SpatialModelV64(nn.Module):
    """
    v6.4 architecture:
      [ViT mid patches] → PatchAttentionPool → mid_pooled  (embed_dim)
      [ViT final patches] → PatchAttentionPool → final_pooled (embed_dim)
      [ViT CLS + final patches] → ArtifactCrossAttention → cross_out (embed_dim)
      [Raw image] → FrequencyStream → freq_out  (freq_dim=128)   ← NEW
      [Raw image] → DCTSpectralBlock → dct_out   (dct_dim=64)    ← NEW

      fusion = cat([mid_pooled, final_pooled, cross_out, freq_out, dct_out])
             = embed_dim*3 + freq_dim + dct_dim

      MLP → 512 → 256 → CosineClassifier(256, 2)
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
        dct_dim:             int   = 64,
    ):
        super().__init__()
        self.input_size = input_size
        self.unfrozen_count = 0
        self.freq_dim   = freq_dim
        self.dct_dim    = dct_dim
        self.register_buffer('temperature', torch.tensor(temperature, dtype=torch.float32))

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

        self.mid_pool   = PatchAttentionPool(embed_dim, hidden=128)
        self.final_pool = PatchAttentionPool(embed_dim, hidden=128)
        self.cross_attn = ArtifactCrossAttention(embed_dim, num_heads=4, dropout=dropout / 3)
        self.drop_path  = DropPath(drop_path_rate)

        # NEW v6.4: frequency streams
        self.freq_stream = FrequencyStream(out_dim=freq_dim)
        self.dct_block   = DCTSpectralBlock(proj_dim=dct_dim)

        fusion_in = embed_dim * 3 + freq_dim + dct_dim
        self.mlp = nn.Sequential(
            nn.LayerNorm(fusion_in),
            nn.Linear(fusion_in, 512),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.GELU(),
            nn.Dropout(dropout / 2),
        )
        self.classifier = CosineClassifier(256, num_classes, initial_scale=15.0)
        self._init_weights()
        self.error_memory = ErrorLearningMemory(max_records=500)

        trainable = sum(p.numel() for p in self.parameters() if p.requires_grad)
        total     = sum(p.numel() for p in self.parameters())
        print(f"  SpatialModelV6.4 — {total:,} total | {trainable:,} trainable ({100*trainable/total:.1f}%)")
        print(f"  Fusion dim: {fusion_in} (ViT×3={embed_dim*3} + FreqStream={freq_dim} + DCT={dct_dim})")

    def _unfreeze_last_n(self, n: int):
        if n <= 0:
            return
        try:
            if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
                blocks = list(self.extractor.model.visual.transformer.resblocks.children())
            elif self.backbone_type == 'torchvision_vitb16':
                blocks = list(self.extractor.model.encoder.layers.children())
            else:
                return
            for block in blocks:
                for p in block.parameters():
                    p.requires_grad = False
            for block in blocks[-n:]:
                for p in block.parameters():
                    p.requires_grad = True
            self.unfrozen_count = n
            print(f"  Backbone: last {n} ViT blocks unfrozen")
        except AttributeError as e:
            print(f"  Could not unfreeze blocks (non-critical): {e}")

    def _enable_grad_checkpoint(self):
        try:
            if 'openclip' in self.backbone_type or 'openai' in self.backbone_type:
                blocks = list(self.extractor.model.visual.transformer.resblocks.children())
            elif self.backbone_type == 'torchvision_vitb16':
                blocks = list(self.extractor.model.encoder.layers.children())
            else:
                print("  Gradient checkpointing: architecture not supported, skipped")
                return

            count = 0
            for block in blocks:
                if not any(p.requires_grad for p in block.parameters()):
                    orig_fwd = block.forward

                    def make_wrapped(fwd):
                        def wrapped(*args, **kwargs):
                            return grad_checkpoint(fwd, *args, use_reentrant=False, **kwargs)
                        return wrapped

                    block.forward = make_wrapped(orig_fwd)
                    count += 1

            if count > 0:
                print(f"  Gradient checkpointing: enabled on {count} frozen backbone blocks")
            else:
                print("  Gradient checkpointing: no frozen blocks to wrap")
        except AttributeError as e:
            print(f"  Gradient checkpointing: skipped ({e})")

    def _init_weights(self):
        for m in self.mlp.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight, gain=0.8)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
        for pool in [self.mid_pool, self.final_pool]:
            for layer in pool.attn:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_normal_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        cls, mid_patches, final_patches = self.extractor(x)
        mid_pooled,   _ = self.mid_pool(mid_patches)
        final_pooled, _ = self.final_pool(final_patches)
        cross_out       = self.cross_attn(cls, final_patches)

        # NEW v6.4: frequency features
        freq_out = self.freq_stream(x)
        dct_out  = self.dct_block(x)

        fused  = self.drop_path(
            torch.cat([mid_pooled, final_pooled, cross_out, freq_out, dct_out], dim=1)
        )
        feat   = self.mlp(fused)
        logits = self.classifier(feat) / self.temperature
        return logits

    def forward_with_entropy(self, x: torch.Tensor):
        """Used during training for attention entropy aux-loss."""
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        cls, mid_patches, final_patches = self.extractor(x)
        mid_pooled,   mid_w   = self.mid_pool(mid_patches)
        final_pooled, final_w = self.final_pool(final_patches)
        cross_out             = self.cross_attn(cls, final_patches)

        mid_entropy   = -(mid_w * (mid_w + 1e-8).log()).sum(dim=1).mean()
        final_entropy = -(final_w * (final_w + 1e-8).log()).sum(dim=1).mean()

        freq_out = self.freq_stream(x)
        dct_out  = self.dct_block(x)

        fused  = self.drop_path(
            torch.cat([mid_pooled, final_pooled, cross_out, freq_out, dct_out], dim=1)
        )
        feat   = self.mlp(fused)
        logits = self.classifier(feat) / self.temperature
        return logits, mid_entropy, final_entropy

    def forward_with_streams(self, x: torch.Tensor):
        """Returns logits + per-stream norms for error memory."""
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        cls, mid_patches, final_patches = self.extractor(x)
        mid_pooled,   mid_w   = self.mid_pool(mid_patches)
        final_pooled, final_w = self.final_pool(final_patches)
        cross_out             = self.cross_attn(cls, final_patches)
        freq_out              = self.freq_stream(x)
        dct_out               = self.dct_block(x)

        stream_norms = {
            'mid_pooled':        mid_pooled.norm(dim=1).mean().item(),
            'final_pooled':      final_pooled.norm(dim=1).mean().item(),
            'cross_out':         cross_out.norm(dim=1).mean().item(),
            'freq_out':          freq_out.norm(dim=1).mean().item(),
            'dct_out':           dct_out.norm(dim=1).mean().item(),
            'mid_attn_entropy':  -(mid_w * (mid_w + 1e-8).log()).sum(dim=1).mean().item(),
            'final_attn_entropy':-(final_w * (final_w + 1e-8).log()).sum(dim=1).mean().item(),
        }
        fused  = self.drop_path(
            torch.cat([mid_pooled, final_pooled, cross_out, freq_out, dct_out], dim=1)
        )
        feat   = self.mlp(fused)
        logits = self.classifier(feat) / self.temperature
        return logits, stream_norms

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
        print(f"  Calibrated temperature: {T_opt:.4f}  (bounded to {temp_range})")
        return T_opt


# ── Factory ────────────────────────────────────────────────────────────────────

def get_spatial_model_v64(
    num_classes:         int   = 2,
    dropout:             float = 0.25,
    drop_path_rate:      float = 0.08,
    freeze_backbone:     bool  = True,
    unfreeze_last_n:     int   = 0,
    use_grad_checkpoint: bool  = True,
    input_size:          int   = 224,
    freq_dim:            int   = 128,
    dct_dim:             int   = 64,
    weights_path:        Optional[str] = None,
    signed:              bool  = False,
) -> SpatialModelV64:
    model = SpatialModelV64(
        num_classes=num_classes, dropout=dropout, drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone, unfreeze_last_n=unfreeze_last_n,
        use_grad_checkpoint=use_grad_checkpoint, input_size=input_size,
        freq_dim=freq_dim, dct_dim=dct_dim,
    )
    if weights_path and os.path.exists(weights_path):
        try:
            if signed:
                ckpt = load_signed(weights_path, strict=False)
            else:
                ckpt = load_checkpoint(weights_path)
            state   = ckpt.get('model_state_dict', ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                print(f"  Missing keys: {len(missing)}")
            if unexpected:
                print(f"  Unexpected keys: {len(unexpected)}")
            print(f"  Loaded weights: {weights_path}")
        except Exception as e:
            print(f"  Could not load weights: {e}")
    return model


# ── Backward-compatible alias ──────────────────────────────────────────────────
# train_v63.py imports SpatialModelV63; this alias lets v6.4 drop in transparently.
SpatialModelV63 = SpatialModelV64
get_spatial_model_v63 = get_spatial_model_v64


# ── Self-test ──────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("SpatialModelV6.4 Self-Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"VRAM: {_vram_gb():.1f} GB")

    model = get_spatial_model_v64(freeze_backbone=True, unfreeze_last_n=0).to(device)
    model.eval()

    print("\nShape tests...")
    for B, H, W in [(1, 224, 224), (4, 224, 224), (2, 128, 128)]:
        x = torch.randn(B, 3, H, W).to(device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 2), f"Expected ({B}, 2), got {out.shape}"
        probs = torch.softmax(out, dim=1)
        print(f"  ({B}, 3, {H}, {W}) -> {out.shape}  FAKE={probs[0,0]:.3f}  REAL={probs[0,1]:.3f}  OK")

    print("\nFrequency stream test...")
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        freq_feat = model.freq_stream(x)
        dct_feat  = model.dct_block(x)
    print(f"  FreqStream: {freq_feat.shape}  DCTBlock: {dct_feat.shape}  OK")

    print("\nForward with entropy test...")
    with torch.no_grad():
        logits, me, fe = model.forward_with_entropy(x)
    print(f"  logits={logits.shape}  mid_entropy={me.item():.4f}  final_entropy={fe.item():.4f}  OK")

    print("\nForward with streams test...")
    with torch.no_grad():
        logits, snorms = model.forward_with_streams(x)
    print(f"  Stream norms: { {k: round(v,3) for k,v in snorms.items()} }")

    print("\nGradCAM test...")
    x_single = torch.randn(1, 3, 224, 224).to(device)
    cam_gen  = GradCAM(model)
    heatmap  = cam_gen.generate(x_single, class_idx=1)
    assert heatmap.shape == (224, 224)
    assert 0.0 <= heatmap.min().item() and heatmap.max().item() <= 1.0
    print(f"  Heatmap: {heatmap.shape}  min={heatmap.min():.4f}  max={heatmap.max():.4f}  OK")

    print("\nAsymmetricFocalLoss test...")
    afl = AsymmetricFocalLoss(gamma_fake=3.0, gamma_real=2.0, weight=None)
    logits_t = torch.randn(8, 2)
    # 4 fake (0), 4 real (1)
    targets  = torch.tensor([0,0,0,0,1,1,1,1])
    loss_val = afl(logits_t, targets)
    print(f"  AFLoss value: {loss_val.item():.4f}  OK")
    afl.update_weight(torch.tensor([1.5, 1.0]))
    print(f"  update_weight OK  weight={afl.weight}")

    print("\nCutMix test...")
    x_cm = torch.randn(4, 3, 224, 224)
    y_cm = torch.tensor([0, 1, 0, 1])
    mix_x, ya, yb, lam = cutmix_data(x_cm, y_cm, alpha=0.5)
    assert mix_x.shape == x_cm.shape
    assert 0.0 <= lam <= 1.0
    print(f"  CutMix lambda={lam:.4f}  OK")

    print("\nError memory test...")
    model.error_memory.record(true_label=0, pred_label=1, confidence=0.75,
                               stream_norms={'freq_out': 1.5, 'mid_pooled': 0.8})
    model.error_memory.record(true_label=0, pred_label=1, confidence=0.68,
                               stream_norms={'dct_out': 1.2, 'final_pooled': 0.6})
    summ = model.error_memory.summary()
    print(f"  Summary: {summ}")
    w = model.error_memory.get_loss_weights([1.0, 1.0])
    print(f"  Adapted weights: FAKE={w[0]:.3f}  REAL={w[1]:.3f}  (FAKE should be >1.0)")
    assert w[0] > 1.0, "Fake weight boost not working"

    print("\nUncertainty test...")
    result = predict_with_uncertainty(model, x_single, n_passes=5)
    print(f"  Pred={result['prediction'].item()}  Conf={result['confidence'].item():.3f}"
          f"  Uncertainty={result['uncertainty'].item():.3f}")

    print("\nCheckpoint save/load test...")
    ckpt_path = '/tmp/test_v64.pt'
    save_checkpoint({'model_state_dict': model.state_dict(), 'epoch': 1,
                     'best_val_acc': 0.93}, ckpt_path)
    loaded = load_checkpoint(ckpt_path)
    assert loaded.get('epoch') == 1
    print(f"  Epoch preserved: {loaded['epoch']}  best_val_acc: {loaded['best_val_acc']}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak VRAM: {peak:.2f} GB")

    print("\n✅ All SpatialModelV6.4 tests passed!")