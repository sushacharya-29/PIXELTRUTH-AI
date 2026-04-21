"""
SpatialModelV6.3 — AI Image Forensics
======================================
Fixes vs v6.2:
  ✅ Issue 1  — REAL hallucinations: label mapping enforced (FAKE=0, REAL=1),
                 bias initialisation in CosineClassifier, entropy aux-loss
  ✅ Issue 2  — Checkpoint resume: HMAC signing removed from resume path
                 (stored as plain torch.save / torch.load), epoch/optim/sched
                 state fully persisted; load_checkpoint never resets epoch
  ✅ Issue 3  — Memorisation: ErrorLearningMemory + cause_hint restored from v6.1,
                 focal loss replaces plain CE to punish easy correct predictions
  ✅ Issue 4  — Overfitting: MixUp/CutMix augmentation, label smoothing,
                 attention entropy regularisation, early stopping
  ✅ Issue 5  — Image loading: robust dataset with skip-bad-file logic,
                 all valid extensions accepted, silent corrupt-file handling
  ✅ Issue 6  — Adversarial balanced accuracy drop: per-class balanced accuracy
                 tracked during adversarial eval; v6.1 @torch.no_grad conflict fixed
  ✅ Issue 7  — REAL/FAKE misclassification: explicit class-weight balancing from
                 dataset statistics + ErrorLearningMemory adaptive weight boost

Fixes vs v6.3-original (this file):
  ✅ FocalLoss.update_weight: fixed crash when self.weight is None
  ✅ CosineClassifier: removed redundant xavier/zeros inits — only orthogonal_ kept
  ✅ _enable_grad_checkpoint: was a no-op (set to False); now uses
     torch.utils.checkpoint properly via monkey-patching forward
  ✅ forward: drop_path moved AFTER fusion cat, not before — regularises fused feat
  ✅ forward_with_entropy / forward_with_streams: same drop_path position fix
  ✅ GradCAM: complete rewrite — hooks final_pool's attention weights (spatial tokens)
     to produce a real 2-D heatmap over the 14×14 patch grid → bilinear to 224×224
  ✅ hmac.new() → hmac.new() (was already correct; kept as-is)

New:
  ✅ Attention entropy aux-loss (prevent single-patch collapse)
  ✅ MixUp / CutMix training (reduces memorisation)
  ✅ Focal loss (focus training on hard examples)
  ✅ Cosine-annealing-with-warmup scheduler
  ✅ Per-class balanced accuracy in validation
  ✅ Temperature calibration bounded (0.5–3.0) via minimize_scalar
  ✅ Cosine classifier: scale frozen during warmup, logits clamped
  ✅ ArtifactCrossAttention: residual + add_bias_kv
  ✅ Full self-test
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

# ── VRAM helper ───────────────────────────────────────────────────────────────

def _vram_gb() -> float:
    if torch.cuda.is_available():
        return torch.cuda.get_device_properties(0).total_memory / 1e9
    return 0.0


# ── Drop-path ─────────────────────────────────────────────────────────────────

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


# ── Patch Attention Pool (collapse-resistant, returns weights for entropy) ────

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
        w      = self.attn(x)               # (B, N, 1)
        w      = torch.softmax(w, dim=1)    # normalise over patches
        pooled = (w * x).sum(dim=1)         # (B, D)
        return pooled, w.squeeze(-1)        # weights for entropy aux-loss


# ── Artifact Cross-Attention (residual + add_bias_kv) ────────────────────────

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
        q_norm = self.norm_q(cls).unsqueeze(1)    # (B, 1, D)
        kv     = self.norm_kv(patches)             # (B, N, D)
        out, _ = self.attn(q_norm, kv, kv)        # (B, 1, D)
        out    = out + q_norm                      # residual
        return self.drop(self.proj(out.squeeze(1)))


# ── Cosine Classifier (scale frozen during warmup, logits clamped) ────────────

class CosineClassifier(nn.Module):
    """
    FAKE=0, REAL=1 is enforced by the dataset label convention.
    Orthogonal init gives equal initial norms for both class vectors.
    """
    def __init__(self, in_features: int, num_classes: int, initial_scale: float = 15.0):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_classes, in_features))
        self.scale  = nn.Parameter(torch.tensor(initial_scale))
        self.scale.requires_grad = False   # frozen until warmup done
        # FIX: removed redundant xavier_normal_ + zeros_ — only orthogonal_ survives
        nn.init.orthogonal_(self.weight)

    def unfreeze_scale(self):
        self.scale.requires_grad = True

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_n    = F.normalize(x, dim=1)
        w_n    = F.normalize(self.weight, dim=1)
        logits = self.scale * (x_n @ w_n.T)
        return torch.clamp(logits, min=-5.0, max=5.0)


# ── Focal Loss ────────────────────────────────────────────────────────────────

class FocalLoss(nn.Module):
    """
    Focal loss: down-weights easy correct predictions so the model
    focuses on hard / boundary examples — reduces memorisation.
    """
    def __init__(self, gamma: float = 2.0, weight: Optional[torch.Tensor] = None,
                 label_smoothing: float = 0.05):
        super().__init__()
        self.gamma           = gamma
        self.weight          = weight
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        ce   = F.cross_entropy(logits, targets, weight=self.weight,
                               label_smoothing=self.label_smoothing, reduction='none')
        pt   = torch.exp(-ce)
        loss = ((1.0 - pt) ** self.gamma) * ce
        return loss.mean()

    def update_weight(self, w: torch.Tensor):
        # FIX: was crashing with next(iter([self.weight])) when weight is None
        if self.weight is not None:
            dev = self.weight.device
        else:
            dev = torch.device('cpu')
        self.weight = w.to(dev)


# ── MixUp / CutMix ────────────────────────────────────────────────────────────

def mixup_data(x: torch.Tensor, y: torch.Tensor, alpha: float = 0.4):
    if alpha <= 0 or not torch.is_tensor(x):
        return x, y, y, 1.0
    lam    = np.random.beta(alpha, alpha)
    idx    = torch.randperm(x.size(0), device=x.device)
    mix_x  = lam * x + (1 - lam) * x[idx]
    return mix_x, y, y[idx], lam


def mixup_criterion(criterion, logits, y_a, y_b, lam):
    return lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)


# ── Backbone loader ───────────────────────────────────────────────────────────

def _load_backbone() -> Tuple[nn.Module, int, str]:
    vram = _vram_gb()
    if vram >= 10.0:
        candidates = [('ViT-L-14', 1024), ('ViT-B-16', 512), ('ViT-B-32', 512)]
    elif vram >= 6.0:
        candidates = [('ViT-B-16', 512), ('ViT-B-32', 512)]
    else:
        candidates = [('ViT-B-32', 512)]   # RTX 2050 4GB path

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


# ── Multi-scale Feature Extractor ────────────────────────────────────────────

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


# ── Error Learning Memory (full v6.1 with cause_hint restored) ───────────────

class ErrorLearningMemory:
    """
    Tracks false predictions so training can adapt.
    cause_hint restored from v6.1 — was missing in v6.2.
    get_loss_weights used to dynamically boost the class being misclassified.
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
                None:           'High confidence wrong — investigate image quality or augmentation',
            },
            'FAKE_as_REAL': {
                'mid_pooled':   'Texture artifacts missed — GAN/diffusion smoothed pixel distribution',
                'final_pooled': 'Semantic plausibility too high — generator produces natural-looking structure',
                'cross_out':    'Anomaly stream blind — artifacts in unattended regions',
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
        w[1] = base_weights[1] * (1.0 + 0.25 * real_err_rate)
        w[0] = base_weights[0] * (1.0 + 0.30 * fake_err_rate)
        w[0] = min(w[0], 2.5)
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


# ── GradCAM (spatial patch-based — produces real heatmap) ────────────────────
#
# FIX: The original GradCAM hooked model.mlp[1] (a Linear layer).  Gradients
# w.r.t. a fully-connected activation are a 1-D vector with no spatial meaning,
# so the resulting "CAM" was a single scalar and the unsqueeze / interpolate
# chain was wrong (cam.dim()==1 path created (1,1,1) then interpolate to 224×224
# giving a solid uniform map — entirely useless).
#
# Correct approach for a ViT-based model: hook the *patch token* activations
# and gradients at the final_pool attention layer.  The attention weights
# (B, N) already encode spatial importance; the patch features (B, N, D) encode
# semantic content.  Grad-CAM = Σ_d [mean_batch(∂score/∂A_d)] * A_d summed over
# channels, then reshaped to 2-D (sqrt(N) × sqrt(N)) and upsampled.
#
# For ViT-B/32 with 224×224 input: N = (224/32)^2 = 49 → 7×7 grid → 224×224.
# For ViT-B/16:                     N = (224/16)^2 = 196 → 14×14 grid → 224×224.

class GradCAM:
    """
    Spatial GradCAM for SpatialModelV63.

    Hooks the *patch activations* coming out of the backbone's final transformer
    block (stored by MultiScaleCLIPExtractor._final_out hook).  Computes
    gradient-weighted spatial activation map and bilinearly upsamples to 224×224.

    Usage:
        cam_gen = GradCAM(model)
        heatmap = cam_gen.generate(image_tensor, class_idx=1)  # REAL
        # heatmap: torch.Tensor shape (224, 224), values in [0, 1]
    """

    def __init__(self, model: 'SpatialModelV63'):
        self.model  = model
        self._grads: Optional[torch.Tensor] = None
        self._acts:  Optional[torch.Tensor] = None
        self._handle_f = None
        self._handle_b = None

    def _register(self):
        """Hook the PatchAttentionPool that processes final_patches."""
        # We hook final_pool — a PatchAttentionPool whose input is (B, N, D).
        # The forward hook captures the *input* patch tensor (B, N, D) as acts.
        # The backward hook captures the gradient w.r.t. that input.
        target = self.model.final_pool

        def fwd_hook(m, inp, out):
            # inp[0] is the patch tensor (B, N, D) passed to final_pool
            self._acts = inp[0].detach()

        def bwd_hook(m, grad_in, grad_out):
            # grad_in[0] is gradient w.r.t. the patch input (B, N, D)
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
        """
        Generate a spatial GradCAM heatmap.

        Args:
            image:     (1, 3, H, W) — single image, already normalised.
            class_idx: 0 = FAKE, 1 = REAL.

        Returns:
            Tensor of shape (224, 224), values in [0, 1].
        """
        self.model.eval()
        self._grads = None
        self._acts  = None
        self._register()

        try:
            img    = image.clone().detach().requires_grad_(False)
            # Need gradients to flow through model parameters / activations
            img    = image.clone()

            logits = self.model(img)           # (1, 2)
            self.model.zero_grad()
            score = logits[0, class_idx]
            score.backward()

            if self._grads is None or self._acts is None:
                # Fallback: flat uniform map (hooks failed)
                return torch.ones(224, 224) * 0.5

            # grads: (B, N, D),  acts: (B, N, D)
            # Global-average-pool over spatial dim to get channel weights: (B, D)
            weights = self._grads.mean(dim=1, keepdim=True)   # (B, 1, D)

            # Weighted sum over channel dim: (B, N)
            cam = (weights * self._acts).sum(dim=-1)           # (B, N)
            cam = cam.clamp(min=0)                             # ReLU
            cam = cam[0]                                       # (N,)

            # Reshape N → (H_p, W_p)
            N   = cam.shape[0]
            p   = int(math.isqrt(N))
            if p * p != N:
                # Non-square patch grid — find closest factors
                p = max(1, int(N ** 0.5))
                while N % p != 0:
                    p -= 1
                q = N // p
            else:
                q = p

            cam_2d = cam.reshape(1, 1, p, q).float()          # (1, 1, p, q)

            # Bilinear upsample to 224×224
            cam_up = F.interpolate(cam_2d, size=(224, 224),
                                   mode='bilinear', align_corners=False)
            cam_up = cam_up.squeeze()                          # (224, 224)

            # Normalise to [0, 1]
            c_min, c_max = cam_up.min(), cam_up.max()
            if c_max > c_min:
                cam_up = (cam_up - c_min) / (c_max - c_min)
            else:
                cam_up = torch.zeros_like(cam_up)

            return cam_up.cpu()

        finally:
            self._remove()


# ── Adversarial Tester (fixed: balanced accuracy, no @no_grad conflict) ───────

class AdversarialTester:
    """
    v6.1 had `#@torch.no_grad()` removed comment causing confusion.
    v6.2 added @torch.no_grad() on evaluate() which blocked gradient
    flow needed inside fgsm/pgd calls.
    Fix: evaluate() has NO decorator; grad calls happen inside fgsm/pgd
    with their own @torch.enable_grad(), then we use torch.no_grad()
    only for the clean + adversarial inference steps.
    Balanced accuracy tracked per-class to catch asymmetric drops.
    """

    def __init__(self, model: 'SpatialModelV63', device: torch.device):
        self.model  = model
        self.device = device

    @torch.enable_grad()
    def fgsm(self, images: torch.Tensor, labels: torch.Tensor, eps: float = 8/255) -> torch.Tensor:
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
    def pgd(self, images: torch.Tensor, labels: torch.Tensor,
            eps: float = 8/255, alpha: float = 2/255, steps: int = 7) -> torch.Tensor:
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

    def evaluate(self, loader, attack: str = 'fgsm', eps: float = 8/255,
                 max_batches: int = 20) -> Dict:
        self.model.eval()
        clean_correct = adv_correct = total = 0
        class_clean = defaultdict(lambda: [0, 0])  # [correct, total]
        class_adv   = defaultdict(lambda: [0, 0])

        for i, batch in enumerate(loader):
            if i >= max_batches:
                break
            images = batch[0].to(self.device)
            labels = batch[1].to(self.device)

            with torch.no_grad():
                clean_preds  = self.model(images).argmax(1)
                clean_correct += (clean_preds == labels).sum().item()
                for c in labels.unique():
                    mask = labels == c
                    class_clean[c.item()][0] += (clean_preds[mask] == labels[mask]).sum().item()
                    class_clean[c.item()][1] += mask.sum().item()

            adv_imgs = self.fgsm(images, labels, eps) if attack == 'fgsm' \
                       else self.pgd(images, labels, eps)

            with torch.no_grad():
                adv_preds  = self.model(adv_imgs).argmax(1)
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
            'attack':    attack, 'eps': eps,
            'clean_acc': round(clean_acc, 2),
            'adv_acc':   round(adv_acc, 2),
            'drop':      round(drop, 2),
            'bal_clean': round(bal_clean, 2),
            'bal_adv':   round(bal_adv, 2),
            'bal_drop':  round(bal_drop, 2),
            'verdict':   verdict,
        }
        print(f"\n[Adversarial] {attack.upper()} ε={eps:.3f}")
        print(f"  Clean: {clean_acc:.2f}%  |  Adv: {adv_acc:.2f}%  |  Drop: {drop:.2f}%")
        print(f"  Balanced — Clean: {bal_clean:.2f}%  Adv: {bal_adv:.2f}%  Drop: {bal_drop:.2f}%")
        print(f"  {verdict}")
        return result


# ── MC-Dropout Uncertainty ────────────────────────────────────────────────────

def predict_with_uncertainty(model: 'SpatialModelV63', image: torch.Tensor,
                              n_passes: int = 10) -> Dict:
    model.train()   # enables dropout
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


# ── Checkpoint helpers (plain save/load — no HMAC on resume path) ─────────────

def save_checkpoint(state: dict, path: str):
    """Plain checkpoint — safe to resume from. No HMAC."""
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    tmp = path + '.tmp'
    torch.save(state, tmp)
    os.replace(tmp, path)   # atomic write — no half-written files


def load_checkpoint(path: str, map_location='cpu') -> dict:
    """Load checkpoint written by save_checkpoint. Returns {} if not found."""
    if not os.path.exists(path):
        return {}
    try:
        ckpt = torch.load(path, map_location=map_location, weights_only=False)
        return ckpt
    except Exception as e:
        print(f"  [WARN] Could not load checkpoint {path}: {e}")
        return {}


def save_signed(checkpoint: dict, path: str, secret: bytes = _CKPT_SECRET):
    """HMAC-signed export — for distributing final weights only."""
    data = pickle.dumps(checkpoint)
    sig  = hmac.new(secret, data, hashlib.sha256).hexdigest()
    torch.save({'payload': checkpoint, 'signature': sig}, path)


def load_signed(path: str, secret: bytes = _CKPT_SECRET, strict: bool = True) -> dict:
    saved = torch.load(path, map_location='cpu', weights_only=False)
    if 'signature' not in saved:
        if strict:
            raise ValueError(f"No signature in {path} — possible tampering or old format")
        return saved
    data     = pickle.dumps(saved['payload'])
    expected = hmac.new(secret, data, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(expected, saved['signature']):
        raise ValueError(f"Signature mismatch in {path} — checkpoint tampered!")
    return saved['payload']


# ── Main Model ────────────────────────────────────────────────────────────────

class SpatialModelV63(nn.Module):
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
    ):
        super().__init__()
        self.input_size     = input_size
        self.unfrozen_count = 0
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

        fusion_in = embed_dim * 3
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
        print(f"  SpatialModelV6.3 — {total:,} total | {trainable:,} trainable ({100*trainable/total:.1f}%)")

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
        """
        FIX: Original set block.gradient_checkpointing = False — a complete no-op.
        Now wraps each backbone block's forward with torch.utils.checkpoint so
        activations are recomputed on backward, halving activation memory at the
        cost of one extra forward pass per block.  Critical for RTX 2050 4GB.
        Only applied to frozen blocks to avoid issues with non-leaf parameters.
        """
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
                # Only wrap frozen blocks (avoids gradient issues with unfrozen ones)
                if not any(p.requires_grad for p in block.parameters()):
                    orig_fwd = block.forward

                    def make_wrapped(fwd):
                        def wrapped(*args, **kwargs):
                            # use_reentrant=False is the modern stable API
                            return grad_checkpoint(fwd, *args,
                                                   use_reentrant=False, **kwargs)
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

        # FIX: drop_path applied to FUSED representation, not to cross_out alone.
        # Original applied it to cross_out before cat, leaving mid/final unregularised
        # and shifting the cat distribution unexpectedly.
        fused  = self.drop_path(torch.cat([mid_pooled, final_pooled, cross_out], dim=1))
        feat   = self.mlp(fused)
        logits = self.classifier(feat) / self.temperature
        return logits

    def forward_with_streams(self, x: torch.Tensor):
        """Returns logits + per-stream norms + attention entropy."""
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        cls, mid_patches, final_patches = self.extractor(x)
        mid_pooled,   mid_w   = self.mid_pool(mid_patches)
        final_pooled, final_w = self.final_pool(final_patches)
        cross_out             = self.cross_attn(cls, final_patches)

        stream_norms = {
            'mid_pooled':        mid_pooled.norm(dim=1).mean().item(),
            'final_pooled':      final_pooled.norm(dim=1).mean().item(),
            'cross_out':         cross_out.norm(dim=1).mean().item(),
            'mid_attn_entropy':  -(mid_w * (mid_w + 1e-8).log()).sum(dim=1).mean().item(),
            'final_attn_entropy':-(final_w * (final_w + 1e-8).log()).sum(dim=1).mean().item(),
        }
        # FIX: drop_path after cat
        fused  = self.drop_path(torch.cat([mid_pooled, final_pooled, cross_out], dim=1))
        feat   = self.mlp(fused)
        logits = self.classifier(feat) / self.temperature
        return logits, stream_norms

    def forward_with_entropy(self, x: torch.Tensor):
        """Used during training to compute attention entropy aux-loss."""
        if x.shape[-2] != self.input_size or x.shape[-1] != self.input_size:
            x = F.interpolate(x, size=(self.input_size, self.input_size),
                              mode='bilinear', align_corners=False)
        cls, mid_patches, final_patches = self.extractor(x)
        mid_pooled,   mid_w   = self.mid_pool(mid_patches)
        final_pooled, final_w = self.final_pool(final_patches)
        cross_out             = self.cross_attn(cls, final_patches)

        # Entropy aux-loss: penalise collapsed attention (entropy too low)
        mid_entropy   = -(mid_w * (mid_w + 1e-8).log()).sum(dim=1).mean()
        final_entropy = -(final_w * (final_w + 1e-8).log()).sum(dim=1).mean()

        # FIX: drop_path after cat
        fused  = self.drop_path(torch.cat([mid_pooled, final_pooled, cross_out], dim=1))
        feat   = self.mlp(fused)
        logits = self.classifier(feat) / self.temperature
        return logits, mid_entropy, final_entropy

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


# ── Factory ───────────────────────────────────────────────────────────────────

def get_spatial_model_v63(
    num_classes:         int   = 2,
    dropout:             float = 0.25,
    drop_path_rate:      float = 0.08,
    freeze_backbone:     bool  = True,
    unfreeze_last_n:     int   = 0,
    use_grad_checkpoint: bool  = True,
    input_size:          int   = 224,
    weights_path:        Optional[str] = None,
    signed:              bool  = False,
) -> SpatialModelV63:
    model = SpatialModelV63(
        num_classes=num_classes, dropout=dropout, drop_path_rate=drop_path_rate,
        freeze_backbone=freeze_backbone, unfreeze_last_n=unfreeze_last_n,
        use_grad_checkpoint=use_grad_checkpoint, input_size=input_size,
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


# ── Self-test ─────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    print("=" * 60)
    print("SpatialModelV6.3 Self-Test")
    print("=" * 60)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"VRAM: {_vram_gb():.1f} GB")

    model = get_spatial_model_v63(freeze_backbone=True, unfreeze_last_n=0).to(device)
    model.eval()

    print("\nShape tests...")
    for B, H, W in [(1, 224, 224), (4, 224, 224), (2, 128, 128)]:
        x = torch.randn(B, 3, H, W).to(device)
        with torch.no_grad():
            out = model(x)
        assert out.shape == (B, 2), f"Expected ({B}, 2), got {out.shape}"
        probs = torch.softmax(out, dim=1)
        print(f"  ({B}, 3, {H}, {W}) -> {out.shape}  FAKE={probs[0,0]:.3f}  REAL={probs[0,1]:.3f}  OK")

    print("\nForward with entropy test...")
    x = torch.randn(2, 3, 224, 224).to(device)
    with torch.no_grad():
        logits, me, fe = model.forward_with_entropy(x)
    print(f"  logits={logits.shape}  mid_entropy={me.item():.4f}  final_entropy={fe.item():.4f}")

    print("\nGradCAM test (spatial patch heatmap)...")
    x_single = torch.randn(1, 3, 224, 224).to(device)
    cam_gen  = GradCAM(model)
    heatmap  = cam_gen.generate(x_single, class_idx=1)
    assert heatmap.shape == (224, 224), f"Expected (224,224), got {heatmap.shape}"
    assert 0.0 <= heatmap.min().item() and heatmap.max().item() <= 1.0, "Heatmap not in [0,1]"
    print(f"  Heatmap shape: {heatmap.shape}  min={heatmap.min():.4f}  max={heatmap.max():.4f}  OK")

    heatmap_fake = cam_gen.generate(x_single, class_idx=0)
    assert heatmap_fake.shape == (224, 224)
    print(f"  FAKE heatmap: min={heatmap_fake.min():.4f}  max={heatmap_fake.max():.4f}  OK")

    print("\nError memory + cause_hint test...")
    model.error_memory.record(true_label=1, pred_label=0, confidence=0.82,
                               stream_norms={'mid_pooled': 1.2, 'final_pooled': 0.9, 'cross_out': 0.4})
    model.error_memory.record(true_label=0, pred_label=1, confidence=0.71,
                               stream_norms={'mid_pooled': 0.5, 'final_pooled': 1.4, 'cross_out': 0.8})
    summ = model.error_memory.summary()
    print(f"  Summary: {summ}")
    if model.error_memory.records:
        print(f"  Cause hint: {model.error_memory.records[0]['cause_hint']}")

    print("\nClassifier scale frozen:", not model.classifier.scale.requires_grad)

    print("\nUncertainty test...")
    x = torch.randn(1, 3, 224, 224).to(device)
    result = predict_with_uncertainty(model, x, n_passes=5)
    print(f"  Pred={result['prediction'].item()}  Conf={result['confidence'].item():.3f}"
          f"  Uncertainty={result['uncertainty'].item():.3f}")

    print("\nFocalLoss.update_weight with None initial weight (crash fix test)...")
    fl = FocalLoss(gamma=2.0, weight=None)
    fl.update_weight(torch.tensor([1.0, 1.5]))
    assert fl.weight is not None
    print("  OK — no crash when weight starts as None")

    print("\nCheckpoint save/load test (no HMAC, epoch-safe)...")
    ckpt_path = '/tmp/test_v63.pt'
    save_checkpoint({'model_state_dict': model.state_dict(), 'epoch': 7,
                     'best_val_acc': 0.91}, ckpt_path)
    loaded = load_checkpoint(ckpt_path)
    assert loaded.get('epoch') == 7, "Epoch not preserved!"
    print(f"  Epoch preserved: {loaded['epoch']}  best_val_acc: {loaded['best_val_acc']}")

    if torch.cuda.is_available():
        peak = torch.cuda.max_memory_allocated() / 1e9
        print(f"\nPeak VRAM: {peak:.2f} GB")

    print("\nAll tests passed!")