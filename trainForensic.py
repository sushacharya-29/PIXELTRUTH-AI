"""
trainForensic.py — Training pipeline for ForensicEngine v3
Hardware: NVIDIA RTX 2050 4GB VRAM + CUDA

Dataset layout:
  data_dir/
    train/real/ + train/fake/
    val/real/ + val/fake/

Labels: FAKE=0, REAL=1

Usage:
  python trainForensic.py --data_dir /path/to/dataset --ckpt_dir ./checkpoints

═══════════════════════════════════════════════════════════════════
FIXES APPLIED (vs original trainForensic.py):
═══════════════════════════════════════════════════════════════════

FIX-1  CRITICAL — fgsm_perturb / pgd_perturb called model() which
       returns raw logits ONLY from forward(), but ForensicEngine's
       standard inference path is forward_with_entropy() → (logits, aux, comp).
       Using model() for adversarial perturbation is correct because
       forward() is defined specifically for single-tensor-out use.
       HOWEVER: the original code called F.cross_entropy(model(imgs_adv), labels)
       which IS correct for ForensicEngine because model.forward() returns
       plain logits (not a tuple). The bug is actually the REVERSE —
       model.forward() is safe here. But to be explicit and futureproof,
       we now use forward_with_entropy and unpack the logits, eliminating
       any ambiguity if forward() is ever changed.

FIX-2  GRADIENT FINITE CHECK LOGIC FLAW — original code called
       scaler.update() even when grads were non-finite (scaler skips
       the step but still needs update()). But it also skipped
       clip_grad_norm_ when non-finite, which is correct. The subtle
       bug: if ANY param has a None grad (e.g. frozen params), the
       `all(p.grad is not None and ...)` generator short-circuits
       False even when all *active* grads are fine. Fixed by checking
       only params that have non-None grads.

FIX-3  SCHEDULER IS WEAK AFTER UNFREEZING — when backbone blocks are
       unfrozen at unfreeze_epoch, the new 'backbone_unfrozen' param
       group is added to the optimizer AFTER the scheduler was built.
       LambdaLR only wraps groups present at construction time, so
       the new group gets NO LR scheduling — it stays at its initial
       lr forever. Fixed by rebuilding the scheduler after unfreezing,
       preserving the cosine phase position.

FIX-4  VALIDATION OOM — val_loader used batch_size*2 which can OOM on
       4GB VRAM when model is in eval mode (no grad, but activations
       still allocated for forward_with_streams). Fixed to use same
       batch_size as training, with an optional --val_batch_size arg.

FIX-5  NON-FINITE LOSS GUARD — if loss itself is NaN/Inf (e.g. from
       a corrupt image producing extreme activations), the original
       code would call loss.backward() and corrupt the model. Added
       a guard that skips the backward pass and warns.

FIX-6  AMP + ADVERSARIAL INTERACTION — adversarial perturbation runs
       outside autocast (correct in original), but the perturbed images
       were computed with float32 while the subsequent forward pass
       ran under autocast (fp16). The AMP context must be explicitly
       exited before adversarial gradient computation and re-entered
       for the adversarial forward pass. Separated clearly.

FIX-7  SCALER.UPDATE() ALWAYS CALLED — PyTorch docs require
       scaler.update() to be called every step regardless of whether
       scaler.step() was skipped, to allow the scale factor to recover.
       Original was correct here, but the guard logic was wrong (FIX-2).
       Now both concerns are cleanly separated.

FIX-8  TRAIN-EPOCH ERROR MEMORY CALLED PER-SAMPLE IN A LOOP — the
       original looped `for j in range(len(labels)): model.step_error_memory(...)`
       calling step_error_memory one sample at a time. Each call also
       updates the classifier bias, so bias can be adjusted multiple
       times per batch. Fixed to batch-process error memory and call
       bias adjustment once per batch.

FIX-9  MISSING NON-FINITE GUARD FOR AUX LOSS — `aux` from
       forward_with_entropy can be NaN if evidential heads produce
       degenerate outputs early in training. Added a nan-guard before
       adding aux to the total loss.

FIX-10 VAL FORWARD_WITH_STREAMS CALLED WITH NO-GRAD BUT ALSO CALLS
       STREAM ROUTER UPDATE (side effect inside no_grad) — this is fine
       in PyTorch but updating the EMA router during validation skews
       routing stats since val batches aren't representative of training
       distribution. Validation now uses model.forward() + softmax for
       clean metrics, and uses forward_with_streams only for the
       router-reporting call every N epochs.

FIX-11 EARLY STOPPING MONITOR — original monitored val bal_acc which
       can oscillate. Added a smoothed EMA-based monitor to avoid
       saving on lucky spikes. Best model is still saved on best
       raw bal_acc but early stopping counter uses a stricter smoothed
       threshold.

FIX-12 OPTIMIZER PARAM GROUP DEDUPLICATION — build_optimizer assigned
       'other' group as a catch-all for params not in any named group.
       If generator_head params were not in any explicit group, they'd
       end up in 'other' with full LR. Now generator_head has its own
       group at lr*0.5 for stable auxiliary training.
"""

import os, sys, io, json, math, time, random, argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional   

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from forensicEngine import (
    ForensicEngine, get_forensic_engine,
    AsymmetricFocalLoss, DeepPostprocessingAug,
    mixup_data, cutmix_data, mixup_criterion,
    save_checkpoint, load_checkpoint,
    predict_with_uncertainty, AdversarialTester,
    CLIP_MEAN, CLIP_STD,
)


# ── Reproducibility ────────────────────────────────────────────────────────────
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
BLD  = "\033[1m"
RST  = "\033[0m"
def _hdr(msg):  print(f"\n{BLD}{CYN}{msg}{RST}")
def set_seed(seed=42):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True

# ── Dataset ────────────────────────────────────────────────────────────────────

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}


def _find_class_dirs(root: Path):
    dirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    fd = dirs.get('fake') or dirs.get('0')
    rd = dirs.get('real') or dirs.get('1')
    if fd is None or rd is None:
        raise FileNotFoundError(
            f"Expected fake/real subdirs in {root}. Found: {list(dirs.keys())}")
    return {0: fd, 1: rd}


class ForensicDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transform
        self.samples = []
        self.skipped = 0
        rp = Path(root)
        cd = _find_class_dirs(rp)
        for label, d in cd.items():
            for ext in VALID_EXTS:
                for fp in d.rglob(f'*{ext}'):
                    self.samples.append((str(fp), label))
                for fp in d.rglob(f'*{ext.upper()}'):
                    self.samples.append((str(fp), label))
        seen, unique = set(), []
        for s in self.samples:
            if s[0] not in seen:
                seen.add(s[0]); unique.append(s)
        self.samples = unique
        random.shuffle(self.samples)
        self.class_counts: Dict[int, int] = defaultdict(int)
        for _, l in self.samples:
            self.class_counts[l] += 1
        print(f"  Dataset [{rp.name}]: "
              f"FAKE={self.class_counts[0]}  "
              f"REAL={self.class_counts[1]}  "
              f"Total={len(self.samples)}")

    def get_class_weights(self):
        n = len(self.samples)
        nf = max(self.class_counts[0], 1)
        nr = max(self.class_counts[1], 1)
        # FAKE cap 2.0x, REAL cap 5.0x — prevents FAKE dominance
        wf = min(n / (2. * nf), 2.0)
        wr = min(n / (2. * nr), 5.0)
        print(f"  Class weights — FAKE:{wf:.3f}  REAL:{wr:.3f}")
        return torch.tensor([wf, wr], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            self.skipped += 1
            img = Image.new('RGB', (224, 224), (0, 0, 0))
        if self.transform:
            img = self.transform(img)
        return img, label, path


def collate_fn(batch):
    imgs   = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths  = [b[2] for b in batch]
    return imgs, labels, paths


# ── Transforms ─────────────────────────────────────────────────────────────────

class _JPEGCompress:
    def __init__(self, lo=60, hi=95, p=0.15):
        self.lo = lo; self.hi = hi; self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, 'JPEG', quality=random.randint(self.lo, self.hi))
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class _DownUpscale:
    def __init__(self, lo=0.65, hi=0.9, p=0.10):
        self.lo = lo; self.hi = hi; self.p = p
    def __call__(self, img):
        if random.random() > self.p:
            return img
        W, H = img.size
        sc = random.uniform(self.lo, self.hi)
        s  = img.resize((max(1, int(W * sc)), max(1, int(H * sc))), Image.BILINEAR)
        return s.resize((W, H), Image.BILINEAR)


class _GaussianNoise:
    def __init__(self, lo=0.003, hi=0.012, p=0.20):
        self.lo = lo; self.hi = hi; self.p = p
    def __call__(self, t):
        if random.random() > self.p:
            return t
        return t + torch.randn_like(t) * random.uniform(self.lo, self.hi)


def get_transforms(input_size=224, is_train=True):
    mean, std = CLIP_MEAN, CLIP_STD
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.05),
            transforms.ColorJitter(brightness=0.08, contrast=0.08,
                                   saturation=0.05, hue=0.02),
            transforms.RandomGrayscale(p=0.02),
            transforms.RandomApply([transforms.GaussianBlur(3)], p=0.06),
            _DownUpscale(p=0.08),
            _JPEGCompress(p=0.15),
            transforms.ToTensor(),
            _GaussianNoise(p=0.12),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.04, scale=(0.02, 0.05)),
        ])
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])


# ── LR Schedule ────────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs, total_epochs,
                                 min_lr_ratio=0.05):
    """
    Standard warmup-cosine decay. Returns a LambdaLR.
    Note: when new param groups are added (backbone unfreeze), call
    _rebuild_scheduler() so the new group is also covered.
    """
    def _lr(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        p = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1. - min_lr_ratio) * 0.5 * (1. + math.cos(math.pi * p))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


def _rebuild_scheduler(optimizer, current_epoch, warmup_epochs, total_epochs,
                        min_lr_ratio=0.05):
    """
    FIX-3: Rebuild the scheduler after adding a new param group so the
    new group participates in cosine decay from the current phase.
    The lambda captures current_epoch as the offset so the new schedule
    continues from the same point in the cosine curve.
    """
    offset = current_epoch

    def _lr(epoch):
        real_epoch = epoch + offset
        if real_epoch < warmup_epochs:
            return (real_epoch + 1) / max(warmup_epochs, 1)
        p = (real_epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1. - min_lr_ratio) * 0.5 * (1. + math.cos(math.pi * p))

    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


# ── Adversarial perturbation ───────────────────────────────────────────────────
#
# FIX-1: Original called F.cross_entropy(model(imgs_adv), labels).backward()
# model() returns raw logits (forward() is the plain logit path in ForensicEngine),
# so that was actually type-correct. But it is fragile — if forward() is ever
# changed to return a tuple, this silently breaks. We now explicitly use
# forward_with_entropy and unpack, making the intent clear and future-safe.
# Additionally, the original fgsm used imgs.grad instead of imgs_adv.grad —
# that was a silent correctness bug since imgs doesn't have requires_grad=True.

@torch.enable_grad()
def fgsm_perturb(model: ForensicEngine, imgs: torch.Tensor,
                 labels: torch.Tensor, eps: float) -> torch.Tensor:
    """
    FGSM adversarial perturbation.
    Runs in eval mode, returns perturbed images detached.
    """
    was_training = model.training
    model.eval()

    imgs_adv = imgs.clone().detach().float().requires_grad_(True)
    # FIX-1: Explicitly unpack logits from forward_with_entropy
    logits, _, _ = model.forward_with_entropy(imgs_adv, labels)
    loss = F.cross_entropy(logits, labels)
    loss.backward()

    with torch.no_grad():
        # FIX-1 (secondary): use imgs_adv.grad, not imgs.grad
        adv = (imgs + eps * imgs_adv.grad.sign()).clamp(-3., 3.)

    if was_training:
        model.train()
    return adv.detach()


@torch.enable_grad()
def pgd_perturb(model: ForensicEngine, imgs: torch.Tensor,
                labels: torch.Tensor, eps: float,
                alpha: float = None, steps: int = 3) -> torch.Tensor:
    """
    PGD adversarial perturbation.
    Runs in eval mode, returns perturbed images detached.
    """
    if alpha is None:
        alpha = eps / 2.0
    was_training = model.training
    model.eval()

    adv = (imgs.clone().detach().float()
           + torch.empty_like(imgs).uniform_(-eps, eps)).clamp(-3., 3.)

    for _ in range(steps):
        adv = adv.detach().requires_grad_(True)
        # FIX-1: Explicitly unpack logits from forward_with_entropy
        logits, _, _ = model.forward_with_entropy(adv, labels)
        loss = F.cross_entropy(logits, labels)
        loss.backward()
        with torch.no_grad():
            adv = torch.min(
                torch.max(adv + alpha * adv.grad.sign(), imgs - eps),
                imgs + eps
            ).clamp(-3., 3.)

    if was_training:
        model.train()
    return adv.detach()


# ── Metrics ────────────────────────────────────────────────────────────────────

def balanced_accuracy(preds, labels):
    per = []
    for c in range(2):
        m = labels == c
        if m.sum() > 0:
            per.append((preds[m] == labels[m]).mean())
    return float(np.mean(per)) if per else 0.


# ── Optimizer ──────────────────────────────────────────────────────────────────

def build_optimizer(model: ForensicEngine, lr: float, weight_decay: float):
    """
    FIX-12: Added explicit generator_head param group at lr*0.5.
    Previously it fell through to 'other' and trained at full lr,
    destabilizing the auxiliary branch during early epochs.
    """
    backbone_params = [p for p in model.extractor.parameters() if p.requires_grad]

    forensic_params = []
    for attr in ('freq_stream', 'dct_block', 'fft_stream', 'noise_block',
                 'jpeg_block', 'ltc_stream', 'ca_stream', 'sfs_stream'):
        m = getattr(model, attr, None)
        if m is not None:
            forensic_params += list(m.parameters())

    fusion_params  = list(model.stream_fusion.parameters())
    head_params    = (list(model.early_pool.parameters())
                      + list(model.mid_pool.parameters())
                      + list(model.final_pool.parameters())
                      + list(model.cross_attn.parameters())
                      + list(model.mlp.parameters()))
    cls_params     = list(model.classifier.parameters())

    # FIX-12: explicit generator_head group
    gen_params = []
    if getattr(model, 'use_generator_head', False) and hasattr(model, 'generator_head'):
        gen_params = list(model.generator_head.parameters())

    assigned = {id(p) for g in [backbone_params, forensic_params, fusion_params,
                                  head_params, cls_params, gen_params] for p in g}
    other = [p for p in model.parameters() if p.requires_grad and id(p) not in assigned]

    groups = [
        {'params': backbone_params, 'lr': lr * 0.05,  'name': 'backbone'},
        {'params': forensic_params, 'lr': lr,          'name': 'forensic_streams'},
        {'params': fusion_params,   'lr': lr * 0.67,   'name': 'stream_fusion'},
        {'params': head_params,     'lr': lr,           'name': 'head'},
        {'params': cls_params,      'lr': lr * 0.33,   'name': 'classifier'},
        {'params': gen_params,      'lr': lr * 0.50,   'name': 'generator_head'},
        {'params': other,           'lr': lr,           'name': 'other'},
    ]
    return torch.optim.AdamW(
        [g for g in groups if g['params']],
        weight_decay=weight_decay
    )


# ── Training step ──────────────────────────────────────────────────────────────

def train_epoch(model: ForensicEngine, aug, loader, optimizer, criterion,
                device, mixup_alpha=0.0, aux_weight=1.0, entropy_weight=0.03,
                scaler=None, amp_enabled=False,
                adv_eps=0.0, adv_weight=0.25, adv_mode='fgsm'):
    model.train()
    aug.train()
    total_loss = 0.
    all_preds  = []
    all_labels = []
    n_batches  = 0

    pbar = tqdm(loader, desc="Training", leave=False)

    for imgs, labels, _ in pbar:
        nb = imgs.is_pinned()
        imgs   = imgs.to(device, non_blocking=nb)
        labels = labels.to(device, non_blocking=nb)

        with torch.no_grad():
            imgs = aug(imgs)

        # REAL-guard MixUp: skip mixing for near-pure-REAL batches
        _real_frac = (labels == 1).float().mean().item()
        if mixup_alpha > 0 and _real_frac < 0.9:
            if random.random() < 0.5:
                imgs, ya, yb, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            else:
                _ca = 0.3 if _real_frac > 0.6 else 0.5
                imgs, ya, yb, lam = cutmix_data(imgs, labels, alpha=_ca)
        else:
            ya, yb, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            # ── AMP path ──────────────────────────────────────────────────────
            with torch.amp.autocast('cuda'):
                logits, aux, comp = model.forward_with_entropy(imgs, labels)
                cls_loss = (
                    mixup_criterion(criterion, logits, ya, yb, lam)
                    if (mixup_alpha > 0 and lam < 1.)
                    else criterion(logits, labels)
                )
                e_ent = comp.get('e_ent', 0.)
                m_ent = comp.get('m_ent', 0.)
                f_ent = comp.get('f_ent', 0.)
                # Negative entropy regulariser — encourages diverse patch attention
                ent_reg = -entropy_weight * (e_ent + m_ent + f_ent)

            if adv_eps > 0:
                # FIX-6: adversarial perturbation runs OUTSIDE autocast.
                # fgsm/pgd_perturb temporarily set model to eval mode
                # and compute float32 gradients; they restore train mode on exit.
                if adv_mode == 'pgd':
                    imgs_adv = pgd_perturb(model, imgs, labels, adv_eps)
                else:
                    imgs_adv = fgsm_perturb(model, imgs, labels, adv_eps)

                with torch.amp.autocast('cuda'):
                    adv_logits, _, _ = model.forward_with_entropy(imgs_adv, labels)
                    adv_loss = criterion(adv_logits, labels)

                # FIX-9: guard against NaN aux (evidential heads can diverge early)
                aux_safe = aux if torch.isfinite(aux) else torch.tensor(0., device=device)
                loss = ((1. - adv_weight) * cls_loss
                        + adv_weight * adv_loss
                        + aux_weight * aux_safe
                        + ent_reg)
            else:
                aux_safe = aux if torch.isfinite(aux) else torch.tensor(0., device=device)
                loss = cls_loss + aux_weight * aux_safe + ent_reg

            # FIX-5: skip backward on NaN/Inf total loss (corrupt batch guard)
            if not torch.isfinite(loss):
                print(f"  ⚠  Non-finite loss ({loss.item():.4f}) — skipping batch")
                scaler.update()   # FIX-7: always call update to recover scale factor
                continue

            pbar.set_postfix({'loss': f"{loss.item():.4f}"})
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)

            # FIX-2: check only params that actually received a gradient.
            # The original `all(p.grad is not None and ...)` returns False
            # for frozen params (grad=None), incorrectly blocking the step.
            grads_ok = all(
                torch.isfinite(p.grad).all()
                for p in model.parameters()
                if p.grad is not None
            )
            if grads_ok:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
            else:
                print("  ⚠  Non-finite grads detected — skipping optimizer step")

            # FIX-7: scaler.update() must ALWAYS be called regardless of whether
            # scaler.step() ran, so the internal scale factor can recover.
            scaler.update()

        else:
            # ── No-AMP path ───────────────────────────────────────────────────
            logits, aux, comp = model.forward_with_entropy(imgs, labels)
            cls_loss = (
                mixup_criterion(criterion, logits, ya, yb, lam)
                if (mixup_alpha > 0 and lam < 1.)
                else criterion(logits, labels)
            )
            e_ent = comp.get('e_ent', 0.)
            m_ent = comp.get('m_ent', 0.)
            f_ent = comp.get('f_ent', 0.)
            ent_reg = -entropy_weight * (e_ent + m_ent + f_ent)

            # FIX-9: NaN aux guard
            aux_safe = aux if torch.isfinite(aux) else torch.tensor(0., device=device)

            if adv_eps > 0:
                imgs_adv = (pgd_perturb(model, imgs, labels, adv_eps)
                            if adv_mode == 'pgd'
                            else fgsm_perturb(model, imgs, labels, adv_eps))
                adv_logits, _, _ = model.forward_with_entropy(imgs_adv, labels)
                loss = ((1. - adv_weight) * cls_loss
                        + adv_weight * criterion(adv_logits, labels)
                        + aux_weight * aux_safe
                        + ent_reg)
            else:
                loss = cls_loss + aux_weight * aux_safe + ent_reg

            # FIX-5: NaN/Inf total loss guard
            if not torch.isfinite(loss):
                print(f"  ⚠  Non-finite loss ({loss.item():.4f}) — skipping batch")
                continue

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            pbar.set_postfix({'loss': f"{loss.item():.4f}"})

        model.training_step_hook()
        total_loss += loss.item()

        with torch.no_grad():
            preds = logits.argmax(1)
            probs = torch.softmax(logits, 1)
            confs = probs.max(1).values

            # FIX-8: batch the error-memory update; call bias adjustment ONCE per batch,
            # not once per sample. Original loop called step_error_memory individually
            # which adjusted the classifier bias up to batch_size times per batch.
            _sn = ({k: v for k, v in comp.items() if isinstance(v, (int, float))}
                   if isinstance(comp, dict) else {})
            for j in range(len(labels)):
                model.error_memory.record(
                    int(labels[j].cpu()), int(preds[j].cpu()),
                    float(confs[j].cpu()), _sn)
            # Single bias adjustment per batch (after recording all samples)
            delta = model.error_memory.suggest_bias_adjustment()
            if abs(delta) > 0.015:
                model.classifier.adapt_bias(delta)
            # Confidence tracker still updated per sample (it's just a deque append)
            for c in confs.cpu():
                model.confidence_tracker.update(float(c))

        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        n_batches += 1

    ap = np.array(all_preds)
    al = np.array(all_labels)
    return {
        'loss':    total_loss / max(n_batches, 1),
        'acc':     (ap == al).mean() * 100,
        'bal_acc': balanced_accuracy(ap, al) * 100,
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(model: ForensicEngine, loader, criterion, device,
             update_router: bool = False):
    """
    FIX-10: Validation now uses model.forward() (plain logit path) for clean
    metrics, avoiding the side-effect of updating the AdaptiveStreamRouter
    EMA with val-set statistics. Router update is optional and only done
    every N epochs via update_router=True.
    """
    model.eval()
    total_loss = 0.
    all_preds  = []
    all_labels = []
    all_confs  = []
    n_batches  = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for imgs, labels, paths in pbar:
            nb = imgs.is_pinned()
            imgs   = imgs.to(device, non_blocking=nb)
            labels = labels.to(device, non_blocking=nb)

            if update_router:
                # Heavy path: updates stream router (only every N epochs)
                logits, info = model.forward_with_streams(imgs)
                _sn = {k: v for k, v in info.items() if isinstance(v, (int, float))}
            else:
                # FIX-10: light path, no router side-effect
                logits = model(imgs)
                _sn = {}

            loss = criterion(logits, labels)
            total_loss += loss.item()
            pbar.set_postfix({'val_loss': f"{loss.item():.4f}"})

            probs = torch.softmax(logits, 1)
            preds = probs.argmax(1)
            confs = probs.max(1).values

            # Error memory update with val predictions (monitoring only — no bias adjust)
            for j in range(len(labels)):
                model.error_memory.record(
                    int(labels[j].cpu()), int(preds[j].cpu()),
                    float(confs[j].cpu()), _sn)
            for c in confs.cpu():
                model.confidence_tracker.update(float(c))

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            n_batches += 1

    ap = np.array(all_preds)
    al = np.array(all_labels)
    acc = (ap == al).mean() * 100
    bal = balanced_accuracy(ap, al) * 100

    fm = al == 0; rm = al == 1
    fr = (ap[fm] == 0).mean() * 100 if fm.sum() > 0 else 0.
    rr = (ap[rm] == 1).mean() * 100 if rm.sum() > 0 else 0.

    if rr < 70.:
        print(f"  ⚠  REAL recall LOW: {rr:.1f}%")
    if fr < 80.:
        print(f"  ⚠  FAKE recall LOW: {fr:.1f}%")

    shift = model.confidence_tracker.check_shift()
    if shift.get('shift_detected'):
        print(f"  ⚠  {shift.get('alert', '')}")

    return {
        'loss':        total_loss / max(n_batches, 1),
        'acc':         acc,
        'bal_acc':     bal,
        'fake_recall': fr,
        'real_recall': rr,
    }


# ── Val dir resolver ───────────────────────────────────────────────────────────

def _resolve_val_dir(data_dir, preferred='val'):
    for name in (preferred, 'test', 'val'):
        c = os.path.join(data_dir, name)
        if os.path.isdir(c):
            if name != preferred:
                print(f"  [INFO] Val dir '{preferred}' not found — using '{name}'")
            return c
    raise FileNotFoundError(f"No val dir in {data_dir}")


# ══════════════════════════════════════════════════════════════════════
# Main training loop
# ══════════════════════════════════════════════════════════════════════

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        p = torch.cuda.get_device_properties(0)
        print(f"GPU: {p.name}  VRAM: {p.total_memory / 1e9:.1f}GB")

    ckpt_dir  = os.path.abspath(args.ckpt_dir)
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_path  = os.path.join(ckpt_dir, 'latest.pt')
    best_path  = os.path.join(ckpt_dir, 'best.pt')
    final_path = os.path.join(ckpt_dir, 'final.pt')
    log_path   = os.path.join(ckpt_dir, 'train_log.jsonl')
    err_path   = os.path.join(ckpt_dir, 'error_memory.json')

    train_tf = get_transforms(args.input_size, True)
    val_tf   = get_transforms(args.input_size, False)
    train_ds = ForensicDataset(os.path.join(args.data_dir, 'train'), train_tf)
    val_ds   = ForensicDataset(_resolve_val_dir(args.data_dir, args.val_split), val_tf)

    _lkw = dict(num_workers=args.workers, pin_memory=args.workers > 0,
                collate_fn=collate_fn, persistent_workers=args.workers > 0)
    if args.workers > 0:
        _lkw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=True, **_lkw)

    # FIX-4: val batch size defaults to training batch size (not *2).
    # On 4GB VRAM, doubling batch during eval can OOM because eval still
    # allocates full intermediate activations. Use --val_batch_size to
    # increase if you have headroom.
    val_bs = args.val_batch_size if args.val_batch_size > 0 else args.batch_size
    val_loader = DataLoader(
        val_ds, batch_size=val_bs,
        shuffle=False, **_lkw)

    model = get_forensic_engine(
        freeze_backbone=True, unfreeze_last_n=0,
        dropout=args.dropout, drop_path_rate=args.drop_path,
        input_size=args.input_size, use_grad_checkpoint=True,
        use_isfcr=True, use_fcw=True, n_iters=args.n_iters,
        evidential_coeff=args.evidential_coeff,
        use_ltc=True, use_ca=True, use_sfs=True,
    ).to(device)

    aug = DeepPostprocessingAug(
        style_jitter_prob=0.10, latent_grid_prob=0.08,
        local_blur_prob=0.10, tone_map_prob=0.10,
    ).to(device)

    base_weights = train_ds.get_class_weights()
    criterion = AsymmetricFocalLoss(
        gamma_fake=args.focal_gamma_fake,
        gamma_real=args.focal_gamma_real,
        weight=base_weights.to(device),
        label_smoothing=args.label_smoothing,
        real_floor_weight=0.05,
    )

    optimizer = build_optimizer(model, args.lr, args.weight_decay)

    # FIX-3: initial scheduler (covers all groups present at construction)
    scheduler = cosine_schedule_with_warmup(
        optimizer, args.warmup_epochs, args.epochs)

    amp_enabled = device.type == 'cuda'
    scaler = (torch.amp.GradScaler('cuda', enabled=amp_enabled)
              if amp_enabled else None)

    start_epoch   = 0
    best_val_acc  = 0.
    _val_acc_ema  = 0.   # FIX-11: smoothed monitor for early stopping

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    if ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except Exception:
            print("  ⚠  Optimizer state mismatch — skipping")
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except Exception:
            print("  ⚠  Scheduler state mismatch — skipping")
        _ep = ckpt.get('epoch', 0)
        start_epoch  = (int(_ep) + 1) if str(_ep).isdigit() else 0
        best_val_acc = ckpt.get('best_val_acc', 0.)
        _val_acc_ema = ckpt.get('val_acc_ema', best_val_acc)
        print(f"  Resumed epoch {_ep}  best_bal={best_val_acc:.2f}%")
        model.error_memory.load(err_path)
    else:
        print("  No checkpoint — fresh start")

    # ── Unfreeze + scheduler rebuild ──────────────────────────────────────────

    def _maybe_unfreeze(epoch, current_scheduler):
        """
        FIX-3: After adding the backbone_unfrozen param group, rebuild the
        scheduler so the new group is covered by cosine decay.
        Returns (possibly new) scheduler.
        """
        if epoch == args.unfreeze_epoch and model.unfrozen_count == 0:
            model._unfreeze_last_n(args.unfreeze_last_n)
            model.classifier.unfreeze_scale()
            new_bb = [p for p in model.extractor.parameters() if p.requires_grad]
            if new_bb:
                optimizer.add_param_group({
                    'params': new_bb,
                    'lr': args.lr * 0.01,
                    'name': 'backbone_unfrozen',
                })
                # FIX-3: rebuild scheduler to cover the new param group,
                # resuming cosine from the current epoch position
                new_sched = _rebuild_scheduler(
                    optimizer, epoch,
                    args.warmup_epochs, args.epochs)
                print(f"  Epoch {epoch}: backbone last {args.unfreeze_last_n} "
                      f"blocks unfrozen (lr={args.lr * 0.01:.2e}) — "
                      f"scheduler rebuilt from epoch {epoch}")
                return new_sched

        # Stage 2: lift backbone LR 10 epochs after unfreeze
        if epoch == args.unfreeze_epoch + 10 and model.unfrozen_count > 0:
            for pg in optimizer.param_groups:
                if pg.get('name') == 'backbone_unfrozen':
                    pg['lr'] = args.lr * 0.03
                    print(f"  Epoch {epoch}: backbone lr lifted to "
                          f"{pg['lr']:.2e} (stage 2)")

        return current_scheduler

    no_improve    = 0
    patience      = args.patience
    _real_recall_ema = 0.
    _rr_alpha        = 0.3

    print("\n",f"Training {args.epochs} epochs (start={start_epoch})")
    print(f"  batch={args.batch_size}  val_batch={val_bs}  lr={args.lr}  "
          f"adv_mode={args.adv_mode}  "
          f"gamma_fake={args.focal_gamma_fake}  gamma_real={args.focal_gamma_real}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()

        # FIX-3: capture possibly-updated scheduler
        scheduler = _maybe_unfreeze(epoch, scheduler)

        # Adaptive class weights — only after warmup AND not in collapse
        _in_collapse = (
            _real_recall_ema < 15.
            or (epoch > 2 and _real_recall_ema < 5.)
        )
        if epoch >= args.warmup_epochs and not _in_collapse:
            aw = model.error_memory.get_loss_weights(
                [base_weights[0].item(), base_weights[1].item()])
            criterion.update_weight(
                torch.tensor(aw, dtype=torch.float32).to(device))
        else:
            criterion.update_weight(base_weights.to(device))

        # REAL recall adaptive floor — graduated response
        if epoch > 0 and hasattr(criterion, 'rfw'):
            if   _real_recall_ema < 50.:  criterion.rfw = 0.15
            elif _real_recall_ema < 75.:  criterion.rfw = 0.08
            elif _real_recall_ema < 85.:  criterion.rfw = 0.05
            else:                         criterion.rfw = 0.03

        # MixUp ramp: disabled during warmup+2, then linear ramp over 5 epochs
        _mx_start  = args.warmup_epochs + 2
        mixup_cur  = (0. if epoch < _mx_start
                      else args.mixup_alpha * min(1., (epoch - _mx_start) / 5.))

        # Adversarial eps ramp
        _adv_ramp   = max(args.warmup_epochs + 5, 12)
        adv_eps_cur = 0.
        if args.adv_eps > 0. and epoch >= args.warmup_epochs:
            adv_eps_cur = args.adv_eps * min(
                1., (epoch - args.warmup_epochs) / max(_adv_ramp, 1))

        # Alternate FGSM/PGD: PGD every 3rd epoch (stronger but more expensive)
        _adv_mode = args.adv_mode
        if _adv_mode == 'auto':
            _adv_mode = 'pgd' if (epoch % 3 == 0 and adv_eps_cur > 0.) else 'fgsm'

        train_stats = train_epoch(
            model=model, aug=aug, loader=train_loader,
            optimizer=optimizer, criterion=criterion, device=device,
            mixup_alpha=mixup_cur, aux_weight=args.aux_weight,
            entropy_weight=args.entropy_weight, scaler=scaler,
            amp_enabled=amp_enabled, adv_eps=adv_eps_cur,
            adv_weight=args.adv_weight, adv_mode=_adv_mode,
        )

        # FIX-10: update router only every 5 epochs to avoid val-set bias in EMA
        _update_router = (epoch % 5 == 0)
        val_stats = validate(model, val_loader, criterion, device,
                             update_router=_update_router)

        scheduler.step()

        _real_recall_ema = ((1. - _rr_alpha) * _real_recall_ema
                            + _rr_alpha * val_stats['real_recall'])
        elapsed  = time.time() - t0
        lr_now   = optimizer.param_groups[0]['lr']

        print(f"Epoch {epoch:03d}/{args.epochs - 1} | "
              f"T_loss={train_stats['loss']:.4f}  "
              f"T_acc={train_stats['acc']:.1f}%  "
              f"T_bal={train_stats['bal_acc']:.1f}% | "
              f"V_loss={val_stats['loss']:.4f}  "
              f"V_acc={val_stats['acc']:.1f}%  "
              f"V_bal={val_stats['bal_acc']:.1f}%  "
              f"FAKE_R={val_stats['fake_recall']:.1f}%  "
              f"REAL_R={val_stats['real_recall']:.1f}%  "
              f"REAL_EMA={_real_recall_ema:.1f}%  "
              f"mx={mixup_cur:.2f} | "
              f"LR={lr_now:.2e}  {elapsed:.0f}s")

        if epoch % 5 == 0:
            print(model.get_routing_report())
            low = model.get_low_importance_streams(threshold=0.03)
            if low:
                print(f"  Low streams (EMA<0.03): {low}")

        err = model.error_memory.summary()
        print(f"  ErrorMem: real_as_fake={err['real_as_fake']}  "
              f"fake_as_real={err['fake_as_real']}  "
              f"ema_fake_err={err['ema_fake_err_rate']:.3f}  "
              f"ema_real_err={err['ema_real_err_rate']:.3f}")

        log_entry = {
            'epoch': epoch,
            **train_stats,
            **{f'val_{k}': v for k, v in val_stats.items()},
            'lr':             lr_now,
            'mixup':          mixup_cur,
            'adv_eps':        adv_eps_cur,
            'real_recall_ema': round(_real_recall_ema, 2),
            **{f'err_{k}': v for k, v in err.items()
               if isinstance(v, (int, float))},
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        cs = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc':         best_val_acc,
            'val_acc_ema':          _val_acc_ema,
            'val_stats':            val_stats,
        }
        save_checkpoint(cs, ckpt_path)

        monitor = val_stats['bal_acc']

        # FIX-11: EMA-smoothed monitor for early stopping (avoids lucky-spike saves)
        _ema_alpha  = 0.4
        _val_acc_ema = (1. - _ema_alpha) * _val_acc_ema + _ema_alpha * monitor

        if monitor > best_val_acc:
            best_val_acc     = monitor
            cs['best_val_acc'] = best_val_acc
            save_checkpoint(cs, best_path)
            print(f"  ✅ New best bal_acc: {best_val_acc:.2f}%  → {best_path}")
            no_improve = 0
        else:
            no_improve += 1
            # FIX-11: early stopping uses EMA threshold — must be meaningfully below best
            if no_improve >= patience and _val_acc_ema < best_val_acc - 1.5:
                print(f"  Early stopping after {patience} epochs without improvement "
                      f"(EMA={_val_acc_ema:.2f}% vs best={best_val_acc:.2f}%)")
                break
            elif no_improve >= patience:
                # Reset counter — EMA still close to best, keep going
                print(f"  No improve counter={no_improve} but EMA={_val_acc_ema:.2f}% "
                      f"still near best — extending patience")
                no_improve = no_improve // 2   # halve rather than reset fully

        model.error_memory.save(err_path)

    # ── Post-training ─────────────────────────────────────────────────────────

    print("\n=== Temperature Calibration ===")
    bc = load_checkpoint(best_path, map_location=device)
    if bc:
        model.load_state_dict(bc['model_state_dict'], strict=False)
    model.calibrate(val_loader, device)

    print("\n=== Stream Pruning ===")
    low = model.get_low_importance_streams(threshold=args.prune_threshold)
    if low:
        print(f"  Pruning: {low}")
        for s in low:
            model.prune_stream(s)
    else:
        print(f"  No streams below {args.prune_threshold}")

    print("\n=== Adversarial Robustness ===")
    adv_tester = AdversarialTester(model, device)
    for attack in ['fgsm', 'pgd']:
        try:
            r = adv_tester.evaluate(
                val_loader, attack=attack, eps=8 / 255, max_batches=10)
            print(f"  {attack.upper()}: adv_acc={r['adv_acc']:.1f}%  "
                  f"bal_drop={r['bal_drop']:.1f}%  {r['verdict']}")
        except Exception as e:
            print(f"  {attack.upper()} eval failed: {e}")

    save_checkpoint({
        'model_state_dict': model.state_dict(),
        'epoch':            'final',
        'best_val_acc':     best_val_acc,
        'error_summary':    model.error_memory.summary(),
        'routing_report':   model.get_routing_report(),
    }, final_path)
    print(f"\nFinal model: {final_path}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    p = argparse.ArgumentParser('ForensicEngine v3 Training')
    p.add_argument('--data_dir',          required=True)
    p.add_argument('--ckpt_dir',          default='./checkpoints')
    p.add_argument('--val_split',         default='val')

    p.add_argument('--epochs',            type=int,   default=30)
    p.add_argument('--batch_size',        type=int,   default=8)
    p.add_argument('--lr',                type=float, default=3e-4)
    p.add_argument('--weight_decay',      type=float, default=1e-4)
    p.add_argument('--dropout',           type=float, default=0.25)
    p.add_argument('--drop_path',         type=float, default=0.08)

    p.add_argument('--focal_gamma_fake',  type=float, default=2.0)
    p.add_argument('--focal_gamma_real',  type=float, default=2.5)
    p.add_argument('--label_smoothing',   type=float, default=0.03)

    p.add_argument('--mixup_alpha',       type=float, default=0.1)
    p.add_argument('--entropy_weight',    type=float, default=0.03)
    p.add_argument('--aux_weight',        type=float, default=1.0)

    p.add_argument('--warmup_epochs',     type=int,   default=3)
    p.add_argument('--unfreeze_epoch',    type=int,   default=10,
                   help='Epoch to unfreeze last N ViT blocks')
    p.add_argument('--unfreeze_last_n',   type=int,   default=3)
    p.add_argument('--patience',          type=int,   default=8)

    p.add_argument('--n_iters',           type=int,   default=2)
    p.add_argument('--evidential_coeff',  type=float, default=0.01)
    p.add_argument('--prune_threshold',   type=float, default=0.03)

    p.add_argument('--adv_eps',           type=float, default=4. / 255,
                   help='Adversarial training epsilon. Set 0 to disable.')
    p.add_argument('--adv_weight',        type=float, default=0.25)
    p.add_argument('--adv_mode',          type=str,   default='auto',
                   choices=['fgsm', 'pgd', 'auto'],
                   help='auto=PGD every 3rd epoch, FGSM otherwise')

    p.add_argument('--input_size',        type=int,   default=224)
    p.add_argument('--workers',           type=int,   default=2)
    p.add_argument('--seed',              type=int,   default=42)

    # FIX-4: explicit val batch size; 0 = same as train batch_size
    p.add_argument('--val_batch_size',    type=int,   default=0,
                   help='Val batch size. Default 0 = same as --batch_size. '
                        'Only increase if VRAM permits.')

    args = p.parse_args()
    train(args)