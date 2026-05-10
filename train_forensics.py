"""
train_forensics.py — Training pipeline for SpatialRobustModel (ForensicsAI / HybridRobust)
Hardware target: NVIDIA RTX 2050 4GB VRAM + CUDA

Dataset layout:
  data_dir/
    train/
      real/   (or REAL/)
      fake/   (or FAKE/)
    val/
      real/
      fake/

Labels: FAKE=0, REAL=1

Usage:
  python train_forensics.py --data_dir /path/to/dataset --ckpt_dir ./checkpoints
"""

import os
import sys
import io
import json
import math
import time
import random
import argparse
import numpy as np
from tqdm import tqdm
from pathlib import Path
from collections import defaultdict
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from ForensicsAI import (
    SpatialRobustModel,
    get_spatial_robust_model,
    AsymmetricFocalLoss,
    DeepPostprocessingAug,
    mixup_data,
    cutmix_data,
    mixup_criterion,
    save_checkpoint,
    load_checkpoint,
    predict_with_uncertainty,
    AdversarialTester,
    CLIP_MEAN,
    CLIP_STD,
)


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = True


# ── Dataset ────────────────────────────────────────────────────────────────────

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}


def _find_class_dirs(root: Path) -> Dict[int, Path]:
    dirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    fake_dir = dirs.get('fake') or dirs.get('0')
    real_dir = dirs.get('real') or dirs.get('1')
    if fake_dir is None or real_dir is None:
        raise FileNotFoundError(
            f"Expected fake/real subdirs in {root}. Found: {list(dirs.keys())}"
        )
    return {0: fake_dir, 1: real_dir}


class ForensicDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list = []
        self.skipped = 0
        root_path = Path(root)
        class_dirs = _find_class_dirs(root_path)

        for label, class_dir in class_dirs.items():
            for ext in VALID_EXTS:
                for fpath in class_dir.rglob(f'*{ext}'):
                    self.samples.append((str(fpath), label))
                for fpath in class_dir.rglob(f'*{ext.upper()}'):
                    self.samples.append((str(fpath), label))

        seen, unique = set(), []
        for s in self.samples:
            if s[0] not in seen:
                seen.add(s[0])
                unique.append(s)
        self.samples = unique
        random.shuffle(self.samples)

        self.class_counts: Dict[int, int] = defaultdict(int)
        for _, label in self.samples:
            self.class_counts[label] += 1

        print(
            f"  Dataset [{root_path.name}]: "
            f"FAKE={self.class_counts[0]}  REAL={self.class_counts[1]}  "
            f"Total={len(self.samples)}"
        )

    def get_class_weights(self) -> torch.Tensor:
        n      = len(self.samples)
        n_fake = max(self.class_counts[0], 1)
        n_real = max(self.class_counts[1], 1)
        # Cap raised for REAL: 3.0 → 5.0 so heavily-imbalanced datasets get
        # meaningful REAL correction without capping at 3× fake weight.
        # FAKE cap kept at 2.0 to prevent FAKE over-domination.
        w_fake = min(n / (2.0 * n_fake), 2.0)
        w_real = min(n / (2.0 * n_real), 5.0)
        print(f"  Class weights — FAKE: {w_fake:.3f}  REAL: {w_real:.3f}")
        return torch.tensor([w_fake, w_real], dtype=torch.float32)

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError, Exception):
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
# Lightweight PIL-space augmentations for the DataLoader.
# Heavy post-processing simulation (blur, sharpen, diffusion artefacts, ISP, etc.)
# is handled by DeepPostprocessingAug on-GPU inside the training step.
# Keeping PIL augmentations minimal avoids double-augmentation conflicts.

class _JPEGCompress:
    def __init__(self, lo=60, hi=95, p=0.20):  # p: 0.4→0.20, lo: 50→60 (less aggressive)
        self.lo, self.hi, self.p = lo, hi, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=random.randint(self.lo, self.hi))
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class _DownUpscale:
    def __init__(self, lo=0.6, hi=0.9, p=0.15):
        self.lo, self.hi, self.p = lo, hi, p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        W, H  = img.size
        scale = random.uniform(self.lo, self.hi)
        small = img.resize((max(1, int(W * scale)), max(1, int(H * scale))), Image.BILINEAR)
        return small.resize((W, H), Image.BILINEAR)


class _GaussianNoise:
    def __init__(self, lo=0.003, hi=0.015, p=0.25):
        self.lo, self.hi, self.p = lo, hi, p

    def __call__(self, t: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return t
        return t + torch.randn_like(t) * random.uniform(self.lo, self.hi)


def get_transforms(input_size: int = 224, is_train: bool = True):
    mean, std = CLIP_MEAN, CLIP_STD
    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.08),
            transforms.ColorJitter(brightness=0.10, contrast=0.10,
                                   saturation=0.06, hue=0.02),
            transforms.RandomGrayscale(p=0.03),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.08),
            _DownUpscale(p=0.10),
            _JPEGCompress(p=0.20),
            transforms.ToTensor(),
            _GaussianNoise(p=0.15),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.05, scale=(0.02, 0.06)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── LR schedule ───────────────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs: int,
                                  total_epochs: int, min_lr_ratio: float = 0.05):
    def _lr(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, _lr)


# ── Adversarial training (FGSM) ────────────────────────────────────────────────

@torch.enable_grad()
def fgsm_perturb(model: torch.nn.Module, imgs: torch.Tensor,
                 labels: torch.Tensor, eps: float) -> torch.Tensor:
    """Single-step FGSM perturbation for adversarial training.
    Returns perturbed images detached from graph (no second-order grad needed).
    """
    imgs_adv = imgs.clone().detach().requires_grad_(True)
    with torch.amp.autocast('cuda', enabled=imgs.device.type == 'cuda'):
        logits = model(imgs_adv)
        loss   = torch.nn.functional.cross_entropy(logits, labels)
    loss.backward()
    with torch.no_grad():
        adv = (imgs + eps * imgs_adv.grad.sign()).clamp(-3.0, 3.0)
    return adv.detach()


# ── Metrics helpers ────────────────────────────────────────────────────────────

def balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    per_class = []
    for c in range(2):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_class.append((preds[mask] == labels[mask]).mean())
    return float(np.mean(per_class)) if per_class else 0.0


# ── Build optimiser ────────────────────────────────────────────────────────────
# Forensic streams learn fast; ViT backbone stays near-frozen until unfreeze_epoch.
# stream_fusion covers AR-1..4 (evidential heads, parity gate, ISFCR, FCW).

def build_optimizer(model: SpatialRobustModel, lr: float,
                    weight_decay: float) -> torch.optim.AdamW:
    backbone_params = [p for p in model.extractor.parameters() if p.requires_grad]

    forensic_params = []
    for attr in ('freq_stream', 'dct_block', 'fft_stream', 'noise_block', 'jpeg_block'):
        m = getattr(model, attr, None)
        if m is not None:
            forensic_params += list(m.parameters())

    fusion_params = list(model.stream_fusion.parameters())

    head_params = (
        list(model.early_pool.parameters()) +
        list(model.mid_pool.parameters()) +
        list(model.final_pool.parameters()) +
        list(model.cross_attn.parameters()) +
        list(model.mlp.parameters())
    )
    classifier_params = list(model.classifier.parameters())

    all_assigned = set(
        id(p)
        for group in [backbone_params, forensic_params, fusion_params,
                      head_params, classifier_params]
        for p in group
    )
    other_params = [p for p in model.parameters()
                    if p.requires_grad and id(p) not in all_assigned]

    param_groups = [
        {'params': backbone_params,  'lr': lr * 0.05,  'name': 'backbone'},
        {'params': forensic_params,  'lr': lr * 2.0,   'name': 'forensic_streams'},
        {'params': fusion_params,    'lr': lr * 1.5,   'name': 'stream_fusion'},
        {'params': head_params,      'lr': lr,          'name': 'head'},
        {'params': classifier_params,'lr': lr * 0.5,   'name': 'classifier'},
        {'params': other_params,     'lr': lr,          'name': 'other'},
    ]
    param_groups = [g for g in param_groups if len(g['params']) > 0]
    return torch.optim.AdamW(param_groups, weight_decay=weight_decay)


# ── Training step ──────────────────────────────────────────────────────────────

def train_epoch(
    model:         SpatialRobustModel,
    aug:           DeepPostprocessingAug,
    loader:        DataLoader,
    optimizer:     torch.optim.AdamW,
    criterion:     AsymmetricFocalLoss,
    device:        torch.device,
    mixup_alpha:   float = 0.2,
    aux_weight:    float = 1.0,
    entropy_weight:float = 0.03,
    scaler=None,
    amp_enabled:   bool = False,
    adv_eps:       float = 0.0,   # FGSM eps for adversarial training; 0 = disabled
    adv_weight:    float = 0.3,   # weight of adversarial loss vs clean loss
) -> Dict:
    model.train()
    aug.train()

    total_loss = 0.0
    all_preds:  list = []
    all_labels: list = []
    n_batches   = 0


    pbar = tqdm(loader, desc="Training", leave=False)

    for imgs, labels, _ in pbar:
        nb     = imgs.is_pinned()
        imgs   = imgs.to(device, non_blocking=nb)
        labels = labels.to(device, non_blocking=nb)

        # On-GPU deep augmentation (10 forensic-aware aug types)
        with torch.no_grad():
            imgs = aug(imgs)

        # MixUp / CutMix (50/50)
        # Guard: skip mixing for pure-REAL batches — mixing REAL with FAKE at low
        # alpha still creates ambiguous supervision that degrades REAL separability.
        # Also skip CutMix with alpha=0.5 for REAL-heavy batches (>60% REAL) since
        # large patches from FAKEs smear forensic-clean regions.
        _real_frac = (labels == 1).float().mean().item()
        if mixup_alpha > 0 and _real_frac < 0.9:
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            else:
                # Use smaller alpha for CutMix when batch is REAL-heavy
                _cutmix_alpha = 0.3 if _real_frac > 0.6 else 0.5
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=_cutmix_alpha)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)

        if amp_enabled and scaler is not None:
            with torch.amp.autocast('cuda'):
                logits, aux_loss, components = model.forward_with_entropy(imgs, labels)

                if mixup_alpha > 0 and lam < 1.0:
                    cls_loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    cls_loss = criterion(logits, labels)

                e_ent = components.get('e_ent', 0.0)
                m_ent = components.get('m_ent', 0.0)
                f_ent = components.get('f_ent', 0.0)
                ent_reg = -entropy_weight * (e_ent + m_ent + f_ent)

                # Adversarial training: blend clean + FGSM adversarial loss
                if adv_eps > 0.0:
                    imgs_adv = fgsm_perturb(model, imgs, labels, eps=adv_eps)
                    with torch.amp.autocast('cuda'):
                        adv_logits, _, _ = model.forward_with_entropy(imgs_adv, labels)
                        adv_loss = criterion(adv_logits, labels)
                    loss = ((1.0 - adv_weight) * cls_loss
                            + adv_weight * adv_loss
                            + aux_weight * aux_loss + ent_reg)
                else:
                    loss = cls_loss + aux_weight * aux_loss + ent_reg

                pbar.set_postfix({
                "loss": f"{loss.item():.4f}"
            })

            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, aux_loss, components = model.forward_with_entropy(imgs, labels)

            if mixup_alpha > 0 and lam < 1.0:
                cls_loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                cls_loss = criterion(logits, labels)

            e_ent = components.get('e_ent', 0.0)
            m_ent = components.get('m_ent', 0.0)
            f_ent = components.get('f_ent', 0.0)
            ent_reg = -entropy_weight * (e_ent + m_ent + f_ent)

            if adv_eps > 0.0:
                imgs_adv = fgsm_perturb(model, imgs, labels, eps=adv_eps)
                adv_logits, _, _ = model.forward_with_entropy(imgs_adv, labels)
                adv_loss = criterion(adv_logits, labels)
                loss = ((1.0 - adv_weight) * cls_loss
                        + adv_weight * adv_loss
                        + aux_weight * aux_loss + ent_reg)
            else:
                loss = cls_loss + aux_weight * aux_loss + ent_reg

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        model.training_step_hook()

        total_loss += loss.item()
        with torch.no_grad():
            preds = logits.argmax(1)
            probs = torch.softmax(logits, dim=1)
            confs = probs.max(1).values
            # Record every training sample — correct and incorrect — so error_memory
            # learns the per-class confidence distribution during training, not only val.
            _stream_info = components if isinstance(components, dict) else {}
            for _j in range(len(labels)):
                model.step_error_memory(
                    labels[_j:_j+1].cpu(),
                    preds[_j:_j+1].cpu(),
                    confs[_j:_j+1].cpu(),
                    _stream_info,
                )
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        n_batches += 1

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc     = (all_preds == all_labels).mean() * 100
    bal_acc = balanced_accuracy(all_preds, all_labels) * 100

    return {
        'loss':    total_loss / max(n_batches, 1),
        'acc':     acc,
        'bal_acc': bal_acc,
    }


# ── Validation step ────────────────────────────────────────────────────────────

def validate(
    model:     SpatialRobustModel,
    loader:    DataLoader,
    criterion: AsymmetricFocalLoss,
    device:    torch.device,
) -> Dict:
    model.eval()
    total_loss  = 0.0
    all_preds:  list = []
    all_labels: list = []
    all_confs:  list = []
    n_batches   = 0

    with torch.no_grad():
        pbar = tqdm(loader, desc="Validation", leave=False)
        for imgs, labels, paths in pbar:    
            nb     = imgs.is_pinned()
            imgs   = imgs.to(device, non_blocking=nb)
            labels = labels.to(device, non_blocking=nb)

            logits, stream_info = model.forward_with_streams(imgs)
            loss  = criterion(logits, labels)
            pbar.set_postfix({
            "val_loss": f"{loss.item():.4f}"
            })
            total_loss += loss.item()

            probs = torch.softmax(logits, dim=1)
            preds = logits.argmax(1)
            confs = probs.max(1).values

            # Error + confidence tracking per sample
            model.step_error_memory(
                labels.cpu(), preds.cpu(), confs.cpu(),
                stream_info,
            )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            n_batches += 1

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc     = (all_preds == all_labels).mean() * 100
    bal_acc = balanced_accuracy(all_preds, all_labels) * 100

    fake_mask   = all_labels == 0
    real_mask   = all_labels == 1
    fake_recall = (all_preds[fake_mask] == 0).mean() * 100 if fake_mask.sum() > 0 else 0.0
    real_recall = (all_preds[real_mask] == 1).mean() * 100 if real_mask.sum() > 0 else 0.0

    if fake_recall < 70.0:
        print(f"  ⚠️  FAKE recall LOW: {fake_recall:.1f}% — model may over-predict REAL")

    shift = model.confidence_tracker.check_shift()
    if shift.get('shift_detected'):
        print(f"  ⚠️  Distribution shift: {shift.get('alert','')}")

    return {
        'loss':        total_loss / max(n_batches, 1),
        'acc':         acc,
        'bal_acc':     bal_acc,
        'fake_recall': fake_recall,
        'real_recall': real_recall,
    }


# ── Resolve val directory ──────────────────────────────────────────────────────

def _resolve_val_dir(data_dir: str, preferred: str = 'val') -> str:
    for name in (preferred, 'test', 'val'):
        candidate = os.path.join(data_dir, name)
        if os.path.isdir(candidate):
            if name != preferred:
                print(f"  [INFO] Val dir '{preferred}' not found — using '{name}'")
            return candidate
    raise FileNotFoundError(
        f"No validation directory found in {data_dir}. Tried: {preferred}, test, val"
    )


# ── Main training loop ─────────────────────────────────────────────────────────

def train(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        print(f"GPU: {props.name}  VRAM: {props.total_memory / 1e9:.1f} GB")

    forensics_ckpt_dir = os.path.join(os.path.dirname(args.ckpt_dir), 'forensics_ckpt')
    os.makedirs(forensics_ckpt_dir, exist_ok=True)
    ckpt_path  = os.path.join(forensics_ckpt_dir, 'forensics_latest.pt')
    best_path  = os.path.join(forensics_ckpt_dir, 'forensics_best.pt')
    final_path = os.path.join(forensics_ckpt_dir, 'forensics_final.pt')
    log_path   = os.path.join(forensics_ckpt_dir, 'train_log.jsonl')
    err_path   = os.path.join(forensics_ckpt_dir, 'error_memory.json')

    train_tf = get_transforms(args.input_size, is_train=True)
    val_tf   = get_transforms(args.input_size, is_train=False)

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir   = _resolve_val_dir(args.data_dir, preferred=args.val_split)

    train_ds = ForensicDataset(train_dir, train_tf)
    val_ds   = ForensicDataset(val_dir,   val_tf)

    _use_persist = args.workers > 0
    _lkw = dict(
        num_workers        = args.workers,
        pin_memory         = args.workers > 0,
        collate_fn         = collate_fn,
        persistent_workers = _use_persist,
    )
    if args.workers > 0:
        _lkw['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, drop_last=True, **_lkw,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2,
        shuffle=False, **_lkw,
    )

    # Model
    model = get_spatial_robust_model(
        freeze_backbone    = True,
        unfreeze_last_n    = 0,
        dropout            = args.dropout,
        drop_path_rate     = args.drop_path,
        input_size         = args.input_size,
        use_grad_checkpoint= True,
        use_isfcr          = True,
        use_fcw            = True,
        n_iters            = args.n_iters,
        evidential_coeff   = args.evidential_coeff,
    ).to(device)

    # Deep post-processing augmentation (on-GPU, forensic-aware)
    # chain_prob=0.3: 30% chance of chaining multiple aug types per image
    aug = DeepPostprocessingAug(
        freq_aug_prob=0.15,   # 0.20 → 0.15
        jpeg_sim_prob=0.30,   # 0.35 → 0.30
        blur_prob=0.20,       # 0.25 → 0.20
        sharpen_prob=0.12,    # 0.15 → 0.12
        resize_prob=0.15,     # 0.20 → 0.15
        noise_prob=0.20,      # 0.25 → 0.20
        chroma_prob=0.10,     # 0.15 → 0.10
        diffusion_prob=0.08,  # 0.15 → 0.08  ← diffusion perturb destroys REAL features
        isp_prob=0.08,        # 0.15 → 0.08  ← ISP sim causes REAL→FAKE confusion
        thumbnail_prob=0.08,  # 0.12 → 0.08
        chain_prob=0.10,      # 0.15 → 0.10  ← chaining multiplies individual probs
    ).to(device)

    base_weights = train_ds.get_class_weights()
    criterion = AsymmetricFocalLoss(
        gamma_fake      = args.focal_gamma_fake,
        gamma_real      = args.focal_gamma_real,
        weight          = base_weights.to(device),
        label_smoothing = args.label_smoothing,
    )

    optimizer = build_optimizer(model, lr=args.lr, weight_decay=args.weight_decay)
    scheduler = cosine_schedule_with_warmup(
        optimizer, warmup_epochs=args.warmup_epochs, total_epochs=args.epochs)

    amp_enabled = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=amp_enabled) if amp_enabled else None

    start_epoch  = 0
    best_val_acc = 0.0

    ckpt = load_checkpoint(ckpt_path, map_location=device)
    if ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except (ValueError, KeyError):
            print("  ⚠️  Optimizer state mismatch — skipping optimizer load")
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except (ValueError, KeyError):
            print("  ⚠️  Scheduler state mismatch — skipping scheduler load")
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"  Resumed from epoch {ckpt['epoch']}  best_val_acc={best_val_acc:.2f}%")
        model.error_memory.load(err_path)
    else:
        print("  No checkpoint — starting from scratch")

    def _maybe_unfreeze(epoch: int):
        # Stage 1: unfreeze last N blocks at args.unfreeze_epoch
        if epoch == args.unfreeze_epoch and model.unfrozen_count == 0:
            model._unfreeze_last_n(args.unfreeze_last_n)
            model.classifier.unfreeze_scale()
            new_bb = [p for p in model.extractor.parameters() if p.requires_grad]
            if new_bb:
                # Use 0.01× base LR (was 0.02×) — finer-grained fine-tuning to
                # prevent representation drift immediately after unfreeze.
                optimizer.add_param_group(
                    {'params': new_bb, 'lr': args.lr * 0.01, 'name': 'backbone_unfrozen'})
            print(f"  Epoch {epoch}: backbone last {args.unfreeze_last_n} blocks unfrozen "
                  f"(lr={args.lr * 0.01:.2e})")

        # Stage 2: after 10 more epochs of fine-tuning, raise backbone lr slightly
        stage2_epoch = args.unfreeze_epoch + 10
        if epoch == stage2_epoch and model.unfrozen_count > 0:
            for pg in optimizer.param_groups:
                if pg.get('name') == 'backbone_unfrozen':
                    pg['lr'] = args.lr * 0.03   # lift from 0.01× to 0.03×
                    print(f"  Epoch {epoch}: backbone lr raised to {pg['lr']:.2e} (stage 2)")

    no_improve = 0
    patience   = args.patience

    # Track REAL recall EMA for adaptive bias
    _real_recall_ema = 0.0
    _real_recall_alpha = 0.3   # fast EMA so we react within 3-4 epochs

    print(f"\nTraining for {args.epochs} epochs  (start={start_epoch})")
    print(
        f"  batch={args.batch_size}  lr={args.lr}  "
        f"mixup={args.mixup_alpha}  gamma_fake={args.focal_gamma_fake}  "
        f"gamma_real={args.focal_gamma_real}  entropy_w={args.entropy_weight}  "
        f"aux_w={args.aux_weight}  warmup={args.warmup_epochs}  "
        f"adv_eps={args.adv_eps:.4f}  adv_weight={args.adv_weight}","\n\n"
    )

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        _maybe_unfreeze(epoch)

        # ── Adaptive class weights from error memory (update every epoch) ──────
        # Only apply adaptive reweighting after warmup so early noisy EMA doesn't
        # destabilise the loss before the model has found a reasonable basin.
        if epoch >= args.warmup_epochs:
            adapted_w = model.error_memory.get_loss_weights(
                [base_weights[0].item(), base_weights[1].item()])
            criterion.update_weight(
                torch.tensor(adapted_w, dtype=torch.float32).to(device))
        else:
            # During warmup: use balanced base weights, no EMA-driven reweighting
            criterion.update_weight(base_weights.to(device))

        # ── REAL-recall adaptive focal bias ────────────────────────────────────
        # If REAL recall EMA drops below 80%, temporarily boost real_floor_weight
        # inside the criterion to ensure REAL samples get non-zero gradient signal.
        if epoch > 0 and hasattr(criterion, 'real_floor_weight'):
            if _real_recall_ema < 80.0:
                criterion.real_floor_weight = 0.10  # boost floor
            elif _real_recall_ema > 90.0:
                criterion.real_floor_weight = 0.03  # relax floor once stable
            # else: keep current value (0.05 default)

        # ── MixUp/CutMix ramp-in ──────────────────────────────────────────────
        # Use pure clean samples during warmup + 2 extra epochs to let the model
        # build stable class representations before introducing mixed targets.
        # After that, ramp alpha from 0 → args.mixup_alpha over 5 epochs.
        _mixup_start_epoch = args.warmup_epochs + 2
        if epoch < _mixup_start_epoch:
            mixup_alpha_cur = 0.0
        else:
            ramp = min(1.0, (epoch - _mixup_start_epoch) / 5.0)
            mixup_alpha_cur = args.mixup_alpha * ramp

        # ── Adversarial eps: ramp from 0 after warmup ─────────────────────────
        adv_ramp_epochs = max(args.warmup_epochs + 5, 12)
        adv_eps_cur = 0.0
        if args.adv_eps > 0.0 and epoch >= args.warmup_epochs:
            ramp_progress = min(1.0, (epoch - args.warmup_epochs) / max(adv_ramp_epochs, 1))
            adv_eps_cur = args.adv_eps * ramp_progress

        train_stats = train_epoch(
            model          = model,
            aug            = aug,
            loader         = train_loader,
            optimizer      = optimizer,
            criterion      = criterion,
            device         = device,
            mixup_alpha    = mixup_alpha_cur,
            aux_weight     = args.aux_weight,
            entropy_weight = args.entropy_weight,
            scaler         = scaler,
            amp_enabled    = amp_enabled,
            adv_eps        = adv_eps_cur,
            adv_weight     = args.adv_weight,
        )
        val_stats = validate(model, val_loader, criterion, device)
        scheduler.step()

        # Update REAL recall EMA for next-epoch adaptive bias
        _real_recall_ema = (
            (1.0 - _real_recall_alpha) * _real_recall_ema
            + _real_recall_alpha * val_stats['real_recall']
        )

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print("\n",
            f"Epoch {epoch:03d}/{args.epochs - 1} | "
            f"T_loss={train_stats['loss']:.4f}  T_acc={train_stats['acc']:.1f}%  "
            f"T_bal={train_stats['bal_acc']:.1f}% | "
            f"V_loss={val_stats['loss']:.4f}  V_acc={val_stats['acc']:.1f}%  "
            f"V_bal={val_stats['bal_acc']:.1f}%  "
            f"FAKE_R={val_stats['fake_recall']:.1f}%  REAL_R={val_stats['real_recall']:.1f}%  "
            f"REAL_EMA={_real_recall_ema:.1f}%  mixup={mixup_alpha_cur:.2f} | "
            f"LR={lr_now:.2e}  {elapsed:.0f}s"
        )

        # Stream routing diagnostics every 5 epochs
        if epoch % 5 == 0:
            print(model.get_routing_report())
            low = model.get_low_importance_streams(threshold=0.03)
            if low:
                print(f"  Low-contribution streams (EMA < 0.03): {low}")

        err_summ = model.error_memory.summary()
        print(
            f"  ErrorMem: real_as_fake={err_summ['real_as_fake']}  "
            f"fake_as_real={err_summ['fake_as_real']}  "
            f"ema_fake_err={err_summ['ema_fake_err_rate']:.3f}  "
            f"ema_real_err={err_summ['ema_real_err_rate']:.3f}","\n\n\n"
        )
        

        log_entry = {
            'epoch': epoch,
            **train_stats,
            **{f'val_{k}': v for k, v in val_stats.items()},
            'lr': lr_now,
            **{f'err_{k}': v for k, v in err_summ.items()
               if isinstance(v, (int, float))},
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        ckpt_state = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc':         best_val_acc,
            'val_stats':            val_stats,
        }
        save_checkpoint(ckpt_state, ckpt_path)

        monitor = val_stats['bal_acc']
        if monitor > best_val_acc:
            best_val_acc               = monitor
            ckpt_state['best_val_acc'] = best_val_acc
            save_checkpoint(ckpt_state, best_path)
            print(f"  ✅ New best bal_acc: {best_val_acc:.2f}%  → {best_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {patience} epochs without improvement.")
                break

        model.error_memory.save(err_path)

    # ── Post-training: calibration, adversarial check, stream pruning ──────────

    print("\n=== Temperature Calibration ===")
    best_ckpt = load_checkpoint(best_path, map_location=device)
    if best_ckpt:
        model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
    model.calibrate(val_loader, device)

    print("\n=== Stream Pruning (post-training) ===")
    low_streams = model.get_low_importance_streams(threshold=args.prune_threshold)
    if low_streams:
        print(f"  Pruning streams with EMA < {args.prune_threshold}: {low_streams}")
        for s in low_streams:
            model.prune_stream(s)
    else:
        print(f"  No streams below threshold {args.prune_threshold} — no pruning needed")

    print("\n=== Adversarial Robustness ===")
    adv_tester = AdversarialTester(model, device)
    try:
        adv_result = adv_tester.evaluate(val_loader, attack='fgsm', eps=8/255, max_batches=10)
        print(f"  FGSM bal_drop: {adv_result['bal_drop']:.2f}%  {adv_result['verdict']}")
    except AttributeError:
        # AdversarialTester.evaluate may not exist; use manual FGSM pass
        adv_result = {}
        model.eval()
        correct, total = 0, 0
        for i, (imgs, labels, _) in enumerate(val_loader):
            if i >= 10:
                break
            adv = adv_tester.fgsm(imgs.to(device), labels.to(device), eps=8/255)
            with torch.no_grad():
                preds = model(adv).argmax(1)
            correct += (preds == labels.to(device)).sum().item()
            total   += labels.size(0)
        adv_acc = 100.0 * correct / max(total, 1)
        print(f"  FGSM accuracy: {adv_acc:.2f}%")
        adv_result = {'fgsm_acc': adv_acc}

    save_checkpoint({
        'model_state_dict':  model.state_dict(),
        'epoch':             'final',
        'best_val_acc':      best_val_acc,
        'adv_result':        adv_result,
        'error_summary':     model.error_memory.summary(),
        'routing_report':    model.get_routing_report(),
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print(f"Error memory summary: {model.error_memory.summary()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train ForensicsAI / SpatialRobustModel')

    parser.add_argument('--data_dir',          required=True,
                        help='Root dataset dir containing train/ and val/ subdirs')
    parser.add_argument('--ckpt_dir',          default='./checkpoints')
    parser.add_argument('--val_split',         default='val',
                        help='Val subdirectory name (fallback: test, val)')

    parser.add_argument('--epochs',            type=int,   default=30)
    parser.add_argument('--batch_size',        type=int,   default=8,
                        help='8 is safe for RTX 2050 4GB with AMP enabled')
    parser.add_argument('--lr',                type=float, default=3e-4)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--dropout',           type=float, default=0.25)
    parser.add_argument('--drop_path',         type=float, default=0.08)

    parser.add_argument('--focal_gamma_fake',  type=float, default=2.0,
                        help='Focal gamma for FAKE — reduced from 3.0; prevents FAKE over-dominance')
    parser.add_argument('--focal_gamma_real',  type=float, default=2.5,
                        help='Focal gamma for REAL — raised from 2.0; harder push on missed REALs')
    parser.add_argument('--label_smoothing',   type=float, default=0.03)

    parser.add_argument('--mixup_alpha',       type=float, default=0.1,
                        help='MixUp/CutMix alpha; 0 to disable. Reduced from 0.2 — ramped in '
                             'gradually after warmup+2 epochs to avoid REAL feature ambiguity')
    parser.add_argument('--entropy_weight',    type=float, default=0.03,
                        help='Attention entropy regularisation weight')
    parser.add_argument('--aux_weight',        type=float, default=1.0,
                        help='Scale for AR-5 AuthenticityConsistencyLoss')

    parser.add_argument('--warmup_epochs',     type=int,   default=3)
    parser.add_argument('--unfreeze_epoch',    type=int,   default=10,
                        help='Epoch at which ViT backbone blocks are unfrozen (was 5; '
                             'now 10 — lets stream fusion stabilise before backbone drifts)')
    parser.add_argument('--unfreeze_last_n',   type=int,   default=3,
                        help='Number of ViT transformer blocks to unfreeze')
    parser.add_argument('--patience',          type=int,   default=8,
                        help='Early stopping patience (epochs without bal_acc improvement)')

    parser.add_argument('--n_iters',           type=int,   default=2,
                        help='AR-3 ISFCR reasoning iterations (1=fast, 2=default, 3=thorough)')
    parser.add_argument('--evidential_coeff',  type=float, default=0.01,
                        help='AR-1 NIG evidential loss coefficient')
    parser.add_argument('--prune_threshold',   type=float, default=0.03,
                        help='Post-training stream pruning EMA threshold')

    parser.add_argument('--input_size',        type=int,   default=224)
    parser.add_argument('--workers',           type=int,   default=2)
    parser.add_argument('--seed',              type=int,   default=42)

    parser.add_argument('--adv_eps',           type=float, default=4.0/255,
                        help='FGSM adversarial training epsilon (default 4/255). '
                             'Set 0 to disable. Ramped from 0 over first 10 post-warmup epochs.')
    parser.add_argument('--adv_weight',        type=float, default=0.25,
                        help='Weight of adversarial loss vs clean loss (default 0.25).')

    args = parser.parse_args()
    train(args)