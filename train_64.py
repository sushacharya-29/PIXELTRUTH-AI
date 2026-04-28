"""
train_v64.py — Training script for SpatialModelV6.4
RTX 2050 4GB VRAM optimised.

Usage:
  python train_v64.py --data_dir /path/to/dataset --ckpt_dir ./checkpoints

Dataset layout expected:
  data_dir/
    train/
      real/   (or REAL/)
      fake/   (or FAKE/)
    val/
      real/
      fake/

Labels: FAKE=0, REAL=1

Key changes vs train_v63.py:
  ✅ AUG-1  JPEG compression simulation in training transforms (q=40-95):
             matches real-world online image distribution (DALL-E, Pixabay output)
  ✅ AUG-2  Noise augmentation: Gaussian + slight downscale-upscale re-encoding sim
  ✅ AUG-3  CutMix added alongside MixUp (50/50 selection per batch)
  ✅ AUG-4  MixUp alpha reduced 0.3→0.2 in defaults
  ✅ OPT-1  Separate LR group for frequency stream (2× head LR)
  ✅ OPT-2  Separate LR group for DCT block (2× head LR)
  ✅ CFG-1  Default unfreeze_last_n=3 (was 2)
  ✅ CFG-2  focal_gamma_fake and focal_gamma_real CLI args (was single focal_gamma)
  ✅ CFG-3  entropy_weight default 0.01→0.03
  ✅ VAL-1  Fake recall threshold warning: prints alert if fake_recall < 70%
  ✅ All v6.3 train fixes preserved
"""

import os
import sys
import io
import argparse
import math
import random
import time
import json
from pathlib import Path
from typing import Dict, Optional

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, UnidentifiedImageError, ImageFilter
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v64spatial import (
    SpatialModelV64, get_spatial_model_v64,
    AsymmetricFocalLoss, FocalLoss,
    mixup_data, cutmix_data, mixup_criterion,
    save_checkpoint, load_checkpoint,
    predict_with_uncertainty, AdversarialTester,
    CLIP_MEAN, CLIP_STD,
)

# Alias for drop-in compatibility if someone imports from train_v63
SpatialModelV63    = SpatialModelV64
get_spatial_model_v63 = get_spatial_model_v64


# ── Reproducibility ────────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── JPEG Compression Simulation Transform (NEW v6.4) ──────────────────────────
#
# DALL-E, Pixabay, and most online sources deliver JPEG-compressed images.
# Without JPEG augmentation the model learns features that are disrupted by
# JPEG artefacts at inference time → higher fake_as_real errors.

class RandomJPEGCompression:
    """
    Simulate JPEG compression by encoding the PIL image to JPEG and decoding.
    quality: random integer in [lo, hi]
    """
    def __init__(self, quality_lo: int = 40, quality_hi: int = 95, p: float = 0.5):
        self.lo = quality_lo
        self.hi = quality_hi
        self.p  = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        quality = random.randint(self.lo, self.hi)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=quality)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class RandomDownscaleUpscale:
    """
    Simulate social-media re-encoding: downscale to [scale_lo, scale_hi] fraction
    then upscale back to original size. Introduces compression-style blurring.
    """
    def __init__(self, scale_lo: float = 0.5, scale_hi: float = 0.85, p: float = 0.2):
        self.lo = scale_lo
        self.hi = scale_hi
        self.p  = p

    def __call__(self, img: Image.Image) -> Image.Image:
        if random.random() > self.p:
            return img
        W, H   = img.size
        scale  = random.uniform(self.lo, self.hi)
        small  = img.resize((max(1, int(W * scale)), max(1, int(H * scale))),
                             Image.BILINEAR)
        return small.resize((W, H), Image.BILINEAR)


class RandomGaussianNoise:
    """Add pixel-level Gaussian noise (applied after ToTensor)."""
    def __init__(self, std_lo: float = 0.005, std_hi: float = 0.02, p: float = 0.3):
        self.lo = std_lo
        self.hi = std_hi
        self.p  = p

    def __call__(self, tensor: torch.Tensor) -> torch.Tensor:
        if random.random() > self.p:
            return tensor
        std   = random.uniform(self.lo, self.hi)
        noise = torch.randn_like(tensor) * std
        return tensor + noise


# ── Dataset ────────────────────────────────────────────────────────────────────

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.avif','.tif', '.webp', '.gif'}
CLASS_NAMES = ['fake', 'real']  # FAKE=0, REAL=1


def _find_class_dirs(root: Path):
    dirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    fake_dir = dirs.get('fake') or dirs.get('0')
    real_dir = dirs.get('real') or dirs.get('1')
    if fake_dir is None or real_dir is None:
        raise FileNotFoundError(
            f"Could not find fake/real subdirs in {root}. Found: {list(dirs.keys())}"
        )
    return {0: fake_dir, 1: real_dir}


class ForensicDataset(Dataset):
    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list = []
        self.skipped = 0

        root_path  = Path(root)
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

        self.class_counts = defaultdict(int)
        for _, label in self.samples:
            self.class_counts[label] += 1

        print(f"  Dataset [{root_path.name}]: "
              f"FAKE={self.class_counts[0]}  REAL={self.class_counts[1]}  "
              f"Total={len(self.samples)}")

    def get_class_weights(self) -> torch.Tensor:
        n      = len(self.samples)
        n_fake = max(self.class_counts[0], 1)
        n_real = max(self.class_counts[1], 1)
        w_fake = n / (2.0 * n_fake)
        w_real = n / (2.0 * n_real)
        w_fake = min(w_fake, 3.0)
        w_real = min(w_real, 3.0)
        print(f"  Class weights — FAKE: {w_fake:.3f}  REAL: {w_real:.3f}")
        return torch.tensor([w_fake, w_real], dtype=torch.float32)

    def __len__(self):
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
        if self.transform:
            img = self.transform(img)
        return img, label, path


def collate_fn(batch):
    imgs   = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths  = [b[2] for b in batch]
    return imgs, labels, paths


# ── Transforms (v6.4: JPEG compression + noise simulation added) ───────────────

def get_transforms(input_size: int = 224, is_train: bool = True):
    mean = CLIP_MEAN
    std  = CLIP_STD

    if is_train:
        return transforms.Compose([
            # Spatial augmentation
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.1, hue=0.03),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            # v6.4: simulate online image re-encoding pipeline
            RandomDownscaleUpscale(scale_lo=0.5, scale_hi=0.85, p=0.2),
            RandomJPEGCompression(quality_lo=40, quality_hi=95, p=0.5),
            # Convert to tensor
            transforms.ToTensor(),
            # v6.4: pixel noise AFTER tensor conversion
            RandomGaussianNoise(std_lo=0.005, std_hi=0.02, p=0.3),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Cosine LR with warmup ──────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int,
                                  min_lr_ratio: float = 0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Balanced accuracy ──────────────────────────────────────────────────────────

def balanced_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int = 2) -> float:
    per_class = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_class.append((preds[mask] == labels[mask]).mean())
    return float(np.mean(per_class)) if per_class else 0.0


# ── Train one epoch (v6.4: CutMix added, 50/50 with MixUp) ───────────────────

def train_epoch(model: SpatialModelV64, loader: DataLoader, optimizer,
                criterion: AsymmetricFocalLoss, device: torch.device,
                use_mixup: bool = True, mixup_alpha: float = 0.2,
                entropy_weight: float = 0.03,
                scaler=None, amp_enabled: bool = False) -> Dict:
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = 0

    for imgs, labels, _ in loader:
        _nb    = imgs.is_pinned()
        imgs   = imgs.to(device, non_blocking=_nb)
        labels = labels.to(device, non_blocking=_nb)

        # v6.4: 50% MixUp, 50% CutMix (when enabled)
        if use_mixup and model.training:
            if random.random() < 0.5:
                imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
            else:
                imgs, y_a, y_b, lam = cutmix_data(imgs, labels, alpha=0.5)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)

        use_amp     = (scaler is not None) and amp_enabled and device.type == 'cuda'
        device_type = device.type

        if use_amp:
            with torch.amp.autocast(device_type):
                logits, mid_ent, fin_ent = model.forward_with_entropy(imgs)
                if use_mixup and lam < 1.0:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, labels)
                # v6.4: stronger entropy weight (0.01→0.03)
                entropy_loss = -entropy_weight * (mid_ent + fin_ent)
                loss = loss + entropy_loss
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            logits, mid_ent, fin_ent = model.forward_with_entropy(imgs)
            if use_mixup and lam < 1.0:
                loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
            else:
                loss = criterion(logits, labels)
            entropy_loss = -entropy_weight * (mid_ent + fin_ent)
            loss = loss + entropy_loss
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        total_loss += loss.item()
        with torch.no_grad():
            preds = logits.argmax(1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        n_batches += 1

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc        = (all_preds == all_labels).mean() * 100
    bal_acc    = balanced_accuracy(all_preds, all_labels) * 100

    return {
        'loss':    total_loss / max(n_batches, 1),
        'acc':     acc,
        'bal_acc': bal_acc,
    }


# ── Validation ─────────────────────────────────────────────────────────────────

def validate(model: SpatialModelV64, loader: DataLoader,
             criterion: AsymmetricFocalLoss, device: torch.device) -> Dict:
    model.eval()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    all_confs  = []
    n_batches  = 0

    with torch.no_grad():
        for batch_idx, (imgs, labels, paths) in enumerate(loader):
            _nb    = imgs.is_pinned()
            imgs   = imgs.to(device, non_blocking=_nb)
            labels = labels.to(device, non_blocking=_nb)
            logits, stream_norms = model.forward_with_streams(imgs)
            loss   = criterion(logits, labels)
            total_loss += loss.item()
            probs  = torch.softmax(logits, dim=1)
            preds  = logits.argmax(1)
            confs  = probs.max(1).values

            for j in range(len(labels)):
                model.error_memory.record(
                    true_label   = labels[j].item(),
                    pred_label   = preds[j].item(),
                    confidence   = confs[j].item(),
                    stream_norms = stream_norms,
                    image_path   = paths[j],
                )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            n_batches += 1

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc        = (all_preds == all_labels).mean() * 100
    bal_acc    = balanced_accuracy(all_preds, all_labels) * 100

    fake_mask   = all_labels == 0
    real_mask   = all_labels == 1
    fake_recall = (all_preds[fake_mask] == 0).mean() * 100 if fake_mask.sum() > 0 else 0.0
    real_recall = (all_preds[real_mask] == 1).mean() * 100 if real_mask.sum() > 0 else 0.0

    # v6.4: warn when fake recall drops below threshold
    if fake_recall < 70.0:
        print(f"  ⚠️  FAKE recall low: {fake_recall:.1f}% — model may be hallucinating REAL")

    return {
        'loss':        total_loss / max(n_batches, 1),
        'acc':         acc,
        'bal_acc':     bal_acc,
        'fake_recall': fake_recall,
        'real_recall': real_recall,
    }


# ── Val-split resolver ─────────────────────────────────────────────────────────

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
        print(f"GPU: {props.name}  VRAM: {props.total_memory/1e9:.1f} GB")

    os.makedirs(args.ckpt_dir, exist_ok=True)
    ckpt_path = os.path.join(args.ckpt_dir, 'checkpoint_latest.pt')
    best_path = os.path.join(args.ckpt_dir, 'checkpoint_best.pt')
    log_path  = os.path.join(args.ckpt_dir, 'train_log.jsonl')
    err_path  = os.path.join(args.ckpt_dir, 'error_memory.json')

    train_tf = get_transforms(args.input_size, is_train=True)
    val_tf   = get_transforms(args.input_size, is_train=False)

    train_dir = os.path.join(args.data_dir, 'train')
    val_dir   = _resolve_val_dir(args.data_dir, preferred=args.val_split)

    train_ds = ForensicDataset(train_dir, train_tf)
    val_ds   = ForensicDataset(val_dir,   val_tf)

    _use_persistent = args.workers > 0
    _loader_kwargs  = dict(
        num_workers        = args.workers,
        pin_memory         = args.workers > 0,
        collate_fn         = collate_fn,
        persistent_workers = _use_persistent,
    )
    if args.workers > 0:
        _loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True, **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        **_loader_kwargs,
    )

    model = get_spatial_model_v64(
        freeze_backbone    = True,
        unfreeze_last_n    = 0,
        dropout            = args.dropout,
        drop_path_rate     = args.drop_path,
        input_size         = args.input_size,
        use_grad_checkpoint= True,
    ).to(device)

    base_weights = train_ds.get_class_weights()

    # v6.4: AsymmetricFocalLoss with separate gamma for fake/real
    criterion = AsymmetricFocalLoss(
        gamma_fake      = args.focal_gamma_fake,
        gamma_real      = args.focal_gamma_real,
        weight          = base_weights.to(device),
        label_smoothing = args.label_smoothing,
    )

    # ── Parameter groups (v6.4: freq stream gets 2× head LR) ──
    head_params = (list(model.mlp.parameters()) +
                   list(model.classifier.parameters()) +
                   list(model.mid_pool.parameters()) +
                   list(model.final_pool.parameters()) +
                   list(model.cross_attn.parameters()))
    # Frequency stream gets higher LR — needs to learn faster than frozen ViT features
    freq_params  = list(model.freq_stream.parameters())
    dct_params   = list(model.dct_block.parameters())
    backbone_params = [p for p in model.extractor.parameters() if p.requires_grad]

    param_groups = [
        {'params': head_params,     'lr': args.lr,           'name': 'head'},
        {'params': freq_params,     'lr': args.lr * 2.0,     'name': 'freq_stream'},
        {'params': dct_params,      'lr': args.lr * 2.0,     'name': 'dct_block'},
        {'params': backbone_params, 'lr': args.lr * 0.1,     'name': 'backbone'},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    _amp_enabled = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=_amp_enabled) if _amp_enabled else None

    start_epoch  = 0
    best_val_acc = 0.0
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    if ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except (ValueError, KeyError):
            print("⚠️ Optimizer state mismatch — skipping optimizer load")
        try:
            scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        except (ValueError, KeyError):
            print("⚠️ Scheduler state mismatch — skipping scheduler load")
        start_epoch  = ckpt['epoch'] + 1
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"  Resumed from epoch {ckpt['epoch']}  best_val_acc={best_val_acc:.2f}%")
        model.error_memory.load(err_path)
    else:
        print("  No checkpoint found — starting from scratch")

    def maybe_unfreeze_backbone(epoch: int):
        if epoch == args.unfreeze_epoch and model.unfrozen_count == 0:
            model._unfreeze_last_n(args.unfreeze_last_n)
            model.classifier.unfreeze_scale()
            new_backbone = [p for p in model.extractor.parameters() if p.requires_grad]
            optimizer.add_param_group({'params': new_backbone, 'lr': args.lr * 0.05,
                                       'name': 'backbone_unfrozen'})
            print(f"  Epoch {epoch}: backbone unfrozen ({args.unfreeze_last_n} blocks)")

    patience   = args.patience
    no_improve = 0

    print(f"\nTraining for {args.epochs} epochs (start={start_epoch})")
    print(f"  batch={args.batch_size}  lr={args.lr}  mixup={args.mixup_alpha}"
          f"  focal_gamma_fake={args.focal_gamma_fake}  focal_gamma_real={args.focal_gamma_real}"
          f"  entropy_weight={args.entropy_weight}  warmup={args.warmup_epochs}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        maybe_unfreeze_backbone(epoch)

        adapted_w = model.error_memory.get_loss_weights(
            [base_weights[0].item(), base_weights[1].item()]
        )
        criterion.update_weight(torch.tensor(adapted_w, dtype=torch.float32))

        train_stats = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup       = args.mixup_alpha > 0,
            mixup_alpha     = args.mixup_alpha,
            entropy_weight  = args.entropy_weight,
            scaler          = scaler,
            amp_enabled     = _amp_enabled,
        )
        val_stats = validate(model, val_loader, criterion, device)
        scheduler.step()

        elapsed = time.time() - t0
        lr_now  = optimizer.param_groups[0]['lr']

        print(
            f"Epoch {epoch:03d}/{args.epochs-1} | "
            f"T_loss={train_stats['loss']:.4f} T_acc={train_stats['acc']:.1f}% "
            f"T_bal={train_stats['bal_acc']:.1f}% | "
            f"V_loss={val_stats['loss']:.4f} V_acc={val_stats['acc']:.1f}% "
            f"V_bal={val_stats['bal_acc']:.1f}% "
            f"FAKE_R={val_stats['fake_recall']:.1f}% REAL_R={val_stats['real_recall']:.1f}% | "
            f"LR={lr_now:.2e}  {elapsed:.0f}s"
        )

        err_summ = model.error_memory.summary()
        print(f"  ErrorMem: real_as_fake={err_summ['real_as_fake']}  "
              f"fake_as_real={err_summ['fake_as_real']}")

        log_entry = {
            'epoch': epoch, **train_stats,
            **{f'val_{k}': v for k, v in val_stats.items()},
            'lr': lr_now,
            **{f'err_{k}': v for k, v in err_summ.items() if isinstance(v, (int, float))},
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
            print(f"  ✅ New best bal_acc: {best_val_acc:.2f}%  → saved to {best_path}")
            no_improve = 0
        else:
            no_improve += 1
            if no_improve >= patience:
                print(f"  Early stopping after {patience} epochs without improvement.")
                break

        model.error_memory.save(err_path)

    print("\n=== Temperature Calibration ===")
    best_ckpt = load_checkpoint(best_path, map_location=device)
    if best_ckpt:
        model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
    model.calibrate(val_loader, device)

    print("\n=== Adversarial Robustness Check ===")
    adv_tester = AdversarialTester(model, device)
    adv_result = adv_tester.evaluate(val_loader, attack='fgsm', eps=8/255, max_batches=10)
    print(f"  Balanced drop: {adv_result['bal_drop']:.2f}%  {adv_result['verdict']}")

    final_path = os.path.join(args.ckpt_dir, 'model_final.pt')
    save_checkpoint({
        'model_state_dict': model.state_dict(),
        'epoch': 'final',
        'adv_result': adv_result,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print(f"Error memory: {model.error_memory.summary()}")


# ── CLI ────────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train SpatialModelV6.4')
    parser.add_argument('--data_dir',          required=True)
    parser.add_argument('--ckpt_dir',          default='./checkpoints')
    parser.add_argument('--val_split',         default='val')
    parser.add_argument('--epochs',            type=int,   default=30)
    parser.add_argument('--batch_size',        type=int,   default=8,
                        help='8 safe for RTX 2050 4GB')
    parser.add_argument('--lr',                type=float, default=3e-4)
    parser.add_argument('--weight_decay',      type=float, default=1e-4)
    parser.add_argument('--dropout',           type=float, default=0.25)
    parser.add_argument('--drop_path',         type=float, default=0.08)
    # v6.4: separate gamma for fake/real
    parser.add_argument('--focal_gamma_fake',  type=float, default=3.0,
                        help='Focal gamma for FAKE class (higher = harder focus on GAN images)')
    parser.add_argument('--focal_gamma_real',  type=float, default=2.0,
                        help='Focal gamma for REAL class (lower = preserve real accuracy)')
    parser.add_argument('--label_smoothing',   type=float, default=0.05)
    parser.add_argument('--mixup_alpha',       type=float, default=0.2,
                        help='MixUp alpha (reduced from 0.3 to avoid boundary blending)')
    parser.add_argument('--entropy_weight',    type=float, default=0.03,
                        help='Attention entropy weight (increased from 0.01)')
    parser.add_argument('--warmup_epochs',     type=int,   default=3)
    parser.add_argument('--unfreeze_epoch',    type=int,   default=5)
    parser.add_argument('--unfreeze_last_n',   type=int,   default=3,
                        help='ViT blocks to unfreeze (increased from 2 to 3)')
    parser.add_argument('--patience',          type=int,   default=8)
    parser.add_argument('--input_size',        type=int,   default=224)
    parser.add_argument('--workers',           type=int,   default=2)
    parser.add_argument('--seed',              type=int,   default=42)

    args = parser.parse_args()
    train(args)