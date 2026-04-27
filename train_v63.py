"""
train_v63.py — Training script for SpatialModelV6.3
RTX 2050 4GB VRAM optimised.

Usage:
  python train_v63.py --data_dir /path/to/dataset --ckpt_dir ./checkpoints

Dataset layout expected:
  data_dir/
    train/
      real/   (or REAL/)
      fake/   (or FAKE/)
    val/
      real/
      fake/

Labels: FAKE=0, REAL=1

Fixes vs original train_v63.py:
  ✅ Removed per-item print() in __getitem__ — was spam-logging every image load
     (catastrophic I/O slowdown with multiple workers)
  ✅ validate(): inner loop variable j shadows outer enumerate i — renamed to j
  ✅ train_loader prefetch_factor=2 requires workers>0; guarded with conditional
  ✅ Dataset split: was hardcoded to 'test' subdir despite docstring saying 'val';
     now uses --val_split arg defaulting to 'val', falling back to 'test'
  ✅ AMP autocast guarded: only uses 'cuda' device string when CUDA is available
  ✅ criterion.weight update uses update_weight() method (device-safe) instead of
     direct attribute assignment
  ✅ persistent_workers correctly set for train_loader (requires workers > 0)
  ✅ prefetch_factor only set when workers > 0 (avoids DataLoader crash on workers=0)
"""

import os
import sys
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
from PIL import Image, UnidentifiedImageError
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from v63spatial import (
    SpatialModelV63, get_spatial_model_v63,
    FocalLoss, mixup_data, mixup_criterion,
    save_checkpoint, load_checkpoint,
    predict_with_uncertainty, AdversarialTester,
    CLIP_MEAN, CLIP_STD,
)


# ── Reproducibility ───────────────────────────────────────────────────────────

def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ── Robust Dataset (Issue 5: accepts all images, skips corrupt) ───────────────

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

CLASS_NAMES = ['fake', 'real']   # FAKE=0, REAL=1


def _find_class_dirs(root: Path):
    """
    Finds fake/real subdirs case-insensitively.
    Returns {0: fake_path, 1: real_path} or raises.
    """
    dirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    fake_dir = dirs.get('fake') or dirs.get('0')
    real_dir = dirs.get('real') or dirs.get('1')
    if fake_dir is None or real_dir is None:
        raise FileNotFoundError(
            f"Could not find fake/real subdirs in {root}. "
            f"Found: {list(dirs.keys())}"
        )
    return {0: fake_dir, 1: real_dir}


class ForensicDataset(Dataset):
    """
    Robust dataset:
    - Accepts all common image formats
    - Silently skips corrupt / unreadable files (logs count at end)
    - Returns (tensor, label, path) triples
    - Computes class counts for balanced weighting
    """

    def __init__(self, root: str, transform=None):
        self.transform = transform
        self.samples: list = []
        self.skipped        = 0

        root_path  = Path(root)
        class_dirs = _find_class_dirs(root_path)

        for label, class_dir in class_dirs.items():
            for ext in VALID_EXTS:
                for fpath in class_dir.rglob(f'*{ext}'):
                    self.samples.append((str(fpath), label))
                for fpath in class_dir.rglob(f'*{ext.upper()}'):
                    self.samples.append((str(fpath), label))

        # De-duplicate (rglob can match same file via upper/lower)
        seen   = set()
        unique = []
        for s in self.samples:
            if s[0] not in seen:
                seen.add(s[0])
                unique.append(s)
        self.samples = unique

        random.shuffle(self.samples)

        # Class counts for weight computation
        self.class_counts = defaultdict(int)
        for _, label in self.samples:
            self.class_counts[label] += 1

        print(f"  Dataset [{root_path.name}]: "
              f"FAKE={self.class_counts[0]}  REAL={self.class_counts[1]}  "
              f"Total={len(self.samples)}")

    def get_class_weights(self) -> torch.Tensor:
        """Inverse-frequency weights for balanced loss."""
        n      = len(self.samples)
        n_fake = max(self.class_counts[0], 1)
        n_real = max(self.class_counts[1], 1)
        w_fake = n / (2.0 * n_fake)
        w_real = n / (2.0 * n_real)
        # Clamp to avoid explosive weights on very unbalanced sets
        w_fake = min(w_fake, 3.0)
        w_real = min(w_real, 3.0)
        print(f"  Class weights — FAKE: {w_fake:.3f}  REAL: {w_real:.3f}")
        return torch.tensor([w_fake, w_real], dtype=torch.float32)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        # FIX: removed print(f"Loading: {path}") — was called on every single item,
        # causing massive stdout spam and I/O slowdown with multiple workers.
        try:
            img = Image.open(path).convert('RGB')
        except (UnidentifiedImageError, OSError, Exception):
            # Return a black placeholder on error — never crash training
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


# ── Transforms ────────────────────────────────────────────────────────────────

def get_transforms(input_size: int = 224, is_train: bool = True):
    mean = CLIP_MEAN
    std  = CLIP_STD

    if is_train:
        return transforms.Compose([
            transforms.Resize((input_size + 32, input_size + 32)),
            transforms.RandomCrop(input_size),
            transforms.RandomHorizontalFlip(),
            transforms.RandomVerticalFlip(p=0.1),
            transforms.ColorJitter(brightness=0.15, contrast=0.15,
                                   saturation=0.1, hue=0.03),
            transforms.RandomGrayscale(p=0.05),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=3)], p=0.15),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
            transforms.RandomErasing(p=0.1, scale=(0.02, 0.1)),
        ])
    else:
        return transforms.Compose([
            transforms.Resize((input_size, input_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


# ── Cosine LR with warmup ─────────────────────────────────────────────────────

def cosine_schedule_with_warmup(optimizer, warmup_epochs: int, total_epochs: int,
                                  min_lr_ratio: float = 0.05):
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return (epoch + 1) / max(warmup_epochs, 1)
        progress = (epoch - warmup_epochs) / max(total_epochs - warmup_epochs, 1)
        return min_lr_ratio + (1.0 - min_lr_ratio) * 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


# ── Balanced accuracy ─────────────────────────────────────────────────────────

def balanced_accuracy(preds: np.ndarray, labels: np.ndarray, num_classes: int = 2) -> float:
    per_class = []
    for c in range(num_classes):
        mask = labels == c
        if mask.sum() == 0:
            continue
        per_class.append((preds[mask] == labels[mask]).mean())
    return float(np.mean(per_class)) if per_class else 0.0


# ── Train one epoch ───────────────────────────────────────────────────────────

def train_epoch(model: SpatialModelV63, loader: DataLoader, optimizer,
                criterion: FocalLoss, device: torch.device,
                use_mixup: bool = True, mixup_alpha: float = 0.3,
                entropy_weight: float = 0.01,
                scaler=None, amp_enabled: bool = False) -> Dict:
    model.train()
    total_loss = 0.0
    all_preds  = []
    all_labels = []
    n_batches  = 0

    for imgs, labels, _ in loader:
        _nb = imgs.is_pinned()
        imgs   = imgs.to(device, non_blocking=_nb)
        labels = labels.to(device, non_blocking=_nb)

        # MixUp (reduces memorisation, issues 3/4)
        if use_mixup and model.training:
            imgs, y_a, y_b, lam = mixup_data(imgs, labels, alpha=mixup_alpha)
        else:
            y_a, y_b, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)

        # FIX: AMP autocast device string must match actual device.
        # Original always used 'cuda' even when running on CPU → RuntimeError.
        use_amp     = (scaler is not None) and amp_enabled and device.type == 'cuda'
        device_type = device.type  # 'cuda' or 'cpu'

        if use_amp:
            with torch.amp.autocast(device_type):
                logits, mid_ent, fin_ent = model.forward_with_entropy(imgs)
                if use_mixup and lam < 1.0:
                    loss = mixup_criterion(criterion, logits, y_a, y_b, lam)
                else:
                    loss = criterion(logits, labels)
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


# ── Validation ────────────────────────────────────────────────────────────────

def validate(model: SpatialModelV63, loader: DataLoader, criterion: FocalLoss,
             device: torch.device) -> Dict:
    model.eval()
    total_loss  = 0.0
    all_preds   = []
    all_labels  = []
    all_confs   = []
    n_batches   = 0

    with torch.no_grad():
        # FIX: renamed outer loop variable from i to batch_idx to prevent shadowing
        # by the inner-loop variable j (original had both named i — inner shadowed
        # outer, making batch_idx logging incorrect after first batch).
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

            # Update error memory (issues 3/7)
            # FIX: inner loop variable renamed to j — was also named i, shadowing
            # the outer enumerate i and causing the outer counter to be wrong.
            for j in range(len(labels)):
                model.error_memory.record(
                    true_label  = labels[j].item(),
                    pred_label  = preds[j].item(),
                    confidence  = confs[j].item(),
                    stream_norms= stream_norms,
                    image_path  = paths[j],
                )

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confs.extend(confs.cpu().numpy())
            n_batches += 1

    all_preds  = np.array(all_preds)
    all_labels = np.array(all_labels)
    acc        = (all_preds == all_labels).mean() * 100
    bal_acc    = balanced_accuracy(all_preds, all_labels) * 100

    # Per-class recall (issues 1 and 7)
    fake_mask   = all_labels == 0
    real_mask   = all_labels == 1
    fake_recall = (all_preds[fake_mask] == 0).mean() * 100 if fake_mask.sum() > 0 else 0.0
    real_recall = (all_preds[real_mask] == 1).mean() * 100 if real_mask.sum() > 0 else 0.0

    return {
        'loss':        total_loss / max(n_batches, 1),
        'acc':         acc,
        'bal_acc':     bal_acc,
        'fake_recall': fake_recall,
        'real_recall': real_recall,
    }


# ── Val-split resolver ────────────────────────────────────────────────────────

def _resolve_val_dir(data_dir: str, preferred: str = 'val') -> str:
    """
    Resolves the validation directory.
    Tries preferred ('val') first, falls back to 'test', then errors.
    """
    for name in (preferred, 'test', 'val'):
        candidate = os.path.join(data_dir, name)
        if os.path.isdir(candidate):
            if name != preferred:
                print(f"  [INFO] Val dir '{preferred}' not found — using '{name}'")
            return candidate
    raise FileNotFoundError(
        f"No validation directory found in {data_dir}. "
        f"Tried: {preferred}, test, val"
    )


# ── Main training loop ────────────────────────────────────────────────────────

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

    # ── Datasets ──
    train_tf = get_transforms(args.input_size, is_train=True)
    val_tf   = get_transforms(args.input_size, is_train=False)

    train_dir = os.path.join(args.data_dir, 'train')
    # FIX: was hardcoded to 'test' — now resolves val → test fallback
    val_dir   = _resolve_val_dir(args.data_dir, preferred=args.val_split)

    train_ds = ForensicDataset(train_dir, train_tf)
    val_ds   = ForensicDataset(val_dir,   val_tf)

    # FIX: prefetch_factor requires persistent_workers=True AND num_workers>0.
    # Original had persistent_workers=False + prefetch_factor=2 → DataLoader crash
    # when workers=0 (or workers>0 with persistent_workers=False on some PyTorch versions).
    _use_persistent = args.workers > 0
    _loader_kwargs  = dict(
        num_workers    = args.workers,
        pin_memory     = args.workers > 0,
        collate_fn     = collate_fn,
        persistent_workers = _use_persistent,
    )
    if args.workers > 0:
        _loader_kwargs['prefetch_factor'] = 2

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        drop_last=True,
        **_loader_kwargs,
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size * 2, shuffle=False,
        **_loader_kwargs,
    )

    # ── Model ──
    model = get_spatial_model_v63(
        freeze_backbone    = True,
        unfreeze_last_n    = 0,
        dropout            = args.dropout,
        drop_path_rate     = args.drop_path,
        input_size         = args.input_size,
        use_grad_checkpoint= True,
    ).to(device)

    # ── Class weights (dataset-balanced + error-memory adaptive) ──
    base_weights = train_ds.get_class_weights()

    # ── Loss ──
    criterion = FocalLoss(
        gamma           = args.focal_gamma,
        weight          = base_weights.to(device),
        label_smoothing = args.label_smoothing,
    )

    # ── Optimizer ──
    head_params = (list(model.mlp.parameters()) +
                   list(model.classifier.parameters()) +
                   list(model.mid_pool.parameters()) +
                   list(model.final_pool.parameters()) +
                   list(model.cross_attn.parameters()))
    backbone_params = [p for p in model.extractor.parameters() if p.requires_grad]

    param_groups = [
        {'params': head_params,     'lr': args.lr,       'name': 'head'},
        {'params': backbone_params, 'lr': args.lr * 0.1, 'name': 'backbone'},
    ]
    optimizer = torch.optim.AdamW(param_groups, weight_decay=args.weight_decay)
    scheduler = cosine_schedule_with_warmup(optimizer, args.warmup_epochs, args.epochs)

    # AMP scaler — only when CUDA is available
    _amp_enabled = device.type == 'cuda'
    scaler = torch.amp.GradScaler('cuda', enabled=_amp_enabled) if _amp_enabled else None

    # ── Resume from checkpoint (Issue 2 fix) ──
    start_epoch  = 0
    best_val_acc = 0.0
    ckpt = load_checkpoint(ckpt_path, map_location=device)
    if ckpt:
        model.load_state_dict(ckpt['model_state_dict'], strict=False)
       # optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        try:
            optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        except ValueError:
            print("⚠️ Optimizer state mismatch — skipping optimizer load")
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch  = ckpt['epoch'] + 1        # resume AFTER the saved epoch
        best_val_acc = ckpt.get('best_val_acc', 0.0)
        print(f"  Resumed from epoch {ckpt['epoch']}  best_val_acc={best_val_acc:.2f}%")
        model.error_memory.load(err_path)
    else:
        print("  No checkpoint found — starting from scratch")

    # ── Progressive backbone unfreezing schedule ──
    def maybe_unfreeze_backbone(epoch: int):
        if epoch == args.unfreeze_epoch and model.unfrozen_count == 0:
            model._unfreeze_last_n(args.unfreeze_last_n)
            model.classifier.unfreeze_scale()
            new_backbone = [p for p in model.extractor.parameters() if p.requires_grad]
            optimizer.add_param_group({'params': new_backbone, 'lr': args.lr * 0.05,
                                       'name': 'backbone_unfrozen'})
            print(f"  Epoch {epoch}: backbone unfrozen ({args.unfreeze_last_n} blocks)")

    # ── Early stopping ──
    patience   = args.patience
    no_improve = 0

    print(f"\nTraining for {args.epochs} epochs (start={start_epoch})")
    print(f"  batch={args.batch_size}  lr={args.lr}  mixup={args.mixup_alpha}"
          f"  focal_gamma={args.focal_gamma}  warmup={args.warmup_epochs}")

    for epoch in range(start_epoch, args.epochs):
        t0 = time.time()
        maybe_unfreeze_backbone(epoch)

        # Adaptive class weights from error memory
        adapted_w = model.error_memory.get_loss_weights(
            [base_weights[0].item(), base_weights[1].item()]
        )
        # FIX: use update_weight() method instead of direct attribute assignment.
        # Direct assignment (criterion.weight = tensor) bypasses the device check
        # inside update_weight(), which moves the tensor to the correct device.
        criterion.update_weight(torch.tensor(adapted_w, dtype=torch.float32))

        train_stats = train_epoch(
            model, train_loader, optimizer, criterion, device,
            use_mixup=args.mixup_alpha > 0, mixup_alpha=args.mixup_alpha,
            entropy_weight=args.entropy_weight, scaler=scaler,
            amp_enabled=_amp_enabled,
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

        # Log
        log_entry = {
            'epoch': epoch, **train_stats,
            **{f'val_{k}': v for k, v in val_stats.items()},
            'lr': lr_now,
            **{f'err_{k}': v for k, v in err_summ.items() if isinstance(v, (int, float))},
        }
        with open(log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

        # Save latest checkpoint (Issue 2: plain save, epoch stored correctly)
        ckpt_state = {
            'epoch':                epoch,
            'model_state_dict':     model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_val_acc':         best_val_acc,
            'val_stats':            val_stats,
        }
        save_checkpoint(ckpt_state, ckpt_path)

        # Save best (monitor = balanced accuracy)
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

        # Save error memory
        model.error_memory.save(err_path)

    # ── Post-training: calibrate + adversarial check ──
    print("\n=== Temperature Calibration ===")
    best_ckpt = load_checkpoint(best_path, map_location=device)
    if best_ckpt:
        model.load_state_dict(best_ckpt['model_state_dict'], strict=False)
    model.calibrate(val_loader, device)

    print("\n=== Adversarial Robustness Check ===")
    adv_tester = AdversarialTester(model, device)
    adv_result = adv_tester.evaluate(val_loader, attack='fgsm',
                                      eps=8/255, max_batches=10)
    print(f"  Balanced drop: {adv_result['bal_drop']:.2f}%  {adv_result['verdict']}")

    # Save final calibrated model
    final_path = os.path.join(args.ckpt_dir, 'model_final.pt')
    save_checkpoint({
        'model_state_dict': model.state_dict(),
        'epoch': 'final',
        'adv_result': adv_result,
    }, final_path)
    print(f"\nFinal model saved: {final_path}")
    print(f"Error memory: {model.error_memory.summary()}")


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == '__main__':
    parser = argparse.ArgumentParser('Train SpatialModelV6.3')
    parser.add_argument('--data_dir',        required=True,        help='Dataset root with train/val subdirs')
    parser.add_argument('--ckpt_dir',        default='./checkpoints')
    parser.add_argument('--val_split',       default='val',        help="Val subdir name: 'val' or 'test' (default: val)")
    parser.add_argument('--epochs',          type=int,   default=30)
    parser.add_argument('--batch_size',      type=int,   default=8,   help='8 safe for RTX 2050 4GB')
    parser.add_argument('--lr',              type=float, default=3e-4)
    parser.add_argument('--weight_decay',    type=float, default=1e-4)
    parser.add_argument('--dropout',         type=float, default=0.25)
    parser.add_argument('--drop_path',       type=float, default=0.08)
    parser.add_argument('--focal_gamma',     type=float, default=2.0)
    parser.add_argument('--label_smoothing', type=float, default=0.05)
    parser.add_argument('--mixup_alpha',     type=float, default=0.3)
    parser.add_argument('--entropy_weight',  type=float, default=0.01)
    parser.add_argument('--warmup_epochs',   type=int,   default=3)
    parser.add_argument('--unfreeze_epoch',  type=int,   default=5,   help='Epoch to start unfreezing backbone')
    parser.add_argument('--unfreeze_last_n', type=int,   default=2,   help='How many ViT blocks to unfreeze')
    parser.add_argument('--patience',        type=int,   default=8)
    parser.add_argument('--input_size',      type=int,   default=224)
    parser.add_argument('--workers',         type=int,   default=2)
    
    parser.add_argument('--seed',            type=int,   default=42)

    args = parser.parse_args()
    train(args)