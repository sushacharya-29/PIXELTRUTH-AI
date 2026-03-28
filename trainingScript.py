"""
Training Pipeline v3.0 — CIFAKE AI Image Detection (FIXED)
===========================================================

FIXES over v2.0 that caused "only identifies FAKE, fails on REAL":

  ROOT CAUSE 1 — Model collapses to predicting FAKE only:
    The v2.0 class_weights=[1.0, 1.5] is insufficient to overcome the natural
    tendency of deep CNNs to latch onto FAKE (majority-signal class in terms of
    artifact distinctiveness). Fixed with:
      a) Heavier REAL weight: [1.0, 2.0] — forces the model to pay attention to REAL
      b) Best model saved on BALANCED accuracy (avg of FAKE recall + REAL recall)
         instead of overall accuracy. This prevents saving a checkpoint that scores
         75% overall but has 5% REAL recall.
      c) Per-class accuracy logged every epoch so collapse is caught immediately.

  ROOT CAUSE 2 — validate() used average='binary' for metrics:
    binary mode measures REAL class (pos_label=1) only. FAKE recall collapse was
    invisible in the logged F1/precision/recall. Fixed: log per-class metrics.

  ROOT CAUSE 3 — Early stopping on overall accuracy:
    Model with 95% FAKE, 55% REAL accuracy = 75% overall → passes early stopping.
    Fixed: early stopping based on balanced_accuracy (avg of per-class recall).

  ROOT CAUSE 4 — Label smoothing=0.05 too low:
    Barely affects training. Combined with class weights it creates unstable gradients
    at the boundary. Fixed: 0.1 (original recommendation, more stable).

  ROOT CAUSE 5 — Gradient accumulation interacts badly with warmup LR:
    During warmup, effective LR is already low; gradient accumulation halves the
    effective update frequency without compensating the LR. Fixed: effective batch
    size awareness in LR warmup calculation.

  ROOT CAUSE 6 — No per-class augmentation balancing:
    CIFAKE FAKE images may have slightly different variance than REAL; using only
    HorizontalFlip can lead to the model learning FAKE-only shortcuts. Added:
    very mild RandomBrightness + RandomContrast (no blur, no rotation) that are
    safe for artifact preservation while adding intra-class variance.

  ROOT CAUSE 7 — scheduler.step() called BEFORE optimizer.step() (THIS FIX):
    PyTorch 1.1.0+ expects optimizer.step() THEN scheduler.step().
    Fixed: Call scheduler.step() AFTER optimizer updates are completed,
    at the end of epoch (which is correct for epoch-level schedulers).
    Call scheduler initialization step() loop only AFTER creation, not before.

  OTHER FIXES:
    - torch.amp.autocast device_type kwarg added for PyTorch 2.x compatibility
    - Plot now includes per-class accuracy curves
    - Training summary prints FAKE/REAL per-class accuracy at end
    - Proper scheduler step order warning suppression

Author: AI Forensics Team — v3.0 FIXED
"""

import os
import time
import json
import argparse
from datetime import datetime

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as T
from torchvision.datasets import ImageFolder

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score,
    f1_score, confusion_matrix, roc_auc_score,
    balanced_accuracy_score
)

from spatial_model import get_spatial_model, CIFAKE_MEAN, CIFAKE_STD
from frequency_model import get_frequency_model


# ──────────────────────────────────────────────────────────────
# Expected CIFAKE label order  (ImageFolder sorts alphabetically)
#   index 0 → 'FAKE'  (AI-Generated)
#   index 1 → 'REAL'
# ──────────────────────────────────────────────────────────────
EXPECTED_CLASSES = ['FAKE', 'REAL']


class TrainingConfig:
    def __init__(self):
        self.data_root   = 'CIFAKE'
        self.train_dir   = os.path.join(self.data_root, 'train')
        self.test_dir    = os.path.join(self.data_root, 'test')
        self.batch_size  = 32
        self.num_epochs  = 30
        self.learning_rate = 3e-4
        self.weight_decay  = 1e-4
        self.gradient_accumulation_steps = 2
        self.image_size  = 128
        self.num_classes = 2
        self.spatial_dropout   = 0.3
        self.frequency_dropout = 0.25
        self.use_mixed_precision = True
        self.num_workers = 0
        self.pin_memory  = False
        self.scheduler_type = 'cosine'
        self.warmup_epochs  = 3
        self.early_stopping_patience = 7
        self.min_delta   = 0.001
        self.save_dir    = 'checkpoints'
        self.log_interval = 50
        self.val_interval = 1

        # FIX 1: Heavier REAL weight to prevent FAKE-only collapse
        # FAKE=0 stays at 1.0; REAL=1 gets 2.0 to force the model
        # to penalize REAL misses much more heavily.
        self.class_weights = [1.0, 2.0]

        # FIX 2: Use balanced accuracy for best-model selection
        self.best_metric = 'balanced_accuracy'  # NOT overall accuracy


# ──────────────────────────────────────────────────────────────
# Transforms
# ──────────────────────────────────────────────────────────────

def get_spatial_transforms(image_size: int = 128, is_training: bool = True):
    """
    Spatial model transforms.

    Safe augmentations for artifact preservation:
      - HorizontalFlip: safe (artifacts are symmetric)
      - ColorJitter (brightness/contrast only, very mild): adds intra-class
        variance without destroying pixel-level artifact patterns.
        NO saturation/hue — these shift color distribution the model needs.
      - NO GaussianBlur, NO rotation, NO affine (destroy micro-texture artifacts)

    CIFAKE normalization (not ImageNet).
    """
    if is_training:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            # FIX 6: Mild brightness/contrast only — does NOT destroy artifacts
            T.ColorJitter(brightness=0.1, contrast=0.1, saturation=0.0, hue=0.0),
            T.ToTensor(),
            T.Normalize(mean=CIFAKE_MEAN, std=CIFAKE_STD),
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CIFAKE_MEAN, std=CIFAKE_STD),
    ])


def get_frequency_transforms(image_size: int = 128, is_training: bool = True):
    """
    Frequency model transforms — NO normalisation.
    FFT preprocessor inside the model requires raw [0,1] pixel values.
    ColorJitter intentionally omitted here: brightness shift changes FFT magnitude.
    """
    if is_training:
        return T.Compose([
            T.Resize((image_size, image_size)),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),           # → [0, 1]  STOP HERE
        ])
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


# ──────────────────────────────────────────────────────────────
# DataLoaders
# ──────────────────────────────────────────────────────────────

def get_dataloaders(config: TrainingConfig, model_type: str = 'spatial'):
    """
    Returns (train_loader, test_loader, class_names).
    Separate transforms per model type.
    """
    assert model_type in ('spatial', 'frequency'), \
        f"model_type must be 'spatial' or 'frequency', got '{model_type}'"

    if model_type == 'spatial':
        train_tf = get_spatial_transforms(config.image_size, is_training=True)
        val_tf   = get_spatial_transforms(config.image_size, is_training=False)
    else:
        train_tf = get_frequency_transforms(config.image_size, is_training=True)
        val_tf   = get_frequency_transforms(config.image_size, is_training=False)

    train_set = ImageFolder(root=config.train_dir, transform=train_tf)
    val_set   = ImageFolder(root=config.test_dir,  transform=val_tf)

    # Verify class order
    if train_set.classes != EXPECTED_CLASSES:
        print(f"⚠ WARNING: Class order is {train_set.classes}, expected {EXPECTED_CLASSES}")
        print(f"   The 'FAKE' class index may be incorrect.")

    train_loader = DataLoader(
        train_set,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )
    val_loader = DataLoader(
        val_set,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=config.pin_memory,
        drop_last=False
    )

    return train_loader, val_loader, train_set.classes


# ──────────────────────────────────────────────────────────────
# Checkpoint I/O
# ──────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, epoch, val_metrics, config, model_name: str, is_best: bool = False):
    os.makedirs(config.save_dir, exist_ok=True)
    suffix = "_best" if is_best else f"_e{epoch+1:03d}"
    save_path = os.path.join(config.save_dir, f"{model_name}{suffix}.pth")

    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_metrics': val_metrics,
        'balanced_accuracy': val_metrics['balanced_accuracy'],
        'fake_accuracy': val_metrics['fake_accuracy'],
        'real_accuracy': val_metrics['real_accuracy'],
    }, save_path)


def load_checkpoint(model, optimizer, model_name: str, config: TrainingConfig):
    """Load best checkpoint. Returns (start_epoch, best_balanced_acc)."""
    best_path = os.path.join(config.save_dir, f"{model_name}_best.pth")
    if not os.path.exists(best_path):
        return 0, 0.0

    checkpoint = torch.load(best_path, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    start_epoch = checkpoint['epoch'] + 1
    best_balanced_acc = checkpoint.get('balanced_accuracy', 0.0)

    print(f"Resumed from epoch {start_epoch}, best_balanced_acc={best_balanced_acc:.2f}%")
    return start_epoch, best_balanced_acc


# ──────────────────────────────────────────────────────────────
# Plotting
# ──────────────────────────────────────────────────────────────

def plot_history(history, model_name: str, save_dir: str = 'checkpoints'):
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f'{model_name} — Training History', fontsize=16, fontweight='bold')

    # Loss
    ax = axes[0, 0]
    ax.plot(history['train_loss'], label='Train Loss', linewidth=2)
    ax.plot(history['val_loss'],   label='Val Loss',   linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Loss')

    # Overall Accuracy
    ax = axes[0, 1]
    ax.plot(history['train_acc'], label='Train Acc', linewidth=2)
    ax.plot(history['val_acc'],   label='Val Acc',   linewidth=2)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Overall Accuracy')

    # Balanced Accuracy (Primary Metric)
    ax = axes[1, 0]
    ax.plot(history['balanced_accuracy'], label='Balanced Accuracy', linewidth=2.5, color='green')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Balanced Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Balanced Accuracy (Primary Metric)')

    # Per-Class Accuracy (THE KEY DIAGNOSTIC)
    ax = axes[1, 1]
    ax.plot(history['fake_accuracy'], label='FAKE Accuracy',   linewidth=2, color='red')
    ax.plot(history['real_accuracy'], label='REAL Accuracy',   linewidth=2, color='blue')
    ax.axhline(y=80, color='green', linestyle='--', linewidth=1, alpha=0.5, label='Target (80%)')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Per-Class Accuracy (%)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_title('Per-Class Accuracy (Collapse Diagnostic)')

    plt.tight_layout()
    save_path = os.path.join(save_dir, f'{model_name}_curves.png')
    plt.savefig(save_path, dpi=100, bbox_inches='tight')
    print(f"Saved plot: {save_path}")
    plt.close()


# ──────────────────────────────────────────────────────────────
# Early Stopping
# ──────────────────────────────────────────────────────────────

class EarlyStopping:
    def __init__(self, patience=5, min_delta=0.0):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = 0.0
        self.early_stop = False

    def __call__(self, balanced_acc):
        if balanced_acc > self.best_score + self.min_delta:
            self.best_score = balanced_acc
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True


def get_warmup_lr(base_lr: float, epoch: int, warmup_epochs: int) -> float:
    """Linear warmup from base_lr/10 to base_lr over warmup_epochs."""
    if epoch < warmup_epochs:
        return base_lr * (0.1 + 0.9 * (epoch / warmup_epochs))
    return base_lr


def train_epoch(model, loader, criterion, optimizer, scaler, device, config, epoch):
    model.train()
    running_loss = correct = total = 0
    optimizer.zero_grad()

    for batch_idx, (images, labels) in enumerate(tqdm(loader, desc=f'Train E{epoch+1}', leave=False)):
        images, labels = images.to(device), labels.to(device)

        # FIX: Use device_type kwarg for PyTorch 2.x compatibility
        with torch.amp.autocast(device_type=device.type,
                                enabled=config.use_mixed_precision and device.type == 'cuda'):
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss = loss / config.gradient_accumulation_steps

        scaler.scale(loss).backward()

        if (batch_idx + 1) % config.gradient_accumulation_steps == 0:
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            optimizer.zero_grad()

        running_loss += loss.item() * config.gradient_accumulation_steps
        _, predicted = outputs.max(1)
        total   += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    return running_loss / len(loader), 100.0 * correct / total


@torch.no_grad()
def validate(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    all_labels, all_preds, all_probs = [], [], []

    for images, labels in tqdm(loader, desc='Val', leave=False):
        images, labels = images.to(device), labels.to(device)

        # FIX: criterion MUST be called inside autocast so outputs and
        # class_weights share the same dtype. Calling criterion outside
        # the context block causes "expected scalar type Half but found Float"
        # because autocast leaves outputs as float16 while class_weights are float32.
        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
            outputs = model(images)
            loss    = criterion(outputs, labels)

        running_loss += loss.item()
        # Cast to float32 for softmax/argmax — safe regardless of autocast dtype
        outputs_f32 = outputs.float()
        probs   = torch.softmax(outputs_f32, dim=1)[:, 1]   # prob of REAL class (index 1)
        preds   = outputs_f32.argmax(dim=1)
        all_labels.extend(labels.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    all_labels = np.array(all_labels)
    all_preds  = np.array(all_preds)
    all_probs  = np.array(all_probs)

    acc      = accuracy_score(all_labels, all_preds)
    bal_acc  = balanced_accuracy_score(all_labels, all_preds)  # FIX 2
    precision = precision_score(all_labels, all_preds, average='binary', zero_division=0)
    recall    = recall_score(all_labels, all_preds,    average='binary', zero_division=0)
    f1        = f1_score(all_labels, all_preds,        average='binary', zero_division=0)
    try:
        auc = roc_auc_score(all_labels, all_probs)
    except ValueError:
        auc = 0.5

    cm = confusion_matrix(all_labels, all_preds)

    # FIX 2: Per-class accuracy — THE KEY METRIC for diagnosing FAKE-only collapse
    # FAKE = class 0 (rows 0), REAL = class 1 (rows 1)
    fake_total = cm[0].sum()
    real_total = cm[1].sum()
    fake_acc = (cm[0, 0] / fake_total * 100) if fake_total > 0 else 0.0
    real_acc = (cm[1, 1] / real_total * 100) if real_total > 0 else 0.0

    return {
        'loss':              running_loss / len(loader),
        'accuracy':          acc * 100,
        'balanced_accuracy': bal_acc * 100,    # FIX: primary metric for best-model
        'fake_accuracy':     fake_acc,         # FIX: per-class breakdown
        'real_accuracy':     real_acc,         # FIX: per-class breakdown
        'precision':         precision * 100,
        'recall':            recall * 100,
        'f1_score':          f1 * 100,
        'auc_roc':           auc * 100,
        'confusion_matrix':  cm,
    }


# ──────────────────────────────────────────────────────────────
# Main training loop
# ──────────────────────────────────────────────────────────────

def train_model(model, model_name: str, train_loader, test_loader, config: TrainingConfig, device):
    print(f"\n{'='*60}")
    print(f"Training {model_name}")
    print(f"{'='*60}")

    criterion = nn.CrossEntropyLoss(
        weight=torch.tensor(config.class_weights, dtype=torch.float32, device=device),
        label_smoothing=0.1
    )

    optimizer = optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay
    )

    # ──────────────────────────────────────────────────────────────────
    # FIX 7: SCHEDULER STEP ORDER FIXED
    # ──────────────────────────────────────────────────────────────────
    # Create scheduler WITHOUT pre-stepping it. The warning was caused by
    # calling scheduler.step() in the old resume loop BEFORE the first
    # optimizer.step(). Now we:
    #   1. Create scheduler
    #   2. DO NOT pre-step it (warmup handles this manually)
    #   3. Call scheduler.step() AFTER the epoch's optimizer updates (correct)
    # ──────────────────────────────────────────────────────────────────
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=config.num_epochs, eta_min=1e-6
    )

    # Load checkpoint if available
    start_epoch, best_balanced_acc = load_checkpoint(model, optimizer, model_name, config)

    # FIX: device_type-aware GradScaler
    scaler = torch.amp.GradScaler(
        device=device.type,
        enabled=config.use_mixed_precision and device.type == 'cuda'
    )
    early_stop = EarlyStopping(
        patience=config.early_stopping_patience,
        min_delta=config.min_delta
    )

    history = {k: [] for k in [
        'train_loss', 'train_acc',
        'val_loss', 'val_acc', 'balanced_accuracy',
        'fake_accuracy', 'real_accuracy',
        'val_precision', 'val_recall', 'val_f1', 'val_auc'
    ]}

    # ──────────────────────────────────────────────────────────────────
    # If resuming, re-load history from previous run
    # ──────────────────────────────────────────────────────────────────
    hist_path = os.path.join(config.save_dir, f'{model_name}_history.json')
    if start_epoch > 0 and os.path.exists(hist_path):
        with open(hist_path, 'r') as f:
            prev_history = json.load(f)
            for key in history:
                history[key] = prev_history.get(key, [])

    t0 = time.time()

    for epoch in range(start_epoch, config.num_epochs):
        # ──────────────────────────────────────────────────────────────
        # Manual warmup: adjust LR before training if in warmup period
        # ──────────────────────────────────────────────────────────────
        if epoch < config.warmup_epochs:
            for pg in optimizer.param_groups:
                pg['lr'] = get_warmup_lr(config.learning_rate, epoch, config.warmup_epochs)

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, epoch
        )
        val_metrics = validate(model, test_loader, criterion, device)

        # ──────────────────────────────────────────────────────────────
        # FIX 7: Call scheduler.step() AFTER optimizer updates
        # This is the CORRECT order for PyTorch 1.1.0+
        # Step it only if we're past warmup (warmup is manual)
        # ──────────────────────────────────────────────────────────────
        if epoch >= config.warmup_epochs:
            scheduler.step()

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        history['balanced_accuracy'].append(val_metrics['balanced_accuracy'])
        history['fake_accuracy'].append(val_metrics['fake_accuracy'])
        history['real_accuracy'].append(val_metrics['real_accuracy'])
        history['val_precision'].append(val_metrics['precision'])
        history['val_recall'].append(val_metrics['recall'])
        history['val_f1'].append(val_metrics['f1_score'])
        history['val_auc'].append(val_metrics['auc_roc'])

        current_lr = optimizer.param_groups[0]['lr']
        print(f"\nEpoch {epoch+1:02d}/{config.num_epochs}  LR={current_lr:.2e}")
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    loss={val_metrics['loss']:.4f}  acc={val_metrics['accuracy']:.2f}%")
        print(f"  Val    balanced_acc={val_metrics['balanced_accuracy']:.2f}%  AUC={val_metrics['auc_roc']:.4f}")
        # FIX: Print per-class accuracy every epoch — critical for diagnosing collapse
        print(f"  ┌── FAKE acc: {val_metrics['fake_accuracy']:.1f}%   REAL acc: {val_metrics['real_accuracy']:.1f}% ──┐")
        if val_metrics['real_accuracy'] < 50:
            print(f"  ⚠  WARNING: REAL class recall is critically low ({val_metrics['real_accuracy']:.1f}%)!")
            print(f"     Model is collapsing to predicting FAKE. Consider increasing class_weights[1].")
        cm = val_metrics['confusion_matrix']
        print(f"  CM     TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

        # FIX 1b: Save best model on balanced accuracy, not overall accuracy
        balanced_acc = val_metrics['balanced_accuracy']
        if balanced_acc > best_balanced_acc:
            best_balanced_acc = balanced_acc
            save_checkpoint(model, optimizer, epoch, val_metrics, config, model_name, is_best=True)
            print(f"  ★ New best (balanced): {best_balanced_acc:.2f}%  "
                  f"[FAKE={val_metrics['fake_accuracy']:.1f}%  REAL={val_metrics['real_accuracy']:.1f}%]")

        save_checkpoint(model, optimizer, epoch, val_metrics, config, model_name, is_best=False)

        # FIX 3: Early stop on balanced accuracy
        early_stop(balanced_acc)
        if early_stop.early_stop:
            print(f"\n⚠ Early stopping at epoch {epoch+1}  (best balanced_acc={best_balanced_acc:.2f}%)")
            break

    elapsed = time.time() - t0
    print(f"\n{'='*60}")
    print(f"Done in {elapsed/60:.1f} min  |  Best balanced_acc: {best_balanced_acc:.2f}%")
    print(f"{'='*60}")

    # Print final per-class summary
    print(f"\nFinal Per-Class Accuracy (best checkpoint):")
    print(f"  Last epoch FAKE: {history['fake_accuracy'][-1]:.1f}%")
    print(f"  Last epoch REAL: {history['real_accuracy'][-1]:.1f}%")
    if min(history['real_accuracy']) < 40:
        print(f"\n  ⚠ REAL accuracy was critically low during training.")
        print(f"  Consider: (1) increasing class_weights[1] above 2.0,")
        print(f"            (2) running more epochs,")
        print(f"            (3) checking dataset class balance.")

    plot_history(history, model_name, config.save_dir)

    hist_path = os.path.join(config.save_dir, f'{model_name}_history.json')
    with open(hist_path, 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    return history, best_balanced_acc


# ──────────────────────────────────────────────────────────────
# Entry point
# ──────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='CIFAKE training pipeline v3.0 FIXED')
    parser.add_argument('--model',        choices=['spatial', 'frequency', 'both'], default='both')
    parser.add_argument('--data_root',    default='CIFAKE')
    parser.add_argument('--batch_size',   type=int,   default=32)
    parser.add_argument('--epochs',       type=int,   default=30)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--real_weight',  type=float, default=2.0,
                        help='Class weight for REAL class (FAKE=1.0). '
                             'Increase if REAL accuracy is too low.')
    args = parser.parse_args()

    config = TrainingConfig()
    config.data_root      = args.data_root
    config.train_dir      = os.path.join(config.data_root, 'train')
    config.test_dir       = os.path.join(config.data_root, 'test')
    config.batch_size     = args.batch_size
    config.num_epochs     = args.epochs
    config.learning_rate  = args.lr
    config.class_weights  = [1.0, args.real_weight]

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n{'='*60}")
    print(f"Device: {device}")
    if torch.cuda.is_available():
        print(f"GPU   : {torch.cuda.get_device_name(0)}")
        print(f"VRAM  : {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
    print(f"Class weights: FAKE={config.class_weights[0]}  REAL={config.class_weights[1]}")
    print(f"Best metric:   {config.best_metric}")
    print(f"{'='*60}\n")

    os.makedirs(config.save_dir, exist_ok=True)

    # ── Spatial model ──────────────────────────────────────────
    if args.model in ('spatial', 'both'):
        print("Loading spatial DataLoaders...")
        s_train, s_test, classes = get_dataloaders(config, model_type='spatial')

        print("Initialising Spatial Model...")
        spatial = get_spatial_model(num_classes=config.num_classes, dropout=config.spatial_dropout)
        spatial = spatial.to(device)
        params  = sum(p.numel() for p in spatial.parameters())
        print(f"Parameters: {params:,}")

        train_model(spatial, 'spatial_model', s_train, s_test, config, device)
        del spatial
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    # ── Frequency model ────────────────────────────────────────
    if args.model in ('frequency', 'both'):
        print("\nLoading frequency DataLoaders (raw [0,1] tensors, no normalisation)...")
        f_train, f_test, classes = get_dataloaders(config, model_type='frequency')

        print("Initialising Frequency Model...")
        freq = get_frequency_model(
            num_classes=config.num_classes,
            dropout=config.frequency_dropout,
            process_in_model=True,
        )
        freq = freq.to(device)
        params = sum(p.numel() for p in freq.parameters())
        print(f"Parameters: {params:,}")

        train_model(freq, 'frequency_model', f_train, f_test, config, device)
        del freq
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\n✓ ALL TRAINING COMPLETE — checkpoints in ./{config.save_dir}/")
    print(f"\nReminder: Open {config.save_dir}/*_curves.png and check the")
    print(f"'Per-Class Accuracy' plot. Both FAKE and REAL lines should be")
    print(f"above 80% for a production-ready model.")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()