import os
import io
import time
import json
import argparse
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)

from v5spatial import get_spatial_model_v5, CLIP_MEAN, CLIP_STD
from v5_1frequency import get_frequency_model_v5

warnings.filterwarnings('ignore', category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Augmentations ──────────────────────────────────────────────────────────────

class JpegCompression:
    def __init__(self, quality_range=(50, 95)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        q = np.random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).copy()


class RandomDownscaleUpscale:
    def __init__(self, p=0.3, min_scale=0.5):
        self.p = p
        self.min_scale = min_scale

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            w, h = img.size
            s = np.random.uniform(self.min_scale, 0.85)
            small = img.resize((max(1, int(w * s)), max(1, int(h * s))), Image.BILINEAR)
            return small.resize((w, h), Image.BILINEAR)
        return img


class AddGaussianNoise:
    def __init__(self, p=0.2, sigma_range=(1, 8)):
        self.p = p
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        if np.random.rand() < self.p:
            arr = np.array(img).astype(np.float32)
            sigma = np.random.uniform(*self.sigma_range)
            arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
            return Image.fromarray(arr)
        return img


class RandomGamma:
    def __call__(self, img: Image.Image) -> Image.Image:
        gamma = np.random.uniform(0.8, 1.3)
        arr = np.power(np.array(img).astype(np.float32) / 255.0, gamma)
        return Image.fromarray((arr * 255).clip(0, 255).astype(np.uint8))


# ── Transforms ────────────────────────────────────────────────────────────────

def get_spatial_train_transform(image_size=224, strong=True):
    ops = [
        T.Resize(int(image_size * 1.15)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if strong:
        ops += [
            # Moderate JPEG — don't destroy spatial features
            T.RandomApply([T.Lambda(JpegCompression((65, 95)))], p=0.35),
            T.RandomApply([T.Lambda(RandomDownscaleUpscale(p=1.0, min_scale=0.6))], p=0.25),
            T.RandomApply([T.Lambda(AddGaussianNoise(p=1.0))], p=0.2),
            T.RandomApply([T.Lambda(RandomGamma())], p=0.25),
        ]
    ops += [
        T.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.15, hue=0.04),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ]
    return T.Compose(ops)


def get_spatial_val_transform(image_size=224):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


def get_frequency_train_transform(image_size=128, strong=True):
    """
    CRITICAL: frequency model needs raw [0,1] pixels.
    Augmentations must be mild enough to preserve frequency artifacts.
    Heavy JPEG (Q<65) destroys the very artifacts we want to detect.
    """
    ops = [
        T.Resize(int(image_size * 1.12)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
    ]
    if strong:
        ops += [
            # Mild JPEG only — preserves sub-pixel artifacts
            T.RandomApply([T.Lambda(JpegCompression((75, 95)))], p=0.3),
            T.RandomApply([T.Lambda(AddGaussianNoise(p=1.0, sigma_range=(1, 5)))], p=0.2),
            # No downscale-upscale for frequency — it mimics AI upsampling patterns
        ]
    ops.append(T.ToTensor())  # [0,1], no normalization
    return T.Compose(ops)


def get_frequency_val_transform(image_size=128):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiSourceDataset(Dataset):
    def __init__(self, primary_root, transform, extra_roots=None, split='train'):
        self.transform = transform
        self.samples = []

        primary_split = os.path.join(primary_root, split)
        if os.path.exists(primary_split):
            for label_name, label_idx in [('FAKE', 0), ('REAL', 1)]:
                label_dir = os.path.join(primary_split, label_name)
                if os.path.exists(label_dir):
                    import glob
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                        for p in glob.glob(os.path.join(label_dir, ext)):
                            self.samples.append((p, label_idx))
        else:
            for label_idx, class_name in enumerate(sorted(os.listdir(primary_root))):
                class_dir = os.path.join(primary_root, class_name)
                if os.path.isdir(class_dir):
                    import glob
                    for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                        for p in glob.glob(os.path.join(class_dir, ext)):
                            self.samples.append((p, label_idx))

        if extra_roots:
            for er in extra_roots:
                self._add_extra(er)

        fake_count = sum(1 for _, l in self.samples if l == 0)
        real_count = len(self.samples) - fake_count
        print(f"  [{split}] Total: {len(self.samples)} | FAKE: {fake_count} | REAL: {real_count}")

    def _add_extra(self, root):
        import glob
        for label_name, label_idx in [('fake', 0), ('real', 1), ('FAKE', 0), ('REAL', 1)]:
            label_dir = os.path.join(root, label_name)
            if os.path.exists(label_dir):
                for ext in ['*.jpg', '*.jpeg', '*.png']:
                    for p in glob.glob(os.path.join(label_dir, ext)):
                        self.samples.append((p, label_idx))

    def get_class_weights(self):
        labels = [l for _, l in self.samples]
        counts = [labels.count(0), labels.count(1)]
        total = sum(counts)
        weights = [total / (2 * c) if c > 0 else 1.0 for c in counts]
        return torch.FloatTensor([weights[l] for _, l in self.samples])

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
            return self.transform(img), label
        except Exception:
            return torch.zeros(3, 128, 128), label


# ── Loss ──────────────────────────────────────────────────────────────────────

class BalancedFocalLoss(nn.Module):
    """
    Focal loss with per-class weights.
    Focal term reduces overconfident easy samples — prevents collapse.
    gamma=1.5 is mild enough to not destabilize training.
    """
    def __init__(self, class_weights=None, gamma=1.5, label_smoothing=0.0):
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits, labels):
        # FIX: cast to float32 before loss computation to prevent AMP fp16
        # overflow from propagating NaN when logits are already large.
        logits = logits.float()

        if self.label_smoothing > 0:
            n_cls = logits.size(1)
            smooth_labels = torch.full_like(logits, self.label_smoothing / n_cls)
            smooth_labels.scatter_(1, labels.unsqueeze(1), 1.0 - self.label_smoothing + self.label_smoothing / n_cls)
            log_probs = F.log_softmax(logits, dim=1)
            base_loss = -(smooth_labels * log_probs).sum(dim=1)
        else:
            base_loss = F.cross_entropy(logits, labels, reduction='none')

        probs = F.softmax(logits, dim=1)
        pt = probs.gather(1, labels.unsqueeze(1)).squeeze(1)
        focal_weight = (1 - pt) ** self.gamma

        loss = focal_weight * base_loss

        if self.class_weights is not None:
            w = self.class_weights[labels]
            loss = loss * w

        return loss.mean()


# ── Training Config ────────────────────────────────────────────────────────────

class TrainingConfigV5:
    data_root: str = 'CIFAKE'
    extra_data: List[str] = None
    image_size_spatial: int = 224
    image_size_frequency: int = 128

    batch_size: int = 8   # RTX 2050 4GB: 8 is safe for freq model; 16 risks OOM
    num_epochs: int = 40
    learning_rate: float = 2e-4
    backbone_lr_multiplier: float = 0.05
    weight_decay: float = 5e-5
    warmup_epochs: int = 5

    # Mild label smoothing — heavy smoothing causes real-bias
    label_smoothing: float = 0.02
    class_weights: List[float] = None

    use_strong_aug: bool = True
    use_adversarial: bool = False
    adversarial_epsilon: float = 0.003
    adversarial_prob: float = 0.15

    use_pretrained: bool = True
    freeze_backbone: bool = True
    unfreeze_last_n: int = 4
    spatial_dropout: float = 0.3
    frequency_dropout: float = 0.35

    calibrate_after_training: bool = True
    save_dir: str = 'checkpoints_v5'
    early_stop_patience: int = 12
    num_classes: int = 2
    no_resume: bool = False   # False = auto-resume (correct default)


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(all_preds, all_labels, all_probs) -> Dict:
    cm = confusion_matrix(all_labels, all_preds)
    fake_acc = real_acc = 0.0
    if cm.shape == (2, 2):
        fake_acc = 100 * cm[0, 0] / (cm[0, 0] + cm[0, 1] + 1e-9)
        real_acc = 100 * cm[1, 1] / (cm[1, 0] + cm[1, 1] + 1e-9)
    try:
        auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
    except Exception:
        auc = 0.5
    return {
        'accuracy': 100 * accuracy_score(all_labels, all_preds),
        'balanced_accuracy': 100 * balanced_accuracy_score(all_labels, all_preds),
        'fake_accuracy': fake_acc,
        'real_accuracy': real_acc,
        'f1_score': f1_score(all_labels, all_preds, zero_division=0),
        'auc_roc': auc,
        'confusion_matrix': cm,
    }


# ── Train / Validate ──────────────────────────────────────────────────────────

def fgsm_attack(model, images, labels, epsilon=0.003):
    images = images.clone().detach().requires_grad_(True)
    loss = F.cross_entropy(model(images), labels)
    loss.backward()
    with torch.no_grad():
        adv = images + epsilon * images.grad.sign()
        adv = torch.clamp(adv, -3.0, 3.0)
    return adv.detach()


def mixup_data(images, labels, alpha=0.2, device='cpu'):
    """Mixup augmentation — blends two random samples and their labels."""
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = np.random.beta(alpha, alpha)
    batch_size = images.size(0)
    idx = torch.randperm(batch_size, device=device)
    mixed = lam * images + (1 - lam) * images[idx]
    labels_a, labels_b = labels, labels[idx]
    return mixed, labels_a, labels_b, lam


def mixup_criterion(criterion, logits, labels_a, labels_b, lam):
    return lam * criterion(logits, labels_a) + (1 - lam) * criterion(logits, labels_b)


def train_epoch(model, loader, criterion, optimizer, scaler, device, config, epoch,
                base_lrs, use_mixup=False):
    """
    Hardened training epoch:
    - Warmup LR scaling from base_lrs (not scheduler-decayed LR).
    - Optional Mixup (use_mixup=True for frequency model).
    - NaN/Inf loss detection with batch skip.
    - AMP GradScaler with consecutive-NaN circuit breaker:
      if the scaler detects >10 consecutive skipped steps it resets
      the scale factor to avoid an irrecoverable stuck-at-zero scale.
    """
    model.train()

    # Apply warmup scaling at the start of the epoch.
    if epoch < config.warmup_epochs:
        lr_scale = (epoch + 1) / config.warmup_epochs
        for pg, base_lr in zip(optimizer.param_groups, base_lrs):
            pg['lr'] = base_lr * lr_scale

    total_loss = 0.0
    correct = 0
    total = 0
    consecutive_nan = 0   # AMP circuit-breaker counter
    pbar = tqdm(loader, desc=f'Train Epoch {epoch+1}', leave=False)

    for batch_idx, (images, labels) in enumerate(pbar):
        if batch_idx < 2:
            continue
        images, labels = images.to(device), labels.to(device)

        if config.use_adversarial and np.random.rand() < config.adversarial_prob:
            model.eval()
            with torch.enable_grad():
                adv = fgsm_attack(model, images, labels, config.adversarial_epsilon)
            model.train()
            mask = torch.rand(len(images), device=device) < 0.5
            images[mask] = adv[mask]

        # Mixup (frequency model only — spatial uses strong aug instead)
        if use_mixup and np.random.rand() < 0.5:
            images, labels_a, labels_b, lam = mixup_data(images, labels, alpha=0.2, device=device)
        else:
            labels_a, labels_b, lam = labels, labels, 1.0

        optimizer.zero_grad()

        with torch.amp.autocast(device_type=device.type, enabled=(device.type == 'cuda')):
        #with torch.amp.autocast(device_type=device.type, enabled=False):
            logits = model(images)
            if lam < 1.0:
                loss = mixup_criterion(criterion, logits, labels_a, labels_b, lam)
            else:
                loss = criterion(logits, labels)

        # --- NaN/Inf guard ---
        if not torch.isfinite(loss):
            consecutive_nan += 1
            print(f"  [WARNING] Non-finite loss at batch {batch_idx}, skipping.")
            optimizer.zero_grad()
            # Circuit breaker: if scaler is stuck, reset its scale to recover.
            if scaler is not None and consecutive_nan > 10:
                if hasattr(scaler, "_scale") and scaler._scale is not None:
                    scaler._scale.fill_(256.0)
                consecutive_nan = 0
                print("  [WARNING] AMP scaler reset after 10 consecutive NaN batches.")
            continue
        consecutive_nan = 0   # reset on good batch
            

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

        preds = logits.argmax(dim=1)
        # Use labels (not labels_a) for accuracy tracking to stay interpretable.
        correct += (preds == labels).sum().item()
        total += len(labels)
        total_loss += loss.item()
        pbar.set_postfix({'loss': f'{total_loss/(batch_idx+1):.4f}', 'acc': f'{100*correct/max(total,1):.1f}%'})

    return total_loss / max(len(loader), 1), 100 * correct / max(total, 1)


@torch.no_grad()
def validate(model, loader, criterion, device) -> Dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []
    for images, labels in loader:
        images, labels = images.to(device), labels.to(device)
        logits = model(images)
        total_loss += criterion(logits, labels).item()
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(dim=1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())
    metrics = compute_metrics(all_preds, all_labels, all_probs)
    metrics['loss'] = total_loss / len(loader)
    return metrics


# ── Checkpoints ───────────────────────────────────────────────────────────────

def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics,
                    config, name, is_best, history, best_balanced_acc, patience_counter):
    os.makedirs(config.save_dir, exist_ok=True)
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': metrics,
        'history': history,
        'best_balanced_acc': best_balanced_acc,
        'patience_counter': patience_counter,
        'config': vars(config) if hasattr(config, '__dict__') else {},
    }
    torch.save(payload, os.path.join(config.save_dir, f'{name}_resume.pth'))
    if is_best:
        best_path = os.path.join(config.save_dir, f'{name}_best.pth')
        torch.save(payload, best_path)
        print(f"  Saved best checkpoint: {best_path}")


def load_resume_checkpoint(model, optimizer, scheduler, scaler, config, name, device):
    fresh = {k: [] for k in ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                               'balanced_accuracy', 'fake_accuracy', 'real_accuracy',
                               'val_f1', 'val_auc']}
    if getattr(config, 'no_resume', False):
        return 0, fresh, 0.0, 0

    save_dir_abs = os.path.abspath(config.save_dir)
    resume_path = os.path.join(save_dir_abs, f'{name}_resume.pth')
    best_path = os.path.join(save_dir_abs, f'{name}_best.pth')

    load_path = resume_path if os.path.exists(resume_path) else (best_path if os.path.exists(best_path) else None)
    if not load_path:
        return 0, fresh, 0.0, 0

    ckpt = torch.load(load_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    # CRITICAL FIX: After loading optimizer state the per-group LRs reflect the
    # *last saved* (possibly cosine-decayed) LR.  We restore them to the
    # configured learning_rate so that warmup and backbone multiplier both
    # compute from the intended base, not a stale near-zero value.
    lr = config.learning_rate
    blr_mult = getattr(config, 'backbone_lr_multiplier', 1.0)
    backbone_keys = ('extractor', 'freq_decomposer', 'artifact_detector', 'backbone')
    for i, pg in enumerate(optimizer.param_groups):
        # Heuristic: if the first param name in this group looks like backbone, use multiplier.
        # param_groups don't store names, so we use position: group 0 = head, group 1 = backbone.
        pg['lr'] = lr if i == 0 else lr * blr_mult

    # Also reset GradScaler to a healthy scale if it had collapsed to near zero.
    if scaler is not None:
        current_scale = scaler.get_scale()
        if current_scale < 1.0:
            scaler._scale.fill_(256.0)
            print(f"  [Resume] GradScaler scale was {current_scale:.2f}, reset to 256.0")

    start_epoch = ckpt['epoch'] + 1
    history = ckpt.get('history', fresh)
    best_balanced_acc = ckpt.get('best_balanced_acc', 0.0)
    patience_counter = ckpt.get('patience_counter', 0)
    print(f"  Resuming epoch {start_epoch+1} | Best: {best_balanced_acc:.2f}% | Patience: {patience_counter}")
    print(f"  LR restored → head={lr:.2e}  backbone={lr*blr_mult:.2e}")
    return start_epoch, history, best_balanced_acc, patience_counter


# ── Main Training Loop ─────────────────────────────────────────────────────────

def train_model_v5(model, model_name, train_loader, val_loader, config, device, use_frequency=False):
    # Class weights — moderate real boost, not extreme
    class_weights = None
    if config.class_weights:
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)

    criterion = BalancedFocalLoss(
        class_weights=class_weights,
        gamma=1.5,
        label_smoothing=config.label_smoothing,
    )

    # Param groups with differential LR
    backbone_params, head_params = [], []
    backbone_keys = ('extractor', 'freq_decomposer', 'artifact_detector', 'backbone')
    for pname, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if any(k in pname.lower() for k in backbone_keys):
            backbone_params.append(param)
        else:
            head_params.append(param)

    if not backbone_params:
        param_groups = [{'params': head_params, 'lr': config.learning_rate}]
    else:
        param_groups = [
            {'params': head_params, 'lr': config.learning_rate},
            {'params': backbone_params, 'lr': config.learning_rate * config.backbone_lr_multiplier},
        ]

    optimizer = optim.AdamW(param_groups, weight_decay=config.weight_decay)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=max(1, config.num_epochs - config.warmup_epochs), eta_min=1e-6
    )
    scaler = torch.amp.GradScaler(device='cuda') if device.type == 'cuda' else None

    print(f"\n{'='*60}\nTraining {model_name}")
    start_epoch, history, best_balanced_acc, patience_counter = load_resume_checkpoint(
        model, optimizer, scheduler, scaler, config, model_name, device
    )
    print(f"{'='*60}")

    if start_epoch >= config.num_epochs:
        print(f"  Already trained {config.num_epochs} epochs.")
        return history, best_balanced_acc

    # FIX: capture the *post-resume* base LRs once so warmup scaling always
    # refers to the intended target LR, not the scheduler-decayed value.
    # If we are past warmup on resume, these are the fully-warmed-up LRs.
    base_lrs = [pg['lr'] for pg in optimizer.param_groups]

    for epoch in range(start_epoch, config.num_epochs):
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, epoch,
            base_lrs,use_mixup=False   

        )
        val_metrics = validate(model, val_loader, criterion, device)

        # Only step the cosine scheduler after warmup is complete.
        if epoch >= config.warmup_epochs:
            scheduler.step()

        for k, v in [('train_loss', train_loss), ('train_acc', train_acc),
                     ('val_loss', val_metrics['loss']), ('val_acc', val_metrics['accuracy']),
                     ('balanced_accuracy', val_metrics['balanced_accuracy']),
                     ('fake_accuracy', val_metrics['fake_accuracy']),
                     ('real_accuracy', val_metrics['real_accuracy']),
                     ('val_f1', val_metrics['f1_score']), ('val_auc', val_metrics['auc_roc'])]:
            history[k].append(v)

        cur_lr = optimizer.param_groups[0]['lr']
        cm = val_metrics['confusion_matrix']
        print(f"\nEpoch {epoch+1:02d}/{config.num_epochs}  LR={cur_lr:.2e}")
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    loss={val_metrics['loss']:.4f}  balanced={val_metrics['balanced_accuracy']:.2f}%  AUC={val_metrics['auc_roc']:.4f}")
        print(f"  FAKE={val_metrics['fake_accuracy']:.1f}%  REAL={val_metrics['real_accuracy']:.1f}%  F1={val_metrics['f1_score']:.4f}")
        if cm.shape == (2, 2):
            print(f"  CM: TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

        # Collapse detection
        if val_metrics['fake_accuracy'] < 20 or val_metrics['real_accuracy'] < 20:
            print(f"  WARNING: Model collapsing to one class!")

        bal = val_metrics['balanced_accuracy']
        is_best = bal > best_balanced_acc
        if is_best:
            best_balanced_acc = bal
            patience_counter = 0
        else:
            patience_counter += 1

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics,
                        config, model_name, is_best, history, best_balanced_acc, patience_counter)

        """if patience_counter >= config.early_stop_patience:
            print(f"\nEarly stopping at epoch {epoch+1}")
            break"""

    print(f"\n{'='*60}\nBest balanced accuracy: {best_balanced_acc:.2f}%\n{'='*60}")

    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, f'{model_name}_history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    return history, best_balanced_acc


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders_v5(config, model_type='spatial'):
    is_spatial = model_type == 'spatial'
    train_tf = (get_spatial_train_transform(config.image_size_spatial, config.use_strong_aug)
                if is_spatial else
                get_frequency_train_transform(image_size=128))
    val_tf = (get_spatial_val_transform(config.image_size_spatial)
              if is_spatial else
            get_frequency_val_transform(image_size=128))

    train_ds = MultiSourceDataset(config.data_root, train_tf, config.extra_data, 'train')
    val_ds = MultiSourceDataset(config.data_root, val_tf, None, 'test')

    sample_weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    num_workers = min(4, os.cpu_count() or 1)
    train_loader = DataLoader(train_ds, batch_size=config.batch_size, sampler=sampler,
                              num_workers=num_workers, pin_memory=True, drop_last=True)
    val_loader = DataLoader(val_ds, batch_size=config.batch_size * 2, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader


# ── Entry Point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['spatial', 'frequency', 'both'], default='both')
    parser.add_argument('--data_root', default='CIFAKE')
    parser.add_argument('--extra_data', nargs='*', default=None)
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--real_weight', type=float, default=1.5,
                        help='Class weight for REAL — keep moderate (1.2-2.0)')
    parser.add_argument('--use_strong_aug', action='store_true', default=True)
    parser.add_argument('--use_adversarial', action='store_true', default=False)
    parser.add_argument('--calibrate', action='store_true', default=True)
    parser.add_argument('--freeze_backbone', action='store_true', default=True)
    parser.add_argument('--unfreeze_last_n', type=int, default=4)
    parser.add_argument('--save_dir', default='checkpoints_v5')
    parser.add_argument('--no_resume', action='store_true', default=False)
    args = parser.parse_args()

    config = TrainingConfigV5()
    config.data_root = args.data_root
    config.extra_data = args.extra_data
    config.batch_size = args.batch_size
    config.num_epochs = args.epochs
    config.learning_rate = args.lr
    config.class_weights = [1.0, args.real_weight]
    config.use_strong_aug = args.use_strong_aug
    config.use_adversarial = args.use_adversarial
    config.calibrate_after_training = args.calibrate
    config.freeze_backbone = args.freeze_backbone
    config.unfreeze_last_n = args.unfreeze_last_n
    config.save_dir = os.path.abspath(args.save_dir)
    config.no_resume = args.no_resume

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"AI Image Detection Training v5.1")
    print(f"Device: {device} | Save: {config.save_dir}")
    print(f"Resume: {'DISABLED' if args.no_resume else 'AUTO'}")
    print(f"Mixup: ON  alpha=0.2")
    print(f"{'='*60}\n")

    if args.model in ('spatial', 'both'):
        print("SPATIAL MODEL")
        train_loader, val_loader = get_dataloaders_v5(config, 'spatial')
        spatial_model = get_spatial_model_v5(
            num_classes=config.num_classes,
            dropout=config.spatial_dropout,
            freeze_backbone=config.freeze_backbone,
            unfreeze_last_n=config.unfreeze_last_n,
            input_size=config.image_size_spatial,
        ).to(device)
        _, _ = train_model_v5(spatial_model, 'spatial_model_v5', train_loader, val_loader, config, device)

        if config.calibrate_after_training:
            best_path = os.path.join(config.save_dir, 'spatial_model_v5_best.pth')
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                spatial_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            spatial_model.calibrate(val_loader, device)
            torch.save({'model_state_dict': spatial_model.state_dict()},
                       os.path.join(config.save_dir, 'spatial_model_v5_calibrated.pth'))

        del spatial_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    if args.model in ('frequency', 'both'):
        print("FREQUENCY MODEL")
        train_loader, val_loader = get_dataloaders_v5(config, 'frequency')
        freq_model = get_frequency_model_v5(
            num_classes=config.num_classes,
            image_size=config.image_size_frequency,
            dropout=config.frequency_dropout,
        ).to(device)
        _, _ = train_model_v5(freq_model, 'frequency_model_v5', train_loader, val_loader, config, device, use_frequency=True)

        if config.calibrate_after_training:
            best_path = os.path.join(config.save_dir, 'frequency_model_v5_best.pth')
            if os.path.exists(best_path):
                ckpt = torch.load(best_path, map_location=device, weights_only=False)
                freq_model.load_state_dict(ckpt['model_state_dict'], strict=False)
            freq_model.calibrate(val_loader, device)
            torch.save({'model_state_dict': freq_model.state_dict()},
                       os.path.join(config.save_dir, 'frequency_model_v5_calibrated.pth'))

        del freq_model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print(f"\nTraining complete. Checkpoints: {config.save_dir}/")


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()