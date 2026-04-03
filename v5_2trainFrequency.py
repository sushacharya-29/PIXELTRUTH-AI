import os
import io
import time
import json
import argparse
import warnings
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
import torchvision.transforms as T
from PIL import Image
import platform, sys

from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score, f1_score,
    roc_auc_score, confusion_matrix
)

from v5_2frequency import get_frequency_model_v5

warnings.filterwarnings('ignore', category=UserWarning)
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"


# ── Augmentations ──────────────────────────────────────────────────────────────
# RULE: frequency model must see raw [0,1] pixels.
# Augmentations that destroy sub-pixel forensic artifacts are excluded.

class JpegCompression:
    def __init__(self, quality_range=(75, 95)):
        self.quality_range = quality_range

    def __call__(self, img: Image.Image) -> Image.Image:
        q = np.random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).copy()


class AddGaussianNoise:
    def __init__(self, sigma_range=(1, 5)):
        self.sigma_range = sigma_range

    def __call__(self, img: Image.Image) -> Image.Image:
        arr = np.array(img).astype(np.float32)
        sigma = np.random.uniform(*self.sigma_range)
        arr = np.clip(arr + np.random.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(arr)


def get_freq_train_transform(image_size: int = 128):
    return T.Compose([
        T.Resize(int(image_size * 1.12)),
        T.RandomCrop(image_size),
        T.RandomHorizontalFlip(p=0.5),
        T.RandomVerticalFlip(p=0.1),
        T.RandomApply([T.Lambda(JpegCompression((75, 95)))], p=0.3),
        T.RandomApply([T.Lambda(AddGaussianNoise((1, 4)))], p=0.2),
        # No ColorJitter — color shifts corrupt frequency signatures
        T.ToTensor(),   # [0, 1], no normalisation
    ])


def get_freq_val_transform(image_size: int = 128):
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
    ])


# ── Dataset ───────────────────────────────────────────────────────────────────

class MultiSourceDataset(Dataset):
    def __init__(self, primary_root: str, transform, extra_roots=None, split: str = 'train'):
        import glob
        self.transform = transform
        self.samples: List = []

        primary_split = os.path.join(primary_root, split)
        root = primary_split if os.path.exists(primary_split) else primary_root

        for label_name, label_idx in [('FAKE', 0), ('REAL', 1), ('fake', 0), ('real', 1)]:
            label_dir = os.path.join(root, label_name)
            if os.path.exists(label_dir):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                    for p in glob.glob(os.path.join(label_dir, '**', ext), recursive=True):
                        self.samples.append((p, label_idx))

        if extra_roots:
            for er in extra_roots:
                self._add_extra(er)

        # Deduplicate
        self.samples = list(dict.fromkeys(self.samples))

        fake_count = sum(1 for _, l in self.samples if l == 0)
        real_count = len(self.samples) - fake_count
        print(f"  [{split}] Total: {len(self.samples)} | FAKE: {fake_count} | REAL: {real_count}")

    def _add_extra(self, root: str):
        import glob
        for label_name, label_idx in [('fake', 0), ('real', 1), ('FAKE', 0), ('REAL', 1)]:
            label_dir = os.path.join(root, label_name)
            if os.path.exists(label_dir):
                for ext in ['*.jpg', '*.jpeg', '*.png', '*.webp']:
                    for p in glob.glob(os.path.join(label_dir, '**', ext), recursive=True):
                        self.samples.append((p, label_idx))

    def get_class_weights(self) -> torch.FloatTensor:
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
            # Return a plausible fallback — uniform noise stays in [0,1]
            return torch.rand(3, 128, 128), label


# ── Loss ──────────────────────────────────────────────────────────────────────
# Symmetric focal loss with per-class weights.
# gamma=1.0 is milder than 1.5 — prevents over-suppression of easy samples
# which was causing the model to effectively ignore the minority class.

class FocalLoss(nn.Module):
    def __init__(self, class_weights: Optional[torch.Tensor] = None,
                 gamma: float = 1.0, label_smoothing: float = 0.05):
        super().__init__()
        self.register_buffer('class_weights', class_weights)
        self.gamma = gamma
        self.label_smoothing = label_smoothing

    def forward(self, logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        logits = logits.float()
        n_cls = logits.size(1)

        # Label smoothing
        smooth = torch.full_like(logits, self.label_smoothing / n_cls)
        smooth.scatter_(1, labels.unsqueeze(1),
                        1.0 - self.label_smoothing + self.label_smoothing / n_cls)
        log_p = F.log_softmax(logits, dim=1)
        base_loss = -(smooth * log_p).sum(dim=1)

        # Focal modulation
        pt = F.softmax(logits, dim=1).gather(1, labels.unsqueeze(1)).squeeze(1).clamp(1e-6, 1.0)
        loss = ((1 - pt) ** self.gamma) * base_loss

        if self.class_weights is not None:
            loss = loss * self.class_weights[labels]

        return loss.mean()


# ── Config ────────────────────────────────────────────────────────────────────

class FreqTrainConfig:
    data_root: str = 'CIFAKE'
    extra_data: Optional[List[str]] = None
    image_size: int = 128
    # RTX 2050 (4 GB): 16 is safe; 32 will OOM with the full pipeline.
    batch_size: int = 16
    num_epochs: int = 60
    learning_rate: float = 3e-4
    # All freq model params share the same group — no backbone freezing.
    weight_decay: float = 1e-4
    warmup_epochs: int = 3
    label_smoothing: float = 0.05
    # [FAKE_weight, REAL_weight] — slight real boost to fight collapse toward FAKE
    class_weights: List[float] = None   # set per run
    save_dir: str = 'checkpoints_freq'
    no_resume: bool = False
    num_classes: int = 2
    dropout: float = 0.35
    early_stop_patience: int = 15
    # Gradient clip — tighter than spatial because freq signals are noisier
    grad_clip: float = 0.5
    num_workers: int = 4


# ── Metrics ───────────────────────────────────────────────────────────────────

def compute_metrics(preds, labels, probs) -> Dict:
    cm = confusion_matrix(labels, preds)
    fake_acc = real_acc = 0.0
    if cm.shape == (2, 2):
        fake_acc = 100 * cm[0, 0] / max(cm[0, 0] + cm[0, 1], 1)
        real_acc = 100 * cm[1, 1] / max(cm[1, 0] + cm[1, 1], 1)
    try:
        auc = roc_auc_score(labels, np.array(probs)[:, 1])
    except Exception:
        auc = 0.5
    return {
        'accuracy':          100 * accuracy_score(labels, preds),
        'balanced_accuracy': 100 * balanced_accuracy_score(labels, preds),
        'fake_accuracy':     fake_acc,
        'real_accuracy':     real_acc,
        'f1_score':          f1_score(labels, preds, zero_division=0),
        'auc_roc':           auc,
        'confusion_matrix':  cm,
    }


# ── Checkpoint I/O ────────────────────────────────────────────────────────────

_HISTORY_KEYS = ['train_loss', 'train_acc', 'val_loss', 'val_acc',
                 'balanced_accuracy', 'fake_accuracy', 'real_accuracy',
                 'val_f1', 'val_auc']


def _fresh_history():
    return {k: [] for k in _HISTORY_KEYS}


def save_checkpoint(model, optimizer, scheduler, scaler, epoch, metrics,
                    config: FreqTrainConfig, is_best: bool,
                    history: Dict, best_acc: float, patience: int):
    os.makedirs(config.save_dir, exist_ok=True)
    payload = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'scheduler_state_dict': scheduler.state_dict(),
        'scaler_state_dict': scaler.state_dict() if scaler else None,
        'metrics': {k: v for k, v in metrics.items() if k != 'confusion_matrix'},
        'history': history,
        'best_balanced_acc': best_acc,
        'patience_counter': patience,
    }
    torch.save(payload, os.path.join(config.save_dir, 'freq_model_resume.pth'))
    if is_best:
        best_path = os.path.join(config.save_dir, 'freq_model_best.pth')
        torch.save(payload, best_path)
        print(f"  ✓ Best checkpoint saved → {best_path}")


def load_checkpoint(model, optimizer, scheduler, scaler,
                    config: FreqTrainConfig, device: torch.device):
    if config.no_resume:
        return 0, _fresh_history(), 0.0, 0

    resume = os.path.join(config.save_dir, 'freq_model_resume.pth')
    best   = os.path.join(config.save_dir, 'freq_model_best.pth')
    path   = resume if os.path.exists(resume) else (best if os.path.exists(best) else None)
    if not path:
        return 0, _fresh_history(), 0.0, 0

    ckpt = torch.load(path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt['model_state_dict'], strict=False)
    optimizer.load_state_dict(ckpt['optimizer_state_dict'])
    if 'scheduler_state_dict' in ckpt:
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
    if scaler and ckpt.get('scaler_state_dict'):
        scaler.load_state_dict(ckpt['scaler_state_dict'])

    # Restore base LR — prevents resuming at a near-zero cosine-decayed LR
    for pg in optimizer.param_groups:
        pg['lr'] = config.learning_rate

    if scaler is not None and scaler.get_scale() < 1.0:
        scaler._scale.fill_(256.0)

    start  = ckpt['epoch'] + 1
    hist   = ckpt.get('history', _fresh_history())
    b_acc  = ckpt.get('best_balanced_acc', 0.0)
    pat    = ckpt.get('patience_counter', 0)
    print(f"  Resuming epoch {start + 1} | Best: {b_acc:.2f}% | Patience: {pat}")
    print(f"  LR restored → {config.learning_rate:.2e}")
    return start, hist, b_acc, pat


# ── Mixup ─────────────────────────────────────────────────────────────────────
# Mixup is used here at low alpha to prevent the model memorising
# dataset-specific texture cues instead of learning frequency forensics.

def mixup_data(images, labels, alpha=0.15, device='cpu'):
    if alpha <= 0:
        return images, labels, labels, 1.0
    lam = float(np.random.beta(alpha, alpha))
    idx = torch.randperm(images.size(0), device=device)
    return lam * images + (1 - lam) * images[idx], labels, labels[idx], lam


# ── Train epoch ───────────────────────────────────────────────────────────────

def train_epoch(model, loader, criterion, optimizer, scaler,
                device, config: FreqTrainConfig, epoch: int, base_lr: float) -> tuple:
    model.train()

    # Linear warmup — scale from base_lr/10 to base_lr over warmup_epochs
    if epoch < config.warmup_epochs:
        lr_scale = 0.1 + 0.9 * (epoch + 1) / config.warmup_epochs
        for pg in optimizer.param_groups:
            pg['lr'] = base_lr * lr_scale

    total_loss = correct = total = 0
    consecutive_nan = 0
    pbar = tqdm(loader, desc=f'Epoch {epoch + 1}', leave=False,
                dynamic_ncols=True)

    for images, labels in pbar:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # Mixup only 40 % of batches — keeps some clean signal
        if np.random.rand() < 0.4:
            images, la, lb, lam = mixup_data(images, labels, alpha=0.15, device=device)
        else:
            la, lb, lam = labels, labels, 1.0

        optimizer.zero_grad(set_to_none=True)

        with torch.amp.autocast(device_type='cuda', enabled=(scaler is not None)):
            logits = model(images)
            if lam < 1.0:
                loss = lam * criterion(logits, la) + (1 - lam) * criterion(logits, lb)
            else:
                loss = criterion(logits, labels)

        if not torch.isfinite(loss):
            consecutive_nan += 1
            optimizer.zero_grad(set_to_none=True)
            if consecutive_nan > 8 and scaler is not None:
                scaler._scale.fill_(256.0)
                consecutive_nan = 0
                print(f'\n  [WARN] AMP scale reset (10 consecutive NaN)')
            continue
        consecutive_nan = 0

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), config.grad_clip)
            optimizer.step()

        with torch.no_grad():
            preds = logits.argmax(1)
            correct += (preds == labels).sum().item()
            total   += labels.size(0)
            total_loss += loss.item()

        pbar.set_postfix(loss=f'{total_loss / max(total, 1):.4f}',
                         acc=f'{100 * correct / max(total, 1):.1f}%')

    return total_loss / max(len(loader), 1), 100 * correct / max(total, 1)


# ── Validate ──────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, loader, criterion, device) -> Dict:
    model.eval()
    total_loss = 0.0
    all_preds, all_labels, all_probs = [], [], []

    for images, labels in loader:
        images = images.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        logits = model(images).float()
        total_loss += criterion(logits, labels).item()
        probs = torch.softmax(logits, dim=1)
        all_preds.extend(logits.argmax(1).cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_probs.extend(probs.cpu().numpy())

    metrics = compute_metrics(all_preds, all_labels, all_probs)
    metrics['loss'] = total_loss / max(len(loader), 1)
    return metrics


# ── Anti-collapse callback ────────────────────────────────────────────────────
# If the model collapses (one class dominates ≥85 %), temporarily boost
# the minority-class weight in the loss and reduce LR slightly.

def handle_collapse(metrics: Dict, optimizer, criterion,
                    config: FreqTrainConfig, base_lr: float) -> bool:
    fa, ra = metrics['fake_accuracy'], metrics['real_accuracy']
    collapsed = fa < 15 or ra < 15
    if collapsed:
        minority = 'REAL' if fa > ra else 'FAKE'
        idx = 1 if minority == 'REAL' else 0
        print(f'  [COLLAPSE] {minority} accuracy critically low. Boosting loss weight.')
        if criterion.class_weights is not None:
            with torch.no_grad():
                criterion.class_weights[idx] = min(criterion.class_weights[idx].item() * 1.5, 5.0)
        # Halve LR temporarily to escape bad basin
        for pg in optimizer.param_groups:
            pg['lr'] = max(pg['lr'] * 0.5, 1e-6)
    return collapsed


# ── Main training loop ────────────────────────────────────────────────────────

def train(model, train_loader, val_loader, config: FreqTrainConfig, device: torch.device):
    class_weights = None
    if config.class_weights:
        class_weights = torch.tensor(config.class_weights, dtype=torch.float32).to(device)

    criterion = FocalLoss(class_weights=class_weights,
                          gamma=1.0,
                          label_smoothing=config.label_smoothing)

    optimizer = optim.AdamW(model.parameters(),
                            lr=config.learning_rate,
                            weight_decay=config.weight_decay,
                            eps=1e-7)          # more numerically stable than 1e-8

    # CosineAnnealingWarmRestarts gives periodic LR resets that help escape
    # the flat frequency-space loss landscape, unlike plain CosineAnnealingLR.
    scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=1, eta_min=5e-6
    )

    scaler = torch.amp.GradScaler('cuda') if device.type == 'cuda' else None

    print(f"\n{'='*60}\nTraining frequency_model_v5  [{device}]")
    start_epoch, history, best_acc, patience = load_checkpoint(
        model, optimizer, scheduler, scaler, config, device
    )
    print(f"{'='*60}")

    if start_epoch >= config.num_epochs:
        print(f"  Already trained {config.num_epochs} epochs.")
        return history, best_acc

    base_lr = config.learning_rate

    for epoch in range(start_epoch, config.num_epochs):
        t0 = time.time()

        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, scaler, device, config, epoch, base_lr
        )
        val_metrics = validate(model, val_loader, criterion, device)

        if epoch >= config.warmup_epochs:
            scheduler.step(epoch - config.warmup_epochs)

        elapsed = time.time() - t0
        cur_lr  = optimizer.param_groups[0]['lr']
        cm      = val_metrics['confusion_matrix']
        bal     = val_metrics['balanced_accuracy']

        print(f"\nEpoch {epoch+1:03d}/{config.num_epochs}  LR={cur_lr:.2e}  [{elapsed:.0f}s]")
        print(f"  Train  loss={train_loss:.4f}  acc={train_acc:.2f}%")
        print(f"  Val    loss={val_metrics['loss']:.4f}  balanced={bal:.2f}%  AUC={val_metrics['auc_roc']:.4f}")
        print(f"  FAKE={val_metrics['fake_accuracy']:.1f}%  REAL={val_metrics['real_accuracy']:.1f}%  F1={val_metrics['f1_score']:.4f}")
        if cm.shape == (2, 2):
            print(f"  CM: TN={cm[0,0]}  FP={cm[0,1]}  FN={cm[1,0]}  TP={cm[1,1]}")

        # Collapse guard
        collapsed = handle_collapse(val_metrics, optimizer, criterion, config, base_lr)
        if collapsed:
            print(f"  WARNING: Model collapsing — intervention applied.")

        for k, v in [('train_loss', train_loss), ('train_acc', train_acc),
                     ('val_loss', val_metrics['loss']), ('val_acc', val_metrics['accuracy']),
                     ('balanced_accuracy', bal),
                     ('fake_accuracy', val_metrics['fake_accuracy']),
                     ('real_accuracy', val_metrics['real_accuracy']),
                     ('val_f1', val_metrics['f1_score']),
                     ('val_auc', val_metrics['auc_roc'])]:
            history[k].append(v)

        is_best = bal > best_acc
        if is_best:
            best_acc = bal
            patience = 0
        else:
            patience += 1

        save_checkpoint(model, optimizer, scheduler, scaler, epoch, val_metrics,
                        config, is_best, history, best_acc, patience)

        if patience >= config.early_stop_patience:
            print(f"\n  Early stop at epoch {epoch+1} (patience={patience})")
            break

    print(f"\n{'='*60}\nBest balanced accuracy: {best_acc:.2f}%\n{'='*60}")

    os.makedirs(config.save_dir, exist_ok=True)
    with open(os.path.join(config.save_dir, 'freq_history.json'), 'w') as f:
        json.dump({k: [float(v) for v in vals] for k, vals in history.items()}, f, indent=2)

    return history, best_acc


# ── DataLoaders ───────────────────────────────────────────────────────────────

def get_dataloaders(config: FreqTrainConfig):
    train_ds = MultiSourceDataset(
        config.data_root,
        get_freq_train_transform(config.image_size),
        config.extra_data, 'train'
    )
    val_ds = MultiSourceDataset(
        config.data_root,
        get_freq_val_transform(config.image_size),
        None, 'test'
    )

    sample_weights = train_ds.get_class_weights()
    sampler = WeightedRandomSampler(sample_weights, len(train_ds), replacement=True)

    nw = min(config.num_workers, os.cpu_count() or 1)
    # persistent_workers=True eliminates per-epoch worker spawn overhead
    # (the single biggest contributor to the 57-min epoch time on RTX 2050)
    train_loader = DataLoader(
        train_ds, batch_size=config.batch_size, sampler=sampler,
        num_workers=nw, pin_memory=True, drop_last=True,
        persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None
    )
    val_loader = DataLoader(
        val_ds, batch_size=config.batch_size * 2, shuffle=False,
        num_workers=nw, pin_memory=True,
        persistent_workers=(nw > 0), prefetch_factor=2 if nw > 0 else None
    )
    return train_loader, val_loader


# ── Entry point ───────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description='Frequency model training (standalone)')
    parser.add_argument('--data_root',    default='CIFAKE')
    parser.add_argument('--extra_data',   nargs='*', default=None)
    parser.add_argument('--batch_size',   type=int,   default=16)
    parser.add_argument('--epochs',       type=int,   default=60)
    parser.add_argument('--lr',           type=float, default=3e-4)
    parser.add_argument('--real_weight',  type=float, default=1.3,
                        help='Loss weight for REAL class (1.0–2.0 recommended)')
    parser.add_argument('--save_dir',     default='checkpoints_freq')
    parser.add_argument('--no_resume',    action='store_true', default=False)
    parser.add_argument('--image_size',   type=int,   default=128)
    parser.add_argument('--workers',      type=int,   default=4)
    parser.add_argument('--dropout',      type=float, default=0.35)
    args = parser.parse_args()

    config = FreqTrainConfig()
    config.data_root     = args.data_root
    config.extra_data    = args.extra_data
    config.batch_size    = args.batch_size
    config.num_epochs    = args.epochs
    config.learning_rate = args.lr
    config.class_weights = [1.0, args.real_weight]
    config.save_dir      = os.path.abspath(args.save_dir)
    config.no_resume     = args.no_resume
    config.image_size    = args.image_size
    config.num_workers   = args.workers
    config.dropout       = args.dropout

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    os.makedirs(config.save_dir, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Frequency Model Training  v5.2")
    print(f"Device: {device} | Save: {config.save_dir}")
    print(f"Resume: {'DISABLED' if args.no_resume else 'AUTO'}")
    print(f"{'='*60}")

    print('\nFREQUENCY MODEL')
    train_loader, val_loader = get_dataloaders(config)

    model = get_frequency_model_v5(
        num_classes=config.num_classes,
        image_size=config.image_size,
        dropout=config.dropout,
    ).to(device)

    # torch.compile gives ~15–25 % throughput gain on CUDA without changing results
    import platform, sys
    _can_compile = (
        platform.system() != 'Windows'
        and sys.version_info >= (3, 8)
        and torch.cuda.is_available()
    )
    if _can_compile:
        try:
            model = torch.compile(model, mode='reduce-overhead')
            print('  torch.compile: ON (reduce-overhead)')
        except Exception as e:
            print(f'  torch.compile: failed ({e}), skipping')
    else:
        print('  torch.compile: skipped (Windows or no CUDA — Triton unavailable)')

    train(model, train_loader, val_loader, config, device)

    best_path = os.path.join(config.save_dir, 'freq_model_best.pth')
    if os.path.exists(best_path):
        ckpt = torch.load(best_path, map_location=device, weights_only=False)
        # unwrap compile wrapper for calibration
        base_model = model._orig_mod if hasattr(model, '_orig_mod') else model
        base_model.load_state_dict(ckpt['model_state_dict'], strict=False)
        base_model.calibrate(val_loader, device)
        torch.save({'model_state_dict': base_model.state_dict()},
                   os.path.join(config.save_dir, 'freq_model_calibrated.pth'))
        print(f"Calibrated model saved.")

    print(f'\nDone. Checkpoints: {config.save_dir}/')


if __name__ == '__main__':
    import multiprocessing
    multiprocessing.freeze_support()
    main()