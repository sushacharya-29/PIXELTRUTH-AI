"""
test_forensics.py — Comprehensive Performance Test for ForensicsAI / SpatialRobustModel
========================================================================================
Tests every major capability of the system:
  1.  Checkpoint loading (latest / best / final)
  2.  Inference correctness (predict API, forward_with_streams)
  3.  Classification metrics (Acc, Bal-Acc, AUC-ROC, F1, Precision, Recall per-class)
  4.  Calibration quality (ECE, MCE, reliability diagram data)
  5.  MC-Dropout uncertainty (predict_with_uncertainty)
  6.  Adversarial robustness (FGSM ε=4/255 and ε=8/255, PGD-7)
  7.  Stream diagnostics (gate weights, reliability, epistemic, parity_alpha)
  8.  Explainability report (get_explainability_report)
  9.  Generator head (forward_with_generator — GAN-type attribution)
  10. GradCAM visual generation
  11. Throughput / latency (images/sec, ms/image)
  12. Confidence distribution analysis (FAKE vs REAL separation)
  13. Error memory summary
  14. NaN / degenerate input stability
  15. Single-image inference (the deploy use-case)

Usage:
  # Evaluate against a labelled dataset
  python test_forensics.py \\
      --ckpt_dir  ./checkpoints \\
      --data_dir  /path/to/test_dataset \\
      --batch_size 8 \\
      --workers   2

  # Single-image quick check (no dataset needed)
  python test_forensics.py \\
      --ckpt_dir  ./checkpoints \\
      --image     /path/to/image.jpg

  # Full adversarial suite (slower)
  python test_forensics.py \\
      --ckpt_dir ./checkpoints \\
      --data_dir /path/to/test_dataset \\
      --adv_eval \\
      --mc_passes 20

Dataset layout (same as training):
  data_dir/
    real/   (or REAL/)   → label 1
    fake/   (or FAKE/)   → label 0
"""

import os
import sys
import json
import time
import random
import argparse
import warnings
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
from ForensicsAI import SpatialRobustModel

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile, UnidentifiedImageError

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)

# ── Make ForensicsAI importable from the same dir or via --src_dir ─────────────
def _add_src(path: str):
    if path and path not in sys.path:
        sys.path.insert(0, path)

# ── ANSI colours for terminal output ───────────────────────────────────────────
GRN  = "\033[92m"
YLW  = "\033[93m"
RED  = "\033[91m"
CYN  = "\033[96m"
BLD  = "\033[1m"
RST  = "\033[0m"

def _ok(msg):  print(f"  {GRN}✅ {msg}{RST}")
def _warn(msg): print(f"  {YLW}⚠️  {msg}{RST}")
def _fail(msg): print(f"  {RED}❌ {msg}{RST}")
def _hdr(msg):  print(f"\n{BLD}{CYN}{'─'*70}\n  {msg}\n{'─'*70}{RST}")
def _sub(msg):  print(f"  {BLD}{msg}{RST}")


# ══════════════════════════════════════════════════════════════════════════════
# 1.  CHECKPOINT LOADING
# ══════════════════════════════════════════════════════════════════════════════

def resolve_checkpoint(ckpt_dir: str, prefer: str = "best") -> Optional[str]:
    """
    Searches ckpt_dir for checkpoint files in priority order.
    Priority: best → final → latest (checkpoint.pt) → any .pt
    """
    d = Path(ckpt_dir)
    candidates = {
        "best":     d / "best.pt",
        "final":    d / "final.pt",
        "latest":   d / "checkpoint.pt",
    }
    priority = ["best", "final", "latest"] if prefer == "best" else [prefer, "best", "final", "latest"]
    for key in priority:
        p = candidates.get(key)
        if p and p.exists():
            return str(p)
    # Fallback: any .pt in the directory
    pts = sorted(d.glob("*.pt"))
    return str(pts[-1]) if pts else None


def load_model(ckpt_path: Optional[str], device: torch.device, args) -> "SpatialRobustModel":
    """
    Build SpatialRobustModel and load checkpoint weights (strict=False for
    forward-compatibility when streams were pruned post-training).
    """
    from ForensicsAI import get_spatial_robust_model, load_checkpoint

    model = get_spatial_robust_model(
        freeze_backbone=True,
        use_generator_head=True,
        n_iters=args.n_iters,
        use_isfcr=True,
        use_fcw=True,
        dropout=0.0,          # deterministic for eval (MC uses train() override)
        drop_path_rate=0.0,
    ).to(device)

    if ckpt_path:
        ckpt = load_checkpoint(ckpt_path, map_location=device)
        if not ckpt:
            _warn(f"Empty or unreadable checkpoint at {ckpt_path} — using random weights")
        else:
            state = ckpt.get("model_state_dict", ckpt)
            missing, unexpected = model.load_state_dict(state, strict=False)
            if missing:
                _warn(f"Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing)>5 else ''}")
            if unexpected:
                _warn(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected)>5 else ''}")
            epoch = ckpt.get("epoch", "?")
            best  = ckpt.get("best_val_acc", None)
            print(f"  Loaded checkpoint: epoch={epoch}  best_bal_acc={best}")
    else:
        _warn("No checkpoint found — running with random weights (sanity check only)")

    model.eval()
    return model


# ══════════════════════════════════════════════════════════════════════════════
# 2.  DATASET
# ══════════════════════════════════════════════════════════════════════════════

VALID_EXTS = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.tif', '.webp', '.gif'}

def _find_class_dirs(root: Path) -> Dict[int, Path]:
    dirs = {d.name.lower(): d for d in root.iterdir() if d.is_dir()}
    fake_dir = dirs.get('fake') or dirs.get('0')
    real_dir = dirs.get('real') or dirs.get('1')
    if fake_dir is None or real_dir is None:
        raise FileNotFoundError(
            f"Expected fake/ and real/ subdirs in {root}. Found: {list(dirs.keys())}"
        )
    return {0: fake_dir, 1: real_dir}


class TestDataset(Dataset):
    def __init__(self, root: str, transform):
        self.transform = transform
        self.samples: List[Tuple[str, int]] = []
        root_path = Path(root)
        class_dirs = _find_class_dirs(root_path)
        seen = set()
        for label, class_dir in class_dirs.items():
            for ext in VALID_EXTS:
                for fpath in class_dir.rglob(f'*{ext}'):
                    if str(fpath) not in seen:
                        seen.add(str(fpath))
                        self.samples.append((str(fpath), label))
                for fpath in class_dir.rglob(f'*{ext.upper()}'):
                    if str(fpath) not in seen:
                        seen.add(str(fpath))
                        self.samples.append((str(fpath), label))
        counts = defaultdict(int)
        for _, l in self.samples:
            counts[l] += 1
        print(f"  TestDataset: FAKE={counts[0]}  REAL={counts[1]}  Total={len(self.samples)}")

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]
        try:
            img = Image.open(path).convert('RGB')
        except Exception:
            img = Image.new('RGB', (224, 224), (127, 127, 127))
        return self.transform(img), label, path


def get_test_transform(input_size: int = 224):
    from ForensicsAI import CLIP_MEAN, CLIP_STD
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


def collate_fn(batch):
    imgs   = torch.stack([b[0] for b in batch])
    labels = torch.tensor([b[1] for b in batch], dtype=torch.long)
    paths  = [b[2] for b in batch]
    return imgs, labels, paths


def build_loader(data_dir: str, batch_size: int, workers: int) -> DataLoader:
    tf = get_test_transform()
    ds = TestDataset(data_dir, tf)
    return DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=workers, pin_memory=True,
        collate_fn=collate_fn, drop_last=False,
    )


# ══════════════════════════════════════════════════════════════════════════════
# 3.  CORE METRICS
# ══════════════════════════════════════════════════════════════════════════════

def balanced_accuracy(preds: np.ndarray, labels: np.ndarray) -> float:
    per_class = []
    for c in range(2):
        mask = labels == c
        if mask.sum() == 0: continue
        per_class.append((preds[mask] == labels[mask]).mean())
    return float(np.mean(per_class)) if per_class else 0.0


def per_class_recall(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    result = {}
    for c, name in [(0, "FAKE"), (1, "REAL")]:
        mask = labels == c
        if mask.sum() == 0:
            result[name] = float("nan")
        else:
            result[name] = float((preds[mask] == c).mean())
    return result


def per_class_precision(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    result = {}
    for c, name in [(0, "FAKE"), (1, "REAL")]:
        mask = preds == c
        if mask.sum() == 0:
            result[name] = float("nan")
        else:
            result[name] = float((labels[mask] == c).mean())
    return result


def f1_score(prec: float, rec: float) -> float:
    if prec + rec == 0: return 0.0
    return 2 * prec * rec / (prec + rec)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Simple trapezoidal AUC (no sklearn required)."""
    thresholds = np.linspace(0, 1, 201)
    tprs, fprs = [], []
    for t in thresholds:
        pred = (scores >= t).astype(int)
        tp = ((pred == 1) & (labels == 1)).sum()
        fp = ((pred == 1) & (labels == 0)).sum()
        tn = ((pred == 0) & (labels == 0)).sum()
        fn = ((pred == 0) & (labels == 1)).sum()
        tprs.append(tp / max(tp + fn, 1))
        fprs.append(fp / max(fp + tn, 1))
    tprs, fprs = np.array(tprs), np.array(fprs)
    order = np.argsort(fprs)
    return float(np.trapz(tprs[order], fprs[order]))


def expected_calibration_error(confs: np.ndarray, corrects: np.ndarray,
                                n_bins: int = 10) -> Tuple[float, float]:
    """Returns (ECE, MCE)."""
    bins   = np.linspace(0, 1, n_bins + 1)
    ece, mce = 0.0, 0.0
    N = len(confs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if mask.sum() == 0: continue
        bin_acc  = corrects[mask].mean()
        bin_conf = confs[mask].mean()
        gap = abs(bin_acc - bin_conf)
        ece += gap * mask.sum() / N
        mce  = max(mce, gap)
    return ece, mce


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    """
    Single pass over the test set.
    Returns collected predictions, confidences, labels, stream diagnostics.
    """
    model.eval()
    all_preds, all_labels, all_confs, all_scores = [], [], [], []
    stream_accum: Dict[str, List[float]] = defaultdict(list)
    gate_accum:   Dict[str, List[float]] = defaultdict(list)
    epistemic_accum: Dict[str, List[float]] = defaultdict(list)
    reliability_accum: Dict[str, List[float]] = defaultdict(list)
    parity_alphas = []
    error_paths: List[Tuple[str, int, int, float]] = []  # (path, true, pred, conf)

    for imgs, labels, paths in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        logits, info = model.forward_with_streams(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        confs  = probs.max(dim=1).values
        scores = probs[:, 1]   # P(REAL) for AUC

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_confs.extend(confs.cpu().numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())

        # Stream diagnostics
        for k, v in info.get('_gate_weights', {}).items():
            gate_accum[k].append(v)
        for k, v in info.get('_epistemic', {}).items():
            epistemic_accum[k].append(v)
        for k, v in info.get('_reliability', {}).items():
            reliability_accum[k].append(v)
        parity_alphas.append(info.get('_parity_alpha', float('nan')))

        # Error memory update
        model.step_error_memory(
            labels.cpu(), preds.cpu(), confs.cpu(), info)

        # Track misclassified paths
        wrong = (preds != labels).cpu()
        for i, w in enumerate(wrong):
            if w:
                error_paths.append((
                    paths[i],
                    int(labels[i].item()),
                    int(preds[i].item()),
                    float(confs[i].item()),
                ))

    return {
        'preds':    np.array(all_preds),
        'labels':   np.array(all_labels),
        'confs':    np.array(all_confs),
        'scores':   np.array(all_scores),
        'gate_weights':   {k: float(np.mean(v)) for k, v in gate_accum.items()},
        'epistemic':      {k: float(np.mean(v)) for k, v in epistemic_accum.items()},
        'reliability':    {k: float(np.mean(v)) for k, v in reliability_accum.items()},
        'parity_alpha':   float(np.nanmean(parity_alphas)),
        'error_paths':    error_paths,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ADVERSARIAL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════

def adversarial_eval(model, loader, device,
                     fgsm_eps_list=(4/255, 8/255),
                     pgd_eps=8/255, pgd_steps=7,
                     max_batches=20) -> Dict:
    """FGSM + PGD evaluation. Returns bal_acc drop and raw accuracies."""
    from ForensicsAI import AdversarialTester
    tester = AdversarialTester(model, device)

    results = {}
    # Clean baseline (first max_batches batches)
    clean_preds, clean_labels = [], []
    for i, (imgs, labels, _) in enumerate(loader):
        if i >= max_batches: break
        imgs, labels = imgs.to(device), labels.to(device)
        with torch.no_grad():
            preds = model(imgs).argmax(1)
        clean_preds.extend(preds.cpu().numpy())
        clean_labels.extend(labels.cpu().numpy())
    clean_bal = balanced_accuracy(np.array(clean_preds), np.array(clean_labels))
    results['clean_bal_acc'] = clean_bal

    # FGSM for each epsilon
    for eps in fgsm_eps_list:
        adv_preds, adv_labels = [], []
        for i, (imgs, labels, _) in enumerate(loader):
            if i >= max_batches: break
            imgs, labels = imgs.to(device), labels.to(device)
            adv = tester.fgsm(imgs, labels, eps=eps)
            with torch.no_grad():
                preds = model(adv).argmax(1)
            adv_preds.extend(preds.cpu().numpy())
            adv_labels.extend(labels.cpu().numpy())
        bal = balanced_accuracy(np.array(adv_preds), np.array(adv_labels))
        key = f"fgsm_{int(eps*255)}px255"
        results[key] = bal
        results[f"{key}_drop"] = clean_bal - bal

    # PGD-7
    pgd_preds, pgd_labels = [], []
    for i, (imgs, labels, _) in enumerate(loader):
        if i >= max_batches: break
        imgs, labels = imgs.to(device), labels.to(device)
        adv = tester.pgd(imgs, labels, eps=pgd_eps, alpha=2/255, steps=pgd_steps)
        with torch.no_grad():
            preds = model(adv).argmax(1)
        pgd_preds.extend(preds.cpu().numpy())
        pgd_labels.extend(labels.cpu().numpy())
    pgd_bal = balanced_accuracy(np.array(pgd_preds), np.array(pgd_labels))
    results['pgd7_bal_acc'] = pgd_bal
    results['pgd7_drop']    = clean_bal - pgd_bal

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MC-DROPOUT UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════

def mc_uncertainty_eval(model, loader, device,
                         n_passes: int = 10,
                         max_batches: int = 20) -> Dict:
    """
    Runs MC-Dropout and measures:
    - Mean uncertainty for correct vs incorrect predictions
    - Fraction of uncertain predictions (uncertainty > threshold)
    """
    from ForensicsAI import predict_with_uncertainty

    unc_correct, unc_wrong = [], []
    for i, (imgs, labels, _) in enumerate(loader):
        if i >= max_batches: break
        imgs = imgs.to(device)
        result = predict_with_uncertainty(model, imgs, n_passes=n_passes)
        preds = result['prediction'].cpu().numpy()
        uncs  = result['uncertainty'].cpu().numpy()
        labs  = labels.numpy()
        for p, u, l in zip(preds, uncs, labs):
            if p == l:
                unc_correct.append(u)
            else:
                unc_wrong.append(u)

    mean_unc_correct = float(np.mean(unc_correct)) if unc_correct else float('nan')
    mean_unc_wrong   = float(np.mean(unc_wrong))   if unc_wrong   else float('nan')
    return {
        'mean_uncertainty_correct': mean_unc_correct,
        'mean_uncertainty_wrong':   mean_unc_wrong,
        'n_correct': len(unc_correct),
        'n_wrong':   len(unc_wrong),
        'separation_ratio': (mean_unc_wrong / (mean_unc_correct + 1e-8)
                             if unc_correct and unc_wrong else float('nan')),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  THROUGHPUT
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def throughput_test(model, device, input_size: int = 224,
                    batch_size: int = 8, n_warmup: int = 5, n_runs: int = 20) -> Dict:
    """Measures forward-pass throughput in images/sec and ms/image."""
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    model.eval()
    # Warm-up
    for _ in range(n_warmup):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        _ = model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_images = batch_size * n_runs
    ips = total_images / elapsed
    ms_per_image = 1000.0 * elapsed / total_images
    return {
        'images_per_sec': round(ips, 1),
        'ms_per_image':   round(ms_per_image, 3),
        'batch_size':     batch_size,
        'n_runs':         n_runs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  SINGLE-IMAGE INFERENCE (deploy use-case)
# ══════════════════════════════════════════════════════════════════════════════

def single_image_test(model, image_path: str, device: torch.device) -> Dict:
    """Runs full predict() + explainability on one image."""
    tf = get_test_transform()
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {'error': str(e)}

    tensor = tf(img_pil).unsqueeze(0).to(device)   # (1, 3, 224, 224)

    model.eval()
    result      = model.predict(tensor)
    expl_report = model.get_explainability_report(tensor)

    label_map = {0: "FAKE / GAN", 1: "REAL"}
    pred_idx  = int(result['prediction'][0].item())
    conf      = float(result['confidence'][0].item())
    unc       = float(result['uncertainty'])

    return {
        'image':           image_path,
        'prediction':      label_map[pred_idx],
        'prediction_idx':  pred_idx,
        'confidence':      conf,
        'uncertainty':     unc,
        'stream_importance':      result['stream_importance'],
        'stream_reliability':     result['stream_reliability'],
        'contradiction_strength': result['contradiction_strength'],
        'explainability':  expl_report,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 9.  GRADCAM TEST
# ══════════════════════════════════════════════════════════════════════════════

def gradcam_test(model, device) -> bool:
    """Checks GradCAM produces a valid heatmap (no crash, correct shape)."""
    from ForensicsAI import GradCAM
    dummy = torch.randn(1, 3, 224, 224, device=device)
    try:
        cam = GradCAM(model)
        heatmap = cam.generate(dummy, class_idx=1)
        assert heatmap.shape == (224, 224), f"Unexpected shape: {heatmap.shape}"
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "Heatmap out of [0,1]"
        pixel_cam = cam.generate_pixel_attn_cam(dummy)
        assert pixel_cam.shape == (224, 224)
        return True
    except Exception as e:
        _fail(f"GradCAM error: {e}")
        return False


# ══════════════════════════════════════════════════════════════════════════════
# 10. GENERATOR HEAD TEST
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def generator_head_test(model, loader, device, max_batches: int = 5) -> Dict:
    """
    Tests forward_with_generator — maps each image to one of:
      real | dalle | midjourney | stable_diffusion | gemini | etc.
    Returns top-1 predicted GAN type breakdown.
    """
    from ForensicsAI import GENERATOR_LABELS

    gen_counts = defaultdict(int)
    total = 0
    for i, (imgs, labels, _) in enumerate(loader):
        if i >= max_batches: break
        imgs = imgs.to(device)
        bin_logits, gen_logits = model.forward_with_generator(imgs)
        if gen_logits is None:
            return {'generator_head': 'disabled'}
        gen_preds = gen_logits.argmax(dim=1).cpu().numpy()
        for p in gen_preds:
            label = GENERATOR_LABELS[p] if p < len(GENERATOR_LABELS) else f"class_{p}"
            gen_counts[label] += 1
            total += 1

    return {
        'generator_type_breakdown': dict(gen_counts),
        'total_images': total,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 11. CONFIDENCE DISTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def confidence_analysis(preds: np.ndarray, labels: np.ndarray,
                        confs: np.ndarray) -> Dict:
    """
    Analyses confidence distributions for FAKE and REAL, separately
    for correct and incorrect predictions.
    """
    def _stats(arr):
        if len(arr) == 0: return {'mean': float('nan'), 'std': float('nan'),
                                   'p10': float('nan'), 'p90': float('nan')}
        return {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'p10':  float(np.percentile(arr, 10)),
            'p90':  float(np.percentile(arr, 90)),
        }

    mask_fake  = labels == 0
    mask_real  = labels == 1
    correct    = preds == labels
    wrong      = ~correct

    return {
        'FAKE_correct': _stats(confs[mask_fake & correct]),
        'FAKE_wrong':   _stats(confs[mask_fake & wrong]),
        'REAL_correct': _stats(confs[mask_real & correct]),
        'REAL_wrong':   _stats(confs[mask_real & wrong]),
        'overall_correct': _stats(confs[correct]),
        'overall_wrong':   _stats(confs[wrong]),
        'high_conf_errors': int(((confs > 0.9) & wrong).sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 12. NaN STABILITY
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def nan_stability_test(model, device) -> bool:
    x_nan  = torch.full((1, 3, 224, 224), float('nan'),  device=device)
    x_zero = torch.zeros((1, 3, 224, 224), device=device)
    x_inf  = torch.full((1, 3, 224, 224), float('inf'),  device=device)
    ok = True
    for name, x in [("NaN", x_nan), ("Zero", x_zero), ("Inf", x_inf)]:
        try:
            out = model(x)
            if not torch.isfinite(out).all():
                _fail(f"{name} input → non-finite logits")
                ok = False
        except Exception as e:
            _fail(f"{name} input crashed model: {e}")
            ok = False
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, pct=False, dec=2):
    if isinstance(v, float) and np.isnan(v): return "N/A"
    if pct: return f"{v*100:.{dec}f}%"
    return f"{v:.{dec}f}"

def _verdict(bal_acc: float) -> str:
    if bal_acc >= 0.90: return f"{GRN}EXCELLENT{RST}"
    if bal_acc >= 0.80: return f"{GRN}GOOD{RST}"
    if bal_acc >= 0.70: return f"{YLW}FAIR{RST}"
    return f"{RED}POOR{RST}"


def print_dataset_results(res: Dict):
    preds  = res['preds']
    labels = res['labels']
    confs  = res['confs']
    scores = res['scores']

    acc     = (preds == labels).mean()
    bal_acc = balanced_accuracy(preds, labels)
    recall  = per_class_recall(preds, labels)
    prec    = per_class_precision(preds, labels)
    auc     = roc_auc(labels, scores)
    ece, mce = expected_calibration_error(confs, (preds == labels).astype(float))

    f1_fake = f1_score(prec.get('FAKE', 0), recall.get('FAKE', 0))
    f1_real = f1_score(prec.get('REAL', 0), recall.get('REAL', 0))
    f1_mac  = (f1_fake + f1_real) / 2

    _hdr("CLASSIFICATION METRICS")
    print(f"  Accuracy          : {_fmt(acc, pct=True)}")
    print(f"  Balanced Accuracy : {_fmt(bal_acc, pct=True)}  [{_verdict(bal_acc)}]")
    print(f"  AUC-ROC           : {_fmt(auc, dec=4)}")
    print(f"  Macro F1          : {_fmt(f1_mac, pct=True)}")
    print()
    print(f"  {'Class':<10}  {'Precision':>10}  {'Recall':>10}  {'F1':>10}")
    print(f"  {'─'*46}")
    for cls in ['FAKE', 'REAL']:
        p = prec.get(cls, float('nan'))
        r = recall.get(cls, float('nan'))
        f = f1_score(p, r) if not (np.isnan(p) or np.isnan(r)) else float('nan')
        print(f"  {cls:<10}  {_fmt(p, pct=True):>10}  {_fmt(r, pct=True):>10}  {_fmt(f, pct=True):>10}")

    _hdr("CALIBRATION")
    print(f"  Expected Calibration Error (ECE) : {_fmt(ece, dec=4)}")
    print(f"  Maximum Calibration Error  (MCE) : {_fmt(mce, dec=4)}")
    if ece < 0.05:  _ok("Well-calibrated (ECE < 5%)")
    elif ece < 0.10: _warn("Moderately calibrated (ECE 5–10%)")
    else:            _fail("Poorly calibrated (ECE > 10%)")

    _hdr("CONFIDENCE DISTRIBUTION")
    cd = confidence_analysis(preds, labels, confs)
    for key in ['REAL_correct', 'REAL_wrong', 'FAKE_correct', 'FAKE_wrong']:
        s = cd[key]
        print(f"  {key:<16}: mean={_fmt(s['mean'], pct=True)}  "
              f"std={_fmt(s['std'], pct=True)}  "
              f"[P10={_fmt(s['p10'], pct=True)}, P90={_fmt(s['p90'], pct=True)}]")
    print(f"  High-conf errors (>90% conf, wrong): {cd['high_conf_errors']}")

    _hdr("STREAM DIAGNOSTICS")
    _sub("Gate weights (how much each stream influences the decision):")
    for k, v in sorted(res['gate_weights'].items(), key=lambda x: -x[1]):
        bar = '█' * int(v * 30)
        print(f"    {k:<18}: {v:.4f}  {bar}")
    _sub(f"\nParity alpha (forensic authority): {res['parity_alpha']:.4f}  "
         f"[0.3=semantic-led, 0.7=forensic-led]")
    _sub("\nStream epistemic uncertainty (lower = more confident):")
    for k, v in sorted(res['epistemic'].items()):
        print(f"    {k:<18}: {v:.4f}")
    _sub("\nStream reliability:")
    for k, v in sorted(res['reliability'].items()):
        print(f"    {k:<18}: {v:.4f}")

    _hdr("ERRORS")
    print(f"  Total misclassified : {len(res['error_paths'])}")
    if res['error_paths']:
        print(f"  Worst 5 (by confidence):")
        worst = sorted(res['error_paths'], key=lambda x: -x[3])[:5]
        for path, true_l, pred_l, cf in worst:
            tname = "REAL" if true_l == 1 else "FAKE"
            pname = "REAL" if pred_l == 1 else "FAKE"
            print(f"    [{tname}→{pname}] conf={cf:.3f}  {Path(path).name}")


def print_adv_results(adv: Dict):
    _hdr("ADVERSARIAL ROBUSTNESS")
    clean = adv.get('clean_bal_acc', float('nan'))
    print(f"  Clean bal_acc : {_fmt(clean, pct=True)}")
    for key, val in adv.items():
        if key == 'clean_bal_acc': continue
        if 'drop' in key:
            color = GRN if val < 0.05 else (YLW if val < 0.15 else RED)
            print(f"  {key:<28}: {color}{_fmt(val, pct=True)}{RST}")
        else:
            print(f"  {key:<28}: {_fmt(val, pct=True)}")


def print_mc_results(mc: Dict):
    _hdr("MC-DROPOUT UNCERTAINTY")
    r = mc.get('separation_ratio', float('nan'))
    print(f"  Mean uncertainty (correct preds) : {_fmt(mc['mean_uncertainty_correct'], dec=4)}")
    print(f"  Mean uncertainty (wrong   preds) : {_fmt(mc['mean_uncertainty_wrong'], dec=4)}")
    print(f"  Uncertainty separation ratio     : {_fmt(r, dec=2)}×")
    if not np.isnan(r):
        if r > 2.0:   _ok("Model is more uncertain on its mistakes (healthy)")
        elif r > 1.2: _warn("Weak uncertainty separation")
        else:          _fail("Model is not more uncertain on wrong predictions")


def print_throughput_results(tp: Dict):
    _hdr("THROUGHPUT / LATENCY")
    print(f"  Images/sec  : {tp['images_per_sec']}")
    print(f"  ms/image    : {tp['ms_per_image']}")
    print(f"  Batch size  : {tp['batch_size']}")


def print_single_image_result(res: Dict):
    _hdr("SINGLE-IMAGE RESULT")
    if 'error' in res:
        _fail(f"Error: {res['error']}")
        return

    is_real = res['prediction_idx'] == 1
    conf    = res['confidence']
    unc     = res['uncertainty']

    # ── Big clear verdict ────────────────────────────────────────────────────
    print()
    if is_real:
        verdict_line = f"{GRN}{BLD}  ✅  VERDICT : REAL IMAGE{RST}"
    else:
        verdict_line = f"{RED}{BLD}  🚨  VERDICT : FAKE / GAN IMAGE{RST}"
    print(verdict_line)
    print()

    # Confidence bar
    bar_len  = 40
    filled   = int(conf * bar_len)
    bar      = (GRN if is_real else RED) + '█' * filled + RST + '░' * (bar_len - filled)
    print(f"  Confidence  : [{bar}] {conf*100:.1f}%")

    # Uncertainty indicator
    unc_label = "Low (confident)" if unc < 0.05 else ("Medium" if unc < 0.15 else "High (uncertain)")
    print(f"  Uncertainty : {unc:.4f}  ({unc_label})")
    print(f"  Image       : {res['image']}")
    print()

    # Stream votes
    _sub("  Stream votes (which forensic signals drove the decision):")
    for k, v in sorted(res['stream_importance'].items(), key=lambda x: -x[1]):
        bar_s = '█' * int(v * 30)
        print(f"    {k:<18}: {v:.3f}  {bar_s}")

    print()
    _sub("  Stream reliability (trust score per stream):")
    for k, v in sorted(res['stream_reliability'].items(), key=lambda x: -x[1]):
        color = GRN if v > 0.7 else (YLW if v > 0.4 else RED)
        print(f"    {k:<18}: {color}{v:.3f}{RST}")

    print()
    contr = res['contradiction_strength']
    contr_label = ("Low — streams agree" if contr < 0.05
                   else ("Medium — some disagreement" if contr < 0.15
                         else "High — streams contradict each other"))
    print(f"  Stream contradiction: {contr:.4f}  ({contr_label})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="ForensicsAI Test Suite")
    p.add_argument("--ckpt_dir",   default="./checkpoints",
                   help="Directory containing best.pt / final.pt / checkpoint.pt")
    p.add_argument("--ckpt_file",  default=None,
                   help="Explicit checkpoint file path (overrides --ckpt_dir search)")
    p.add_argument("--data_dir",   default=None,
                   help="Test dataset root (real/ + fake/ subdirs)")
    p.add_argument("--image",      default=None,
                   help="Single image path for quick inference test")
    p.add_argument("--src_dir",    default=".",
                   help="Directory containing ForensicsAI.py (default: current dir)")
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--workers",    type=int, default=2)
    p.add_argument("--n_iters",    type=int, default=2,
                   help="ISFCR reasoning iterations (match training value)")
    p.add_argument("--adv_eval",   action="store_true",
                   help="Run adversarial robustness evaluation (slower)")
    p.add_argument("--adv_batches",type=int, default=20,
                   help="Max batches for adversarial eval")
    p.add_argument("--mc_eval",    action="store_true",
                   help="Run MC-Dropout uncertainty evaluation")
    p.add_argument("--mc_passes",  type=int, default=10,
                   help="MC-Dropout forward passes")
    p.add_argument("--mc_batches", type=int, default=20)
    p.add_argument("--gradcam",    action="store_true",
                   help="Run GradCAM sanity check")
    p.add_argument("--save_json",  default=None,
                   help="Save full results dict to this JSON path")
    p.add_argument("--seed",       type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    _add_src(args.src_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{BLD}ForensicsAI Test Suite{RST}  |  device={device}")
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(0)
        print(f"  GPU: {prop.name}  {prop.total_memory/1e9:.1f} GB")

    # ── Resolve checkpoint ───────────────────────────────────────────────────
    _hdr("CHECKPOINT LOADING")
    ckpt_path = args.ckpt_file or resolve_checkpoint(args.ckpt_dir)
    if ckpt_path:
        print(f"  Using checkpoint: {ckpt_path}")
    else:
        _warn("No checkpoint found. All results use random weights.")

    model = load_model(ckpt_path, device, args)
    total_params = sum(p.numel() for p in model.parameters())
    train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: total={total_params/1e6:.1f}M  trainable={train_params/1e6:.1f}M")
    _ok("Model loaded")

    all_results = {}

    # ── NaN stability (always runs) ──────────────────────────────────────────
    _hdr("STABILITY TESTS")
    if nan_stability_test(model, device):
        _ok("NaN / zero / inf inputs all produce finite logits")
    else:
        _fail("Stability test FAILED")

    # ── Throughput (always runs) ─────────────────────────────────────────────
    tp = throughput_test(model, device, batch_size=args.batch_size)
    print_throughput_results(tp)
    all_results['throughput'] = tp

    # ── GradCAM ─────────────────────────────────────────────────────────────
    if args.gradcam:
        _hdr("GRADCAM")
        ok = gradcam_test(model, device)
        if ok: _ok("FreqCAM + PixelAttnCAM both produce 224×224 heatmaps")
        all_results['gradcam_ok'] = ok

    # ── Single-image inference ───────────────────────────────────────────────
    if args.image:
        res = single_image_test(model, args.image, device)
        print_single_image_result(res)
        all_results['single_image'] = {k: v for k, v in res.items()
                                        if k != 'explainability'}

    # ── Dataset evaluation ───────────────────────────────────────────────────
    if args.data_dir:
        _hdr("LOADING TEST DATASET")
        loader = build_loader(args.data_dir, args.batch_size, args.workers)

        _hdr("RUNNING EVALUATION")
        t0 = time.time()
        res = evaluate(model, loader, device)
        print(f"  Evaluated {len(res['preds'])} images in {time.time()-t0:.1f}s")
        print_dataset_results(res)
        all_results['metrics'] = {
            'accuracy':         float((res['preds'] == res['labels']).mean()),
            'balanced_accuracy': float(balanced_accuracy(res['preds'], res['labels'])),
            'auc_roc':          float(roc_auc(res['labels'], res['scores'])),
            'recall':           per_class_recall(res['preds'], res['labels']),
            'precision':        per_class_precision(res['preds'], res['labels']),
            'gate_weights':     res['gate_weights'],
            'parity_alpha':     res['parity_alpha'],
        }

        # Error memory summary
        _hdr("ERROR MEMORY SUMMARY")
        err_summary = model.error_memory.summary()
        print(f"  real_as_fake       : {err_summary.get('real_as_fake', '?')}")
        print(f"  fake_as_real       : {err_summary.get('fake_as_real', '?')}")
        print(f"  ema_fake_err_rate  : {err_summary.get('ema_fake_err_rate', 0):.4f}")
        print(f"  ema_real_err_rate  : {err_summary.get('ema_real_err_rate', 0):.4f}")
        all_results['error_memory'] = {
            k: v for k, v in err_summary.items() if isinstance(v, (int, float))
        }

        # Routing report
        _hdr("STREAM ROUTING REPORT")
        print(model.get_routing_report())

        # Generator head
        _hdr("GENERATOR HEAD (GAN-type attribution)")
        gen_res = generator_head_test(model, loader, device)
        if 'generator_head' in gen_res and gen_res['generator_head'] == 'disabled':
            _warn("Generator head not enabled in this checkpoint")
        else:
            for k, v in sorted(gen_res['generator_type_breakdown'].items(),
                                key=lambda x: -x[1]):
                bar = '█' * max(1, int(30 * v / max(gen_res['total_images'], 1)))
                print(f"  {k:<20}: {v:>5}  {bar}")
        all_results['generator_head'] = gen_res

        # MC-Dropout
        if args.mc_eval:
            _hdr("MC-DROPOUT UNCERTAINTY")
            mc = mc_uncertainty_eval(model, loader, device,
                                      n_passes=args.mc_passes,
                                      max_batches=args.mc_batches)
            print_mc_results(mc)
            all_results['mc_uncertainty'] = mc

        # Adversarial
        if args.adv_eval:
            _hdr("ADVERSARIAL ROBUSTNESS (running…)")
            adv = adversarial_eval(model, loader, device,
                                    max_batches=args.adv_batches)
            print_adv_results(adv)
            all_results['adversarial'] = adv

    # ── Save JSON ────────────────────────────────────────────────────────────
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        _ok(f"Results saved to {args.save_json}")

    # ── Final summary ────────────────────────────────────────────────────────
    print(f"\n{'═'*70}")
    print(f"  {BLD}TEST COMPLETE{RST}")
    if 'metrics' in all_results:
        b = all_results['metrics']['balanced_accuracy']
        a = all_results['metrics']['auc_roc']
        r_real = all_results['metrics']['recall'].get('REAL', float('nan'))
        r_fake = all_results['metrics']['recall'].get('FAKE', float('nan'))
        print(f"  Balanced Acc: {_fmt(b, pct=True)}  AUC: {_fmt(a, dec=4)}  "
              f"REAL-Recall: {_fmt(r_real, pct=True)}  FAKE-Recall: {_fmt(r_fake, pct=True)}")
        print(f"  Verdict: {_verdict(b)}")
    print(f"{'═'*70}\n")


if __name__ == '__main__':
    main()