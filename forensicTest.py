"""
forensicTest.py — Comprehensive Test Suite for ForensicEngine (Stable Edition)
================================================================================
Aligned with authenticityEngine.py. Tests every public API surface:

  1.  Checkpoint loading (best / final / latest / explicit path)
  2.  Inference correctness (predict, forward_with_streams, forward_with_entropy)
  3.  Classification metrics (Acc, Bal-Acc, AUC-ROC, F1, Precision, Recall per-class)
  4.  Calibration quality (ECE, MCE)
  5.  MC-Dropout uncertainty (predict_with_uncertainty)
  6.  Adversarial robustness (FGSM ε=4/255 + ε=8/255, PGD-7, CW-L2)
  7.  Stream diagnostics (gate weights, reliability, epistemic, effective, parity_alpha)
  8.  Explainability report (get_explainability_report, routing report)
  9.  GradCAM — generate, generate_pixel_attn_cam, generate_overlay
  10. Error memory summary (recall/failure pattern tracking)
  11. Throughput / latency (images/sec, ms/image)
  12. Confidence distribution analysis (FAKE vs REAL, correct vs wrong)
  13. NaN / degenerate input stability
  14. Single-image inference (the deploy use-case)

Usage:
  # Evaluate against a labelled dataset
  python forensicTest.py \\
      --ckpt_dir  ./checkpoints \\
      --data_dir  /path/to/test_dataset \\
      --batch_size 8 \\
      --workers   2

  # Single-image quick check (no dataset needed)
  python forensicTest.py \\
      --ckpt_dir  ./checkpoints \\
      --image     /path/to/image.jpg

  # Full suite including adversarial + MC uncertainty
  python forensicTest.py \\
      --ckpt_dir  ./checkpoints \\
      --data_dir  /path/to/test_dataset \\
      --adv_eval  --mc_eval --gradcam \\
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
from ForensicsAI import ForensicEngine

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True
warnings.filterwarnings("ignore", category=UserWarning)


# ── Path helper ────────────────────────────────────────────────────────────────
def _add_src(path: str):
    if path and path not in sys.path:
        sys.path.insert(0, path)


# ── ANSI colours ───────────────────────────────────────────────────────────────
GRN = "\033[92m"
YLW = "\033[93m"
RED = "\033[91m"
CYN = "\033[96m"
BLD = "\033[1m"
RST = "\033[0m"

def _ok(msg):   print(f"  {GRN}✅ {msg}{RST}")
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
    Priority: best → final → checkpoint (latest) → any .pt
    """
    d = Path(ckpt_dir)
    if not d.is_dir():
        return None
    candidates = {
        "best":   d / "best.pt",
        "final":  d / "final.pt",
        "latest": d / "checkpoint.pt",
    }
    priority = ["best", "final", "latest"] if prefer == "best" else [prefer, "best", "final", "latest"]
    for key in priority:
        p = candidates.get(key)
        if p and p.exists():
            return str(p)
    pts = sorted(d.glob("*.pt"))
    return str(pts[-1]) if pts else None


def load_model(ckpt_path: Optional[str], device: torch.device, args) -> "ForensicEngine":
    """
    Build ForensicEngine (Stable Edition) via get_forensic_engine() and load
    checkpoint weights with strict=False for forward-compatibility.
    """
    from authenticityEngine import get_forensic_engine, load_checkpoint, ForensicEngine

    model: ForensicEngine = get_forensic_engine(
        freeze_backbone=True,
        n_iters=args.n_iters,
        use_isfcr=True,
        use_fcw=True,
        dropout=0.0,        # deterministic for eval; MC passes override via .train()
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
                _warn(f"Missing keys ({len(missing)}): {missing[:5]}{'…' if len(missing) > 5 else ''}")
            if unexpected:
                _warn(f"Unexpected keys ({len(unexpected)}): {unexpected[:5]}{'…' if len(unexpected) > 5 else ''}")
            epoch = ckpt.get("epoch", "?")
            best  = ckpt.get("best_val_acc", ckpt.get("best_bal_acc", None))
            bal   = f"{best:.4f}" if isinstance(best, float) else str(best)
            print(f"  Loaded checkpoint: epoch={epoch}  best_bal_acc={bal}")
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
        seen: set = set()
        for label, class_dir in _find_class_dirs(Path(root)).items():
            for ext in VALID_EXTS:
                for fpath in list(class_dir.rglob(f'*{ext}')) + list(class_dir.rglob(f'*{ext.upper()}')):
                    key = str(fpath)
                    if key not in seen:
                        seen.add(key)
                        self.samples.append((key, label))
        counts: Dict[int, int] = defaultdict(int)
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
    from authenticityEngine import CLIP_MEAN, CLIP_STD
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
    ds = TestDataset(data_dir, get_test_transform())
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
        if mask.sum() == 0:
            continue
        per_class.append((preds[mask] == labels[mask]).mean())
    return float(np.mean(per_class)) if per_class else 0.0


def per_class_recall(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    out = {}
    for c, name in [(0, "FAKE"), (1, "REAL")]:
        mask = labels == c
        out[name] = float((preds[mask] == c).mean()) if mask.sum() else float('nan')
    return out


def per_class_precision(preds: np.ndarray, labels: np.ndarray) -> Dict[str, float]:
    out = {}
    for c, name in [(0, "FAKE"), (1, "REAL")]:
        mask = preds == c
        out[name] = float((labels[mask] == c).mean()) if mask.sum() else float('nan')
    return out


def f1_score(prec: float, rec: float) -> float:
    if prec + rec == 0:
        return 0.0
    return 2 * prec * rec / (prec + rec)


def roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    """Trapezoidal AUC — no sklearn required."""
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


def expected_calibration_error(
    confs: np.ndarray, corrects: np.ndarray, n_bins: int = 10
) -> Tuple[float, float]:
    """Returns (ECE, MCE)."""
    bins = np.linspace(0, 1, n_bins + 1)
    ece = mce = 0.0
    N = len(confs)
    for i in range(n_bins):
        lo, hi = bins[i], bins[i + 1]
        mask = (confs >= lo) & (confs < hi)
        if not mask.sum():
            continue
        gap = abs(corrects[mask].mean() - confs[mask].mean())
        ece += gap * mask.sum() / N
        mce  = max(mce, gap)
    return ece, mce


# ══════════════════════════════════════════════════════════════════════════════
# 4.  EVALUATION LOOP
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def evaluate(model, loader, device) -> Dict:
    """
    Single pass over the test set using forward_with_streams (the same path
    used during training diagnostics). Collects predictions, confidences,
    stream diagnostics, and feeds the error-learning memory.
    """
    model.eval()
    all_preds, all_labels, all_confs, all_scores = [], [], [], []
    gate_accum:        Dict[str, List[float]] = defaultdict(list)
    epistemic_accum:   Dict[str, List[float]] = defaultdict(list)
    reliability_accum: Dict[str, List[float]] = defaultdict(list)
    effective_accum:   Dict[str, List[float]] = defaultdict(list)
    parity_alphas: List[float] = []
    error_paths: List[Tuple[str, int, int, float]] = []  # path, true, pred, conf

    for imgs, labels, paths in loader:
        imgs   = imgs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        # forward_with_streams: returns (logits, info_dict)
        # info_dict carries _gate_weights, _epistemic, _reliability,
        # _effective, _parity_alpha, stream norms — all from the latest batch
        logits, info = model.forward_with_streams(imgs)
        probs  = torch.softmax(logits, dim=1)
        preds  = probs.argmax(dim=1)
        confs  = probs.max(dim=1).values
        scores = probs[:, 1]   # P(REAL) for AUC

        all_preds.extend(preds.cpu().numpy().tolist())
        all_labels.extend(labels.cpu().numpy().tolist())
        all_confs.extend(confs.cpu().numpy().tolist())
        all_scores.extend(scores.cpu().numpy().tolist())

        # Accumulate stream diagnostics
        for k, v in info.get('_gate_weights', {}).items():
            gate_accum[k].append(float(v))
        for k, v in info.get('_epistemic', {}).items():
            epistemic_accum[k].append(float(v))
        for k, v in info.get('_reliability', {}).items():
            reliability_accum[k].append(float(v))
        for k, v in info.get('_effective', {}).items():
            effective_accum[k].append(float(v))
        parity_alphas.append(info.get('_parity_alpha', float('nan')))

        # Feed error-learning memory: model learns which stream norms correlate
        # with false predictions and adjusts the cosine classifier bias accordingly
        model.step_error_memory(
            labels.cpu(), preds.cpu(), confs.cpu(), info
        )

        # Track misclassified paths for the error report
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
        'preds':        np.array(all_preds),
        'labels':       np.array(all_labels),
        'confs':        np.array(all_confs),
        'scores':       np.array(all_scores),
        'gate_weights': {k: float(np.mean(v)) for k, v in gate_accum.items()},
        'epistemic':    {k: float(np.mean(v)) for k, v in epistemic_accum.items()},
        'reliability':  {k: float(np.mean(v)) for k, v in reliability_accum.items()},
        'effective':    {k: float(np.mean(v)) for k, v in effective_accum.items()},
        'parity_alpha': float(np.nanmean(parity_alphas)) if parity_alphas else float('nan'),
        'error_paths':  error_paths,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 5.  ADVERSARIAL ROBUSTNESS
# ══════════════════════════════════════════════════════════════════════════════

def adversarial_eval(
    model, loader, device,
    fgsm_eps_list=(4/255, 8/255),
    pgd_eps=8/255, pgd_steps=7,
    cw_batches=5,
    max_batches=20,
) -> Dict:
    """
    FGSM + PGD-7 + CW-L2 evaluation.
    AdversarialTester is imported from authenticityEngine — same module as model.
    """
    from authenticityEngine import AdversarialTester
    tester = AdversarialTester(model, device)
    results: Dict = {}

    # ── Clean baseline ────────────────────────────────────────────────────────
    clean_preds, clean_labels = [], []
    for i, (imgs, labels, _) in enumerate(loader):
        if i >= max_batches:
            break
        with torch.no_grad():
            preds = model(imgs.to(device)).argmax(1)
        clean_preds.extend(preds.cpu().numpy())
        clean_labels.extend(labels.numpy())
    clean_bal = balanced_accuracy(np.array(clean_preds), np.array(clean_labels))
    results['clean_bal_acc'] = clean_bal

    # ── FGSM ─────────────────────────────────────────────────────────────────
    for eps in fgsm_eps_list:
        adv_preds, adv_labels = [], []
        for i, (imgs, labels, _) in enumerate(loader):
            if i >= max_batches:
                break
            adv = tester.fgsm(imgs.to(device), labels.to(device), eps=eps)
            with torch.no_grad():
                preds = model(adv).argmax(1)
            adv_preds.extend(preds.cpu().numpy())
            adv_labels.extend(labels.numpy())
        bal = balanced_accuracy(np.array(adv_preds), np.array(adv_labels))
        key = f"fgsm_{int(eps * 255)}px255"
        results[key]           = bal
        results[f"{key}_drop"] = clean_bal - bal

    # ── PGD-7 ─────────────────────────────────────────────────────────────────
    pgd_preds, pgd_labels = [], []
    for i, (imgs, labels, _) in enumerate(loader):
        if i >= max_batches:
            break
        adv = tester.pgd(imgs.to(device), labels.to(device),
                         eps=pgd_eps, alpha=2/255, steps=pgd_steps)
        with torch.no_grad():
            preds = model(adv).argmax(1)
        pgd_preds.extend(preds.cpu().numpy())
        pgd_labels.extend(labels.numpy())
    pgd_bal = balanced_accuracy(np.array(pgd_preds), np.array(pgd_labels))
    results['pgd7_bal_acc'] = pgd_bal
    results['pgd7_drop']    = clean_bal - pgd_bal

    # ── CW-L2 (light eval, few batches) ──────────────────────────────────────
    cw_preds, cw_labels = [], []
    for i, (imgs, labels, _) in enumerate(loader):
        if i >= cw_batches:
            break
        adv = tester.cw_l2(imgs.to(device), labels.to(device))
        with torch.no_grad():
            preds = model(adv).argmax(1)
        cw_preds.extend(preds.cpu().numpy())
        cw_labels.extend(labels.numpy())
    if cw_preds:
        cw_bal = balanced_accuracy(np.array(cw_preds), np.array(cw_labels))
        results['cw_l2_bal_acc'] = cw_bal
        results['cw_l2_drop']    = clean_bal - cw_bal

    return results


# ══════════════════════════════════════════════════════════════════════════════
# 6.  MC-DROPOUT UNCERTAINTY
# ══════════════════════════════════════════════════════════════════════════════

def mc_uncertainty_eval(
    model, loader, device,
    n_passes: int = 10,
    max_batches: int = 20,
) -> Dict:
    """
    Runs predict_with_uncertainty (MC-Dropout) and measures whether the model
    is correctly more uncertain on its mistakes than on its correct predictions.
    A healthy separation ratio > 2.0 means uncertainty is a useful signal.
    """
    from authenticityEngine import predict_with_uncertainty

    unc_correct: List[float] = []
    unc_wrong:   List[float] = []

    for i, (imgs, labels, _) in enumerate(loader):
        if i >= max_batches:
            break
        result = predict_with_uncertainty(model, imgs.to(device), n_passes=n_passes)
        preds = result['prediction'].cpu().numpy()
        uncs  = result['uncertainty'].cpu().numpy()
        for p, u, l in zip(preds, uncs, labels.numpy()):
            (unc_correct if p == l else unc_wrong).append(float(u))

    mean_uc = float(np.mean(unc_correct)) if unc_correct else float('nan')
    mean_uw = float(np.mean(unc_wrong))   if unc_wrong   else float('nan')
    return {
        'mean_uncertainty_correct': mean_uc,
        'mean_uncertainty_wrong':   mean_uw,
        'n_correct':         len(unc_correct),
        'n_wrong':           len(unc_wrong),
        'separation_ratio':  (mean_uw / (mean_uc + 1e-8)
                              if unc_correct and unc_wrong else float('nan')),
    }


# ══════════════════════════════════════════════════════════════════════════════
# 7.  THROUGHPUT / LATENCY
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def throughput_test(
    model, device,
    input_size: int = 224,
    batch_size: int = 8,
    n_warmup: int = 5,
    n_runs: int = 20,
) -> Dict:
    """Measures forward-pass throughput on dummy data."""
    dummy = torch.randn(batch_size, 3, input_size, input_size, device=device)
    model.eval()
    for _ in range(n_warmup):
        model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()

    t0 = time.perf_counter()
    for _ in range(n_runs):
        model(dummy)
    if device.type == 'cuda':
        torch.cuda.synchronize()
    elapsed = time.perf_counter() - t0

    total_images = batch_size * n_runs
    return {
        'images_per_sec': round(total_images / elapsed, 1),
        'ms_per_image':   round(1000.0 * elapsed / total_images, 3),
        'batch_size':     batch_size,
        'n_runs':         n_runs,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 8.  GRADCAM
# ══════════════════════════════════════════════════════════════════════════════

def gradcam_test(model, device) -> bool:
    """
    Tests all three GradCAM surfaces from authenticityEngine.GradCAM:
      - generate()                 → multi-scale weighted freq-CAM
      - generate_pixel_attn_cam()  → SRM pixel attention heatmap
      - generate_overlay()         → fused freq + pixel CAM
    All outputs must be (224, 224) tensors with values in [0, 1].
    """
    from authenticityEngine import GradCAM
    dummy = torch.randn(1, 3, 224, 224, device=device)
    ok = True

    try:
        cam = GradCAM(model)

        # 1. Multi-scale freq-CAM
        heatmap = cam.generate(dummy, class_idx=1)
        assert heatmap.shape == (224, 224), f"generate() shape={heatmap.shape}"
        assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0, "generate() out of [0,1]"
        _ok("generate() → 224×224 freq-CAM ✓")

        # 2. Pixel attention CAM (SRM PixelFrequencyHead)
        pixel_cam = cam.generate_pixel_attn_cam(dummy)
        assert pixel_cam.shape == (224, 224), f"generate_pixel_attn_cam() shape={pixel_cam.shape}"
        _ok("generate_pixel_attn_cam() → 224×224 pixel-attn-CAM ✓")

        # 3. Fused overlay (freq + pixel)
        overlay = cam.generate_overlay(dummy, class_idx=1)
        assert 'freq_cam'  in overlay, "generate_overlay() missing freq_cam"
        assert 'pixel_cam' in overlay, "generate_overlay() missing pixel_cam"
        assert 'fused_cam' in overlay, "generate_overlay() missing fused_cam"
        assert overlay['fused_cam'].shape == (224, 224), \
            f"generate_overlay() fused shape={overlay['fused_cam'].shape}"
        _ok("generate_overlay() → freq_cam + pixel_cam + fused_cam ✓")

    except Exception as e:
        _fail(f"GradCAM error: {e}")
        ok = False

    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 9.  NaN / DEGENERATE INPUT STABILITY
# ══════════════════════════════════════════════════════════════════════════════

@torch.no_grad()
def nan_stability_test(model, device) -> bool:
    """
    Feeds NaN, zero, and +Inf tensors through the model.
    authenticityEngine._forward_core() calls torch.nan_to_num() on input before
    any computation, so all three must produce finite logits without crashing.
    """
    inputs = [
        ("NaN",  torch.full((1, 3, 224, 224), float('nan'), device=device)),
        ("Zero", torch.zeros((1, 3, 224, 224), device=device)),
        ("Inf",  torch.full((1, 3, 224, 224), float('inf'), device=device)),
    ]
    ok = True
    for name, x in inputs:
        try:
            out = model(x)
            if not torch.isfinite(out).all():
                _fail(f"{name} input → non-finite logits: {out}")
                ok = False
            else:
                _ok(f"{name} input → finite logits {[round(v, 3) for v in out[0].tolist()]}")
        except Exception as e:
            _fail(f"{name} input crashed model: {e}")
            ok = False
    return ok


# ══════════════════════════════════════════════════════════════════════════════
# 10. SINGLE-IMAGE INFERENCE
# ══════════════════════════════════════════════════════════════════════════════

def single_image_test(model, image_path: str, device: torch.device) -> Dict:
    """
    Runs the full deploy pipeline on one image:
      model.predict()                 → prediction, confidence, uncertainty, stream_importance
      model.get_explainability_report() → gate weights, epistemic, routing report
    """
    tf = get_test_transform()
    try:
        img_pil = Image.open(image_path).convert('RGB')
    except Exception as e:
        return {'error': str(e)}

    tensor = tf(img_pil).unsqueeze(0).to(device)
    model.eval()

    result = model.predict(tensor)
    expl   = model.get_explainability_report(tensor)

    pred_idx = int(result['prediction'][0].item())
    conf     = float(result['confidence'][0].item())
    unc      = float(result['uncertainty'])

    return {
        'image':                 image_path,
        'prediction':            "REAL" if pred_idx == 1 else "FAKE / GAN",
        'prediction_idx':        pred_idx,
        'confidence':            conf,
        'uncertainty':           unc,
        'stream_importance':     result['stream_importance'],
        'stream_reliability':    result['stream_reliability'],
        'contradiction_strength': result['contradiction_strength'],
        'explainability':        expl,
    }


# ══════════════════════════════════════════════════════════════════════════════
# 11. CONFIDENCE DISTRIBUTION ANALYSIS
# ══════════════════════════════════════════════════════════════════════════════

def confidence_analysis(
    preds: np.ndarray, labels: np.ndarray, confs: np.ndarray
) -> Dict:
    def _stats(arr):
        if not len(arr):
            return {'mean': float('nan'), 'std': float('nan'),
                    'p10': float('nan'), 'p90': float('nan')}
        return {
            'mean': float(np.mean(arr)),
            'std':  float(np.std(arr)),
            'p10':  float(np.percentile(arr, 10)),
            'p90':  float(np.percentile(arr, 90)),
        }

    mask_fake = labels == 0
    mask_real = labels == 1
    correct   = preds == labels
    wrong     = ~correct

    return {
        'FAKE_correct':    _stats(confs[mask_fake & correct]),
        'FAKE_wrong':      _stats(confs[mask_fake & wrong]),
        'REAL_correct':    _stats(confs[mask_real & correct]),
        'REAL_wrong':      _stats(confs[mask_real & wrong]),
        'overall_correct': _stats(confs[correct]),
        'overall_wrong':   _stats(confs[wrong]),
        'high_conf_errors': int(((confs > 0.9) & wrong).sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# PRINTING / REPORTING
# ══════════════════════════════════════════════════════════════════════════════

def _fmt(v, pct=False, dec=2):
    if isinstance(v, float) and np.isnan(v):
        return "N/A"
    return f"{v * 100:.{dec}f}%" if pct else f"{v:.{dec}f}"


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

    acc      = (preds == labels).mean()
    bal_acc  = balanced_accuracy(preds, labels)
    recall   = per_class_recall(preds, labels)
    prec     = per_class_precision(preds, labels)
    auc      = roc_auc(labels, scores)
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
    print(f"  {'─' * 46}")
    for cls in ['FAKE', 'REAL']:
        p = prec.get(cls, float('nan'))
        r = recall.get(cls, float('nan'))
        f = f1_score(p, r) if not (np.isnan(p) or np.isnan(r)) else float('nan')
        print(f"  {cls:<10}  {_fmt(p, pct=True):>10}  {_fmt(r, pct=True):>10}  {_fmt(f, pct=True):>10}")

    _hdr("CALIBRATION")
    print(f"  Expected Calibration Error (ECE) : {_fmt(ece, dec=4)}")
    print(f"  Maximum Calibration Error  (MCE) : {_fmt(mce, dec=4)}")
    if   ece < 0.05: _ok("Well-calibrated (ECE < 5%)")
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
    _sub("Gate weights (influence on final decision) — descending:")
    for k, v in sorted(res['gate_weights'].items(), key=lambda x: -x[1]):
        bar = '█' * max(1, int(v * 30))
        print(f"    {k:<18}: {v:.4f}  {bar}")

    if res.get('effective'):
        _sub("\nEffective contribution (gate × evidential reliability) — descending:")
        for k, v in sorted(res['effective'].items(), key=lambda x: -x[1]):
            bar = '█' * max(1, int(v * 30))
            print(f"    {k:<18}: {v:.4f}  {bar}")

    print(f"\n  Parity alpha (forensic authority): {res['parity_alpha']:.4f}"
          f"  [0.3=semantic-led ↔ 0.7=forensic-led]")

    _sub("\nStream epistemic uncertainty (lower = more confident):")
    for k, v in sorted(res['epistemic'].items()):
        color = GRN if v < 0.1 else (YLW if v < 0.3 else RED)
        print(f"    {k:<18}: {color}{v:.4f}{RST}")

    _sub("\nStream reliability (higher = more trustworthy):")
    for k, v in sorted(res['reliability'].items(), key=lambda x: -x[1]):
        color = GRN if v > 0.7 else (YLW if v > 0.4 else RED)
        print(f"    {k:<18}: {color}{v:.4f}{RST}")

    _hdr("ERRORS")
    print(f"  Total misclassified : {len(res['error_paths'])}")
    if res['error_paths']:
        print("  Worst 5 (highest confidence wrong predictions):")
        worst = sorted(res['error_paths'], key=lambda x: -x[3])[:5]
        for path, true_l, pred_l, cf in worst:
            tname = "REAL" if true_l == 1 else "FAKE"
            pname = "REAL" if pred_l == 1 else "FAKE"
            print(f"    [{tname}→{pname}] conf={cf:.3f}  {Path(path).name}")


def print_adv_results(adv: Dict):
    _hdr("ADVERSARIAL ROBUSTNESS")
    clean = adv.get('clean_bal_acc', float('nan'))
    print(f"  Clean bal_acc  : {_fmt(clean, pct=True)}")
    print()
    attack_order = [
        ('fgsm_4px255',    'fgsm_4px255_drop'),
        ('fgsm_8px255',    'fgsm_8px255_drop'),
        ('pgd7_bal_acc',   'pgd7_drop'),
        ('cw_l2_bal_acc',  'cw_l2_drop'),
    ]
    for bal_key, drop_key in attack_order:
        if bal_key not in adv:
            continue
        bal  = adv[bal_key]
        drop = adv.get(drop_key, float('nan'))
        color = GRN if drop < 0.05 else (YLW if drop < 0.15 else RED)
        label = bal_key.replace('_bal_acc', '').replace('px255', '/255')
        print(f"  {label:<14} bal_acc={_fmt(bal, pct=True)}  "
              f"drop={color}{_fmt(drop, pct=True)}{RST}")
    print()
    worst_drop = max(
        (adv.get(k, 0) for k in ['fgsm_8px255_drop', 'pgd7_drop', 'cw_l2_drop']),
        default=0.0
    )
    if   worst_drop < 0.05:  _ok("ROBUST — worst-case bal_acc drop < 5%")
    elif worst_drop < 0.15:  _warn("MODERATE — worst-case drop 5–15%")
    else:                     _fail("VULNERABLE — worst-case drop > 15%")


def print_mc_results(mc: Dict):
    _hdr("MC-DROPOUT UNCERTAINTY")
    r = mc.get('separation_ratio', float('nan'))
    print(f"  Mean uncertainty (correct preds) : {_fmt(mc['mean_uncertainty_correct'], dec=4)}")
    print(f"  Mean uncertainty (wrong   preds) : {_fmt(mc['mean_uncertainty_wrong'],   dec=4)}")
    print(f"  Uncertainty separation ratio     : {_fmt(r, dec=2)}×")
    if not np.isnan(r):
        if   r > 2.0: _ok("Model is more uncertain on its mistakes (healthy)")
        elif r > 1.2: _warn("Weak uncertainty separation")
        else:          _fail("Model is NOT more uncertain on wrong predictions")


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

    print()
    if is_real:
        print(f"{GRN}{BLD}  ✅  VERDICT : REAL IMAGE{RST}")
    else:
        print(f"{RED}{BLD}  🚨  VERDICT : FAKE / GAN IMAGE{RST}")
    print()

    bar_len = 40
    filled  = int(conf * bar_len)
    bar     = (GRN if is_real else RED) + '█' * filled + RST + '░' * (bar_len - filled)
    print(f"  Confidence  : [{bar}] {conf * 100:.1f}%")

    unc_label = "Low (confident)" if unc < 0.05 else ("Medium" if unc < 0.15 else "High (uncertain)")
    print(f"  Uncertainty : {unc:.4f}  ({unc_label})")
    print(f"  Image       : {res['image']}")
    print()

    _sub("  Stream votes (which forensic signals drove the decision):")
    for k, v in sorted(res['stream_importance'].items(), key=lambda x: -x[1]):
        bar_s = '█' * max(1, int(v * 30))
        print(f"    {k:<18}: {v:.3f}  {bar_s}")

    print()
    _sub("  Stream reliability (trust score per stream):")
    for k, v in sorted(res['stream_reliability'].items(), key=lambda x: -x[1]):
        color = GRN if v > 0.7 else (YLW if v > 0.4 else RED)
        print(f"    {k:<18}: {color}{v:.3f}{RST}")

    print()
    contr = res['contradiction_strength']
    contr_label = (
        "Low — streams agree" if contr < 0.05 else
        ("Medium — some disagreement" if contr < 0.15 else
         "High — streams contradict each other")
    )
    print(f"  Stream contradiction: {contr:.4f}  ({contr_label})")


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def parse_args():
    p = argparse.ArgumentParser(description="ForensicEngine (Stable Edition) Test Suite")
    p.add_argument("--ckpt_dir",    default="./checkpoints",
                   help="Directory containing best.pt / final.pt / checkpoint.pt")
    p.add_argument("--ckpt_file",   default=None,
                   help="Explicit checkpoint path (overrides --ckpt_dir search)")
    p.add_argument("--data_dir",    default=None,
                   help="Test dataset root with real/ and fake/ subdirs")
    p.add_argument("--image",       default=None,
                   help="Single image path for quick inference")
    p.add_argument("--src_dir",     default=".",
                   help="Directory containing authenticityEngine.py (default: current dir)")
    p.add_argument("--batch_size",  type=int, default=8)
    p.add_argument("--workers",     type=int, default=2)
    p.add_argument("--n_iters",     type=int, default=2,
                   help="ISFCR reasoning iterations (must match training value)")
    p.add_argument("--adv_eval",    action="store_true",
                   help="Run FGSM + PGD-7 + CW-L2 adversarial evaluation")
    p.add_argument("--adv_batches", type=int, default=20,
                   help="Max batches for FGSM/PGD adversarial eval")
    p.add_argument("--cw_batches",  type=int, default=5,
                   help="Max batches for CW-L2 (slower attack, fewer batches)")
    p.add_argument("--mc_eval",     action="store_true",
                   help="Run MC-Dropout uncertainty evaluation")
    p.add_argument("--mc_passes",   type=int, default=10,
                   help="MC-Dropout forward passes per image")
    p.add_argument("--mc_batches",  type=int, default=20)
    p.add_argument("--gradcam",     action="store_true",
                   help="Run GradCAM sanity check (generate, pixel_attn_cam, overlay)")
    p.add_argument("--save_json",   default=None,
                   help="Save full results dict to this JSON path")
    p.add_argument("--seed",        type=int, default=42)
    return p.parse_args()


def main():
    args = parse_args()
    _add_src(args.src_dir)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\n{BLD}ForensicEngine — Test Suite (Stable Edition){RST}  |  device={device}")
    if device.type == 'cuda':
        prop = torch.cuda.get_device_properties(0)
        print(f"  GPU: {prop.name}  {prop.total_memory / 1e9:.1f} GB")

    # ── Checkpoint ────────────────────────────────────────────────────────────
    _hdr("CHECKPOINT LOADING")
    ckpt_path = args.ckpt_file or resolve_checkpoint(args.ckpt_dir)
    if ckpt_path:
        print(f"  Using: {ckpt_path}")
    else:
        _warn("No checkpoint found — all results use random weights.")

    model = load_model(ckpt_path, device, args)
    total  = sum(p.numel() for p in model.parameters())
    train_ = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"  Parameters: total={total/1e6:.1f}M  trainable={train_/1e6:.1f}M")
    _ok("Model loaded")

    all_results: Dict = {}

    # ── Stability ─────────────────────────────────────────────────────────────
    _hdr("STABILITY TESTS (NaN / Zero / Inf inputs)")
    stable = nan_stability_test(model, device)
    if stable:
        _ok("All degenerate inputs produced finite logits")
    else:
        _fail("Stability test FAILED — model may crash on corrupted images")
    all_results['stability_ok'] = stable

    # ── Throughput ────────────────────────────────────────────────────────────
    tp = throughput_test(model, device, batch_size=args.batch_size)
    print_throughput_results(tp)
    all_results['throughput'] = tp

    # ── GradCAM ───────────────────────────────────────────────────────────────
    if args.gradcam:
        _hdr("GRADCAM SANITY CHECK")
        ok = gradcam_test(model, device)
        if ok:
            _ok("All GradCAM surfaces (freq, pixel_attn, overlay) produce 224×224 heatmaps")
        all_results['gradcam_ok'] = ok

    # ── Single image ──────────────────────────────────────────────────────────
    if args.image:
        res = single_image_test(model, args.image, device)
        print_single_image_result(res)
        all_results['single_image'] = {k: v for k, v in res.items()
                                        if k != 'explainability'}

    # ── Dataset evaluation ────────────────────────────────────────────────────
    if args.data_dir:
        _hdr("LOADING TEST DATASET")
        loader = build_loader(args.data_dir, args.batch_size, args.workers)

        _hdr("RUNNING EVALUATION")
        t0  = time.time()
        res = evaluate(model, loader, device)
        elapsed = time.time() - t0
        n = len(res['preds'])
        print(f"  Evaluated {n} images in {elapsed:.1f}s  ({n/elapsed:.0f} img/s)")
        print_dataset_results(res)

        all_results['metrics'] = {
            'accuracy':          float((res['preds'] == res['labels']).mean()),
            'balanced_accuracy': float(balanced_accuracy(res['preds'], res['labels'])),
            'auc_roc':           float(roc_auc(res['labels'], res['scores'])),
            'recall':            per_class_recall(res['preds'], res['labels']),
            'precision':         per_class_precision(res['preds'], res['labels']),
            'gate_weights':      res['gate_weights'],
            'effective':         res['effective'],
            'parity_alpha':      res['parity_alpha'],
        }

        # ── Error memory ──────────────────────────────────────────────────────
        _hdr("ERROR MEMORY SUMMARY")
        err = model.error_memory.summary()
        print(f"  Total errors        : {err.get('total_errors', '?')}")
        print(f"  Total successes     : {err.get('total_successes', '?')}")
        print(f"  real_as_fake        : {err.get('real_as_fake', '?')}")
        print(f"  fake_as_real        : {err.get('fake_as_real', '?')}")
        print(f"  ema_fake_err_rate   : {err.get('ema_fake_err_rate', 0):.4f}")
        print(f"  ema_real_err_rate   : {err.get('ema_real_err_rate', 0):.4f}")
        print(f"  avg_conf_real_err   : {err.get('avg_conf_real_err', 0):.4f}  (high = overconfident on real)")
        print(f"  avg_conf_fake_err   : {err.get('avg_conf_fake_err', 0):.4f}  (high = overconfident on fake)")
        if err.get('top_error_patterns'):
            _sub("  Top error patterns:")
            for pat, cnt in err['top_error_patterns'].items():
                print(f"    {pat:<40}: {cnt}")
        all_results['error_memory'] = {k: v for k, v in err.items()
                                        if isinstance(v, (int, float))}

        # ── Stream routing report ─────────────────────────────────────────────
        _hdr("STREAM ROUTING REPORT")
        print(model.get_routing_report())
        low = model.get_low_importance_streams(threshold=0.03)
        if low:
            _warn(f"Low-contribution streams (< 3% effective): {low}")
        else:
            _ok("All streams contributing above threshold")

        # ── MC uncertainty ────────────────────────────────────────────────────
        if args.mc_eval:
            mc = mc_uncertainty_eval(
                model, loader, device,
                n_passes=args.mc_passes,
                max_batches=args.mc_batches,
            )
            print_mc_results(mc)
            all_results['mc_uncertainty'] = mc

        # ── Adversarial ───────────────────────────────────────────────────────
        if args.adv_eval:
            _hdr("ADVERSARIAL ROBUSTNESS (FGSM + PGD-7 + CW-L2) — running…")
            adv = adversarial_eval(
                model, loader, device,
                max_batches=args.adv_batches,
                cw_batches=args.cw_batches,
            )
            print_adv_results(adv)
            all_results['adversarial'] = adv

    # ── Save JSON ─────────────────────────────────────────────────────────────
    if args.save_json:
        with open(args.save_json, 'w') as f:
            json.dump(all_results, f, indent=2, default=str)
        _ok(f"Results saved → {args.save_json}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'═' * 70}")
    print(f"  {BLD}TEST COMPLETE{RST}")
    if 'metrics' in all_results:
        m     = all_results['metrics']
        b     = m['balanced_accuracy']
        a     = m['auc_roc']
        r_real = m['recall'].get('REAL', float('nan'))
        r_fake = m['recall'].get('FAKE', float('nan'))
        print(f"  Balanced Acc : {_fmt(b, pct=True)}  AUC : {_fmt(a, dec=4)}  "
              f"REAL-Recall : {_fmt(r_real, pct=True)}  FAKE-Recall : {_fmt(r_fake, pct=True)}")
        print(f"  Verdict      : {_verdict(b)}")
    print(f"{'═' * 70}\n")


if __name__ == '__main__':
    main()
