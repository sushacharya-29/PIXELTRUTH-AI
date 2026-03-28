"""
Production Inference Script v5.0 — AI Image Detection
======================================================

FIXES OVER v4.x:

  ✅ FIX #7 (Naive 0.5/0.5 ensemble): DYNAMIC ensemble weighting.
     FrequencyModelV5 outputs a CONFIDENCE SCORE.
     When frequency model is uncertain → trust semantic model more.
     When both agree → amplify confidence.

  ✅ FIX #9 (Calibration): Uses calibrated models.
     Temperature was tuned on validation set — confidence scores are RELIABLE.

  ✅ FIX #10 (No semantics): Now explains WHY an image is predicted fake/real.
     (Dominant signal: semantic/physics/frequency)

USAGE:
  # Single image (spatial only):
  python v5inference.py --image photo.jpg \
      --spatial_weights checkpoints_v5/spatial_model_v5_calibrated.pth

  # Full ensemble (recommended):
  python v5inference.py --image photo.jpg \
      --spatial_weights checkpoints_v5/spatial_model_v5_calibrated.pth \
      --frequency_weights checkpoints_v5/frequency_model_v5_calibrated.pth \
      --ensemble

  # Batch:
  python v5inference.py --image_dir my_images/ \
      --spatial_weights checkpoints_v5/spatial_model_v5_calibrated.pth \
      --frequency_weights checkpoints_v5/frequency_model_v5_calibrated.pth \
      --ensemble --output_csv results.csv
"""

import os
import argparse
from pathlib import Path
from typing import Dict, Optional, List

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image
import numpy as np

from v5spatial import get_spatial_model_v5, CLIP_MEAN, CLIP_STD
from v5frequency import get_frequency_model_v5


# ══════════════════════════════════════════════════════════════
# Transforms
# ══════════════════════════════════════════════════════════════

def get_spatial_inference_transform(image_size: int = 224) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),  # CLIP normalization
    ])


def get_frequency_inference_transform(image_size: int = 128) -> T.Compose:
    return T.Compose([
        T.Resize((image_size, image_size)),
        T.ToTensor(),  # Raw [0,1] — no normalization for frequency model
    ])


# ══════════════════════════════════════════════════════════════
# Single Model Prediction
# ══════════════════════════════════════════════════════════════

def predict_spatial(
    image_path: str,
    model: nn.Module,
    transform: T.Compose,
    device: torch.device,
) -> Dict:
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits = model(tensor)
        probs = torch.softmax(logits, dim=1)[0]

    fake_p = probs[0].item()
    real_p = probs[1].item()
    prediction = 'FAKE' if fake_p > real_p else 'REAL'
    confidence = max(fake_p, real_p) * 100

    return {
        'fake_prob': fake_p,
        'real_prob': real_p,
        'prediction': prediction,
        'confidence': confidence,
        'signal': 'semantic+physics'
    }


def predict_frequency(
    image_path: str,
    model: nn.Module,
    transform: T.Compose,
    device: torch.device,
) -> Dict:
    img = Image.open(image_path).convert('RGB')
    tensor = transform(img).unsqueeze(0).to(device)

    model.eval()
    with torch.no_grad():
        logits, freq_confidence = model(tensor, return_confidence=True)
        probs = torch.softmax(logits, dim=1)[0]

    fake_p = probs[0].item()
    real_p = probs[1].item()
    prediction = 'FAKE' if fake_p > real_p else 'REAL'
    confidence = max(fake_p, real_p) * 100
    freq_conf_score = freq_confidence[0].item()  # 0-1, how confident the freq model is

    return {
        'fake_prob': fake_p,
        'real_prob': real_p,
        'prediction': prediction,
        'confidence': confidence,
        'freq_confidence_score': freq_conf_score,
        'signal': 'frequency+upsampling'
    }


# ══════════════════════════════════════════════════════════════
# Dynamic Ensemble (FIX #7)
# ══════════════════════════════════════════════════════════════

def dynamic_ensemble_predict(
    image_path: str,
    spatial_model: nn.Module,
    freq_model: nn.Module,
    spatial_transform: T.Compose,
    freq_transform: T.Compose,
    device: torch.device,
) -> Dict:
    """
    DYNAMIC ensemble — frequency model's confidence score determines its weight.

    Logic:
    - If frequency model is HIGH confidence → it caught something real → weight it more
    - If frequency model is LOW confidence → modern AI cleaned freq artifacts → trust semantic
    - Both agree → amplify confidence
    - They disagree → defer to semantic (more reliable on modern AI)
    """
    spatial_result = predict_spatial(image_path, spatial_model, spatial_transform, device)
    freq_result    = predict_frequency(image_path, freq_model, freq_transform, device)

    freq_conf_score = freq_result['freq_confidence_score']  # 0-1

    # Dynamic weighting: frequency gets weight proportional to its own confidence
    # Minimum 10%, maximum 40% for frequency (semantic is always dominant)
    freq_weight    = 0.10 + 0.30 * freq_conf_score  # [0.10, 0.40]
    spatial_weight = 1.0 - freq_weight

    # Weighted average of probabilities
    fake_prob = (spatial_weight * spatial_result['fake_prob'] +
                 freq_weight    * freq_result['fake_prob'])
    real_prob = (spatial_weight * spatial_result['real_prob'] +
                 freq_weight    * freq_result['real_prob'])

    prediction = 'FAKE' if fake_prob > real_prob else 'REAL'
    confidence = max(fake_prob, real_prob) * 100

    # Determine dominant signal for explainability
    if freq_conf_score > 0.75 and freq_result['prediction'] == prediction:
        dominant = 'frequency artifacts (upsampling patterns detected)'
    elif abs(fake_prob - real_prob) > 0.4:
        dominant = 'semantic/physics anomalies (strong signal)'
    elif spatial_result['prediction'] != freq_result['prediction']:
        dominant = 'semantic model (frequency model disagreed — lower weight given)'
    else:
        dominant = 'both models agree (high confidence)'

    # Agreement flag
    models_agree = spatial_result['prediction'] == freq_result['prediction']
    if models_agree:
        # Amplify confidence when both agree (consensus boost)
        confidence = min(99.0, confidence * 1.05)

    return {
        'fake_prob': fake_prob,
        'real_prob': real_prob,
        'prediction': prediction,
        'confidence': confidence,
        'dominant_signal': dominant,
        'models_agree': models_agree,
        'spatial_pred': spatial_result['prediction'],
        'spatial_conf': spatial_result['confidence'],
        'frequency_pred': freq_result['prediction'],
        'frequency_conf': freq_result['confidence'],
        'frequency_self_confidence': freq_conf_score,
        'spatial_weight': spatial_weight,
        'frequency_weight': freq_weight,
    }


# ══════════════════════════════════════════════════════════════
# Batch Processing
# ══════════════════════════════════════════════════════════════

def process_directory(
    image_dir: str,
    spatial_model: Optional[nn.Module],
    freq_model: Optional[nn.Module],
    spatial_transform: T.Compose,
    freq_transform: T.Compose,
    device: torch.device,
    use_ensemble: bool = True,
    output_csv: Optional[str] = None,
) -> List[Dict]:
    extensions = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
    paths = []
    for ext in extensions:
        paths.extend(Path(image_dir).rglob(f'*{ext}'))
        paths.extend(Path(image_dir).rglob(f'*{ext.upper()}'))

    print(f"\nFound {len(paths)} images")
    results = []

    for img_path in paths:
        try:
            if use_ensemble and spatial_model and freq_model:
                r = dynamic_ensemble_predict(
                    str(img_path), spatial_model, freq_model,
                    spatial_transform, freq_transform, device
                )
            elif spatial_model:
                r = predict_spatial(str(img_path), spatial_model, spatial_transform, device)
            else:
                r = predict_frequency(str(img_path), freq_model, freq_transform, device)

            r['image_path'] = str(img_path)
            results.append(r)
            agree_str = ' [AGREE]' if r.get('models_agree', True) else ' [DISAGREE]'
            print(f"  {img_path.name}: {r['prediction']} ({r['confidence']:.1f}%){agree_str}")

        except Exception as e:
            print(f"  Error: {img_path.name}: {e}")

    # CSV export
    if output_csv and results:
        import csv
        flat_results = []
        for r in results:
            flat = {k: v for k, v in r.items() if not isinstance(v, dict)}
            flat_results.append(flat)
        with open(output_csv, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_results[0].keys())
            writer.writeheader()
            writer.writerows(flat_results)
        print(f"\n✓ Results saved to {output_csv}")

    # Summary
    if results:
        fake_n = sum(1 for r in results if r['prediction'] == 'FAKE')
        real_n = len(results) - fake_n
        avg_conf = np.mean([r['confidence'] for r in results])
        print(f"\n{'='*60}")
        print(f"  Total: {len(results)} | FAKE: {fake_n} ({100*fake_n/len(results):.1f}%) | "
              f"REAL: {real_n} ({100*real_n/len(results):.1f}%)")
        print(f"  Avg confidence: {avg_conf:.1f}%")
        print(f"{'='*60}")

    return results


# ══════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(description='AI Image Detection Inference v5.0')
    parser.add_argument('--image', type=str)
    parser.add_argument('--image_dir', type=str)
    parser.add_argument('--spatial_weights', type=str)
    parser.add_argument('--frequency_weights', type=str)
    parser.add_argument('--ensemble', action='store_true')
    parser.add_argument('--output_csv', type=str)
    parser.add_argument('--spatial_size', type=int, default=224)
    parser.add_argument('--freq_size', type=int, default=128)
    parser.add_argument('--temperature', type=float, default=1.0)
    args = parser.parse_args()

    if not args.image and not args.image_dir:
        parser.error("Must specify --image or --image_dir")
    if not args.spatial_weights and not args.frequency_weights:
        parser.error("Must specify at least one model")
    if args.ensemble and (not args.spatial_weights or not args.frequency_weights):
        parser.error("Ensemble requires both --spatial_weights and --frequency_weights")

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nDevice: {device}")

    spatial_model = freq_model = None

    if args.spatial_weights:
        print(f"Loading spatial model (CLIP-semantic)...")
        spatial_model = get_spatial_model_v5(temperature=args.temperature).to(device)
        ckpt = torch.load(args.spatial_weights, map_location=device, weights_only=False)
        spatial_model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        spatial_model.eval()
        print("✓ Spatial model loaded")

    if args.frequency_weights:
        print(f"Loading frequency model (upsampling detector)...")
        freq_model = get_frequency_model_v5(temperature=args.temperature).to(device)
        ckpt = torch.load(args.frequency_weights, map_location=device, weights_only=False)
        freq_model.load_state_dict(ckpt.get('model_state_dict', ckpt), strict=False)
        freq_model.eval()
        print("✓ Frequency model loaded")

    spatial_transform = get_spatial_inference_transform(args.spatial_size)
    freq_transform    = get_frequency_inference_transform(args.freq_size)

    if args.image:
        print(f"\nAnalyzing: {args.image}")

        if args.ensemble:
            r = dynamic_ensemble_predict(
                args.image, spatial_model, freq_model,
                spatial_transform, freq_transform, device
            )
            print(f"\n{'='*60}")
            print(f"  v5 DYNAMIC ENSEMBLE PREDICTION")
            print(f"{'='*60}")
            print(f"  Result:     {r['prediction']}")
            print(f"  Confidence: {r['confidence']:.1f}%")
            print(f"  FAKE prob:  {r['fake_prob']*100:.1f}%")
            print(f"  REAL prob:  {r['real_prob']*100:.1f}%")
            print(f"\n  Dominant signal: {r['dominant_signal']}")
            print(f"  Models agree:    {'YES ✓' if r['models_agree'] else 'NO ⚠'}")
            print(f"\n  Breakdown:")
            print(f"    Semantic/Physics: {r['spatial_pred']} ({r['spatial_conf']:.1f}%)  "
                  f"[weight: {r['spatial_weight']:.2f}]")
            print(f"    Frequency:        {r['frequency_pred']} ({r['frequency_conf']:.1f}%)  "
                  f"[weight: {r['frequency_weight']:.2f}]  "
                  f"[self-confidence: {r['frequency_self_confidence']:.2f}]")
            print(f"{'='*60}\n")

        elif spatial_model:
            r = predict_spatial(args.image, spatial_model, spatial_transform, device)
            print(f"\n{'='*60}")
            print(f"  v5 SEMANTIC MODEL PREDICTION")
            print(f"{'='*60}")
            print(f"  Result:     {r['prediction']}")
            print(f"  Confidence: {r['confidence']:.1f}%")
            print(f"  FAKE:       {r['fake_prob']*100:.1f}%")
            print(f"  REAL:       {r['real_prob']*100:.1f}%")
            print(f"  Signal:     {r['signal']}")
            print(f"{'='*60}\n")

        else:
            r = predict_frequency(args.image, freq_model, freq_transform, device)
            print(f"\n{'='*60}")
            print(f"  v5 FREQUENCY MODEL PREDICTION")
            print(f"{'='*60}")
            print(f"  Result:            {r['prediction']}")
            print(f"  Confidence:        {r['confidence']:.1f}%")
            print(f"  FAKE:              {r['fake_prob']*100:.1f}%")
            print(f"  REAL:              {r['real_prob']*100:.1f}%")
            print(f"  Model confidence:  {r['freq_confidence_score']:.2f}")
            print(f"{'='*60}\n")

    elif args.image_dir:
        process_directory(
            args.image_dir, spatial_model, freq_model,
            spatial_transform, freq_transform, device,
            use_ensemble=args.ensemble,
            output_csv=args.output_csv,
        )


if __name__ == '__main__':
    main()