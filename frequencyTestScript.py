"""
Frequency Model - Folder Test Script
======================================
Tests ONLY the frequency model on a folder of images.

Usage:
    # Test on real images:
    python test_frequency.py --folder CIFAKE/test/REAL --true_label REAL

    # Test on AI generated images:
    python test_frequency.py --folder CIFAKE/test/FAKE --true_label FAKE

    # Quick test on first 100 images only:
    python test_frequency.py --folder CIFAKE/test/REAL --true_label REAL --limit 100

    # Just get verdicts, no accuracy:
    python test_frequency.py --folder my_images
"""

import os
import argparse
from inference import AIImageDetectorNN

# ─────────────────────────────────────────
# Arguments
# ─────────────────────────────────────────

parser = argparse.ArgumentParser(description='Test Frequency Model on a folder of images')

parser.add_argument(
    '--folder',
    type=str,
    required=True,
    help='Path to folder containing images'
)

parser.add_argument(
    '--true_label',
    choices=['REAL', 'FAKE'],
    default=None,
    help='True label for all images in the folder (enables accuracy calculation)'
)

parser.add_argument(
    '--limit',
    type=int,
    default=None,
    help='Only test this many images (e.g. --limit 100 for a quick test)'
)

args = parser.parse_args()

# ─────────────────────────────────────────
# Validate folder
# ─────────────────────────────────────────

if not os.path.exists(args.folder):
    print(f"ERROR: Folder not found: {args.folder}")
    exit(1)

ALLOWED = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tiff'}
images  = sorted([
    f for f in os.listdir(args.folder)
    if os.path.splitext(f)[1].lower() in ALLOWED
])

if not images:
    print(f"ERROR: No images found in {args.folder}")
    print(f"Supported formats: {', '.join(ALLOWED)}")
    exit(1)

if args.limit:
    images = images[:args.limit]

# ─────────────────────────────────────────
# Check checkpoint
# ─────────────────────────────────────────

FREQUENCY_CKPT = 'checkpoints/frequency_model_best.pth'

if not os.path.exists(FREQUENCY_CKPT):
    print(f"ERROR: Frequency checkpoint not found: {FREQUENCY_CKPT}")
    print("       Train first: python trainingScript.py --model frequency")
    exit(1)

# ─────────────────────────────────────────
# Header
# ─────────────────────────────────────────

print("=" * 65)
print("  FREQUENCY MODEL — FOLDER TEST")
print("=" * 65)
print(f"  Folder      : {args.folder}")
print(f"  Images      : {len(images)}")
print(f"  Checkpoint  : {FREQUENCY_CKPT}")
if args.true_label:
    print(f"  True label  : {args.true_label}  (accuracy will be calculated)")
else:
    print(f"  True label  : not provided  (verdicts only)")
print("=" * 65)

# ─────────────────────────────────────────
# Load frequency model ONLY
# ─────────────────────────────────────────

print("\nLoading frequency model...")
detector = AIImageDetectorNN(
    spatial_weights=None,              # spatial model NOT loaded
    frequency_weights=FREQUENCY_CKPT,
)
print("Frequency model loaded.\n")

# ─────────────────────────────────────────
# Run detection
# ─────────────────────────────────────────

expected_verdict = None
if args.true_label:
    expected_verdict = 'REAL' if args.true_label == 'REAL' else 'AI-GENERATED'

correct  = 0
wrong    = 0
errors   = 0

# Confidence buckets
very_high_correct = 0
high_correct      = 0
medium_correct    = 0
low_correct       = 0

print(f"{'#':<5} {'Image':<40} {'Verdict':<16} {'Real %':<10} {'AI %':<10} {'Result'}")
print("-" * 90)

for idx, img_name in enumerate(images, start=1):
    img_path = os.path.join(args.folder, img_name)

    try:
        result = detector.detect(img_path)
        f      = result.get('frequency')

        if not f or 'error' in f:
            print(f"{idx:<5} {img_name:<40} ERROR: {f.get('error', 'unknown') if f else 'no result'}")
            errors += 1
            continue

        verdict   = f['verdict']
        real_prob = f['real_probability']
        ai_prob   = f['ai_probability']

        # Correctness
        is_correct = None
        tick       = ""
        if expected_verdict:
            is_correct = (verdict == expected_verdict)
            tick       = "✓ CORRECT" if is_correct else "✗ WRONG"
            if is_correct:
                correct += 1
                conf = max(real_prob, ai_prob)
                if conf >= 90:   very_high_correct += 1
                elif conf >= 75: high_correct      += 1
                elif conf >= 60: medium_correct    += 1
                else:            low_correct       += 1
            else:
                wrong += 1

        print(f"{idx:<5} {img_name:<40} {verdict:<16} {real_prob:<10.1f} {ai_prob:<10.1f} {tick}")

    except Exception as ex:
        print(f"{idx:<5} {img_name:<40} ERROR: {ex}")
        errors += 1

# ─────────────────────────────────────────
# Summary
# ─────────────────────────────────────────

processed = correct + wrong

print("\n" + "=" * 65)
print("  FREQUENCY MODEL — SUMMARY")
print("=" * 65)
print(f"  Total images : {len(images)}")
print(f"  Processed    : {processed}")
print(f"  Errors       : {errors}")

if expected_verdict and processed > 0:
    accuracy = correct / processed * 100

    print(f"\n  True label   : {args.true_label}")
    print(f"  Correct      : {correct}")
    print(f"  Wrong        : {wrong}")
    print(f"  Accuracy     : {accuracy:.1f}%")

    print(f"\n  --- Confidence breakdown (correct predictions) ---")
    print(f"  Very High confidence (>=90%) : {very_high_correct}")
    print(f"  High confidence (>=75%)      : {high_correct}")
    print(f"  Medium confidence (>=60%)    : {medium_correct}")
    print(f"  Low confidence (<60%)        : {low_correct}")

    print(f"\n  --- What frequency model looks for ---")
    print(f"  The frequency model detects AI images by analysing")
    print(f"  patterns in the FFT spectrum of the image.")
    print(f"  AI generators leave unnatural frequency fingerprints")
    print(f"  that real cameras do not produce.")

    print(f"\n  --- Verdict ---")
    if accuracy >= 90:
        print(f"  Frequency model is performing WELL ({accuracy:.1f}%)")
    elif accuracy >= 75:
        print(f"  Frequency model is performing DECENTLY ({accuracy:.1f}%) — more training may help.")
    elif accuracy >= 60:
        print(f"  Frequency model is STRUGGLING ({accuracy:.1f}%) — needs more training.")
    else:
        print(f"  Frequency model has LOW accuracy ({accuracy:.1f}%) — check training setup.")

print("=" * 65)