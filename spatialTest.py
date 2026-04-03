import os
import argparse
from pathlib import Path

import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T

# Import your model
from v5spatial import get_spatial_model_v5, CLIP_MEAN, CLIP_STD


# ── Image Transform ─────────────────────────────────────────────
def get_transform():
    return T.Compose([
        T.Resize((224, 224)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ── Load Model ──────────────────────────────────────────────────
def load_model(weights_path, device):
    model = get_spatial_model_v5(
        freeze_backbone=True,
        unfreeze_last_n=2
    )

    if weights_path and os.path.exists(weights_path):
        ckpt = torch.load(weights_path, map_location=device)
        state = ckpt.get('model_state_dict', ckpt)
        model.load_state_dict(state, strict=False)
        print(f"Loaded weights: {weights_path}")

    model.to(device)
    model.eval()
    return model


# ── Predict Single Image ────────────────────────────────────────
def predict_image(model, image_path, device, transform):
    img = Image.open(image_path).convert("RGB")
    x = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(x)
        probs = F.softmax(logits, dim=1)[0]

    fake_prob = probs[0].item()
    real_prob = probs[1].item()

    label = "FAKE" if fake_prob > real_prob else "REAL"
    confidence = max(fake_prob, real_prob)

    return label, confidence, fake_prob, real_prob


# ── Folder Prediction ───────────────────────────────────────────
def predict_folder(model, folder_path, device, transform):
    image_exts = [".jpg", ".jpeg", ".png", ".webp", ".bmp"]

    files = [f for f in Path(folder_path).iterdir()
             if f.suffix.lower() in image_exts]

    if not files:
        print("No images found in folder.")
        return

    print(f"\nProcessing {len(files)} images...\n")

    for img_path in files:
        try:
            label, conf, fake_p, real_p = predict_image(
                model, str(img_path), device, transform
            )

            print(f"{img_path.name:30s} | {label:5s} | "
                  f"Conf: {conf:.3f} | F:{fake_p:.3f} R:{real_p:.3f}")

        except Exception as e:
            print(f"{img_path.name} -> ERROR: {e}")


# ── Main ───────────────────────────────────────────────────────
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--folder", type=str, help="Path to folder")
    parser.add_argument("--weights", type=str, required=True, help="Model weights path")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    transform = get_transform()
    model = load_model(args.weights, device)

    if args.image:
        label, conf, fake_p, real_p = predict_image(
            model, args.image, device, transform
        )

        print("\n===== RESULT =====")
        print(f"Image: {args.image}")
        print(f"Prediction: {label}")
        print(f"Confidence: {conf:.4f}")
        print(f"FAKE: {fake_p:.4f} | REAL: {real_p:.4f}")

    elif args.folder:
        predict_folder(model, args.folder, device, transform)

    else:
        print("Provide --image or --folder")


if __name__ == "__main__":
    main()