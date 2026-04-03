import torch
import torch.nn.functional as F
from PIL import Image
import torchvision.transforms as T
import argparse
import os

# Import your model
from v5spatial import get_spatial_model_v5, CLIP_MEAN, CLIP_STD


# ─────────────────────────────────────────────
# Image preprocessing (CLIP style)
# ─────────────────────────────────────────────
def get_transform(input_size=224):
    return T.Compose([
        T.Resize((input_size, input_size)),
        T.ToTensor(),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])


# ─────────────────────────────────────────────
# Load image safely
# ─────────────────────────────────────────────
def load_image(path, transform):
    img = Image.open(path).convert("RGB")
    return transform(img).unsqueeze(0)  # (1, 3, H, W)


# ─────────────────────────────────────────────
# Prediction function
# ─────────────────────────────────────────────
def predict(model, img_tensor, device):
    img_tensor = img_tensor.to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)

    fake_prob = probs[0, 0].item()
    real_prob = probs[0, 1].item()

    label = "REAL" if real_prob > fake_prob else "FAKE"

    return label, fake_prob, real_prob


# ─────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────
def main(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"\nUsing device: {device}")

    # Load model
    model = get_spatial_model_v5(
        weights_path=args.weights,
        freeze_backbone=True,
        unfreeze_last_n=2
    ).to(device)

    model.eval()

    transform = get_transform()

    if os.path.isfile(args.image):
        # Single image
        img_tensor = load_image(args.image, transform)
        label, fake_prob, real_prob = predict(model, img_tensor, device)

        print(f"\nImage: {args.image}")
        print(f"Prediction: {label}")
        print(f"FAKE: {fake_prob:.4f} | REAL: {real_prob:.4f}")

    elif os.path.isdir(args.image):
        # Folder test
        print(f"\nTesting folder: {args.image}\n")

        for file in os.listdir(args.image):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                path = os.path.join(args.image, file)
                img_tensor = load_image(path, transform)

                label, fake_prob, real_prob = predict(model, img_tensor, device)

                print(f"{file} -> {label} "
                      f"(F: {fake_prob:.3f}, R: {real_prob:.3f})")

    else:
        print("Invalid path!")


# ─────────────────────────────────────────────
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True,
                        help="Path to image or folder")
    parser.add_argument("--weights", type=str, default=None,
                        help="Path to model weights (.pth)")

    args = parser.parse_args()
    main(args)