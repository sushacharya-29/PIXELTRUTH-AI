import os
import argparse
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms

from v64spatial import get_spatial_model_v64, CLIP_MEAN, CLIP_STD


# ── Load model ─────────────────────────────────────────────

def load_model(weights_path, device):
    model = get_spatial_model_v64(input_size=224)
    ckpt = torch.load(weights_path, map_location=device, weights_only=False)

    state = ckpt.get('model_state_dict', ckpt)
    model.load_state_dict(state, strict=False)

    model.to(device)
    model.eval()
    return model


# ── Transform ─────────────────────────────────────────────

def get_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])


# ── Predict single image ───────────────────────────────────

def predict_image(model, image_path, device, transform):
    try:
        img = Image.open(image_path).convert('RGB')
    except Exception as e:
        print(f"❌ Error loading {image_path}: {e}")
        return

    img_tensor = transform(img).unsqueeze(0).to(device)

    with torch.no_grad():
        logits = model(img_tensor)
        probs = F.softmax(logits, dim=1)

    pred = probs.argmax(1).item()
    conf = probs.max().item()

    label_map = {0: "FAKE", 1: "REAL"}

    print(f"{image_path} → {label_map[pred]} ({conf*100:.2f}%)")


# ── Predict folder ─────────────────────────────────────────

def predict_folder(model, folder_path, device, transform):
    valid_exts = ('.jpg', '.jpeg', '.png', '.bmp', '.webp')

    files = [f for f in os.listdir(folder_path)
             if f.lower().endswith(valid_exts)]

    print(f"\n📂 Found {len(files)} images\n")

    for file in files:
        path = os.path.join(folder_path, file)
        predict_image(model, path, device, transform)


# ── Main ──────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', required=True, help='Path to checkpoint_best.pt')
    parser.add_argument('--image', help='Single image path')
    parser.add_argument('--folder', help='Folder path')

    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    model = load_model(args.weights, device)
    transform = get_transform()

    if args.image:
        predict_image(model, args.image, device, transform)

    elif args.folder:
        predict_folder(model, args.folder, device, transform)

    else:
        print("❌ Provide either --image or --folder")


if __name__ == "__main__":
    main()