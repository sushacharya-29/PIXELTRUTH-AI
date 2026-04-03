import torch
import torch.nn.functional as F
from PIL import Image
import argparse
import os

from v5_1oldfrequency import get_frequency_model_v5


# -------------------------------
# Image Preprocessing
# -------------------------------
def preprocess_image(image_path, image_size=128):
    img = Image.open(image_path).convert("RGB")

    # Resize
    img = img.resize((image_size, image_size))

    # Convert to tensor [0,1]
    img = torch.tensor(list(img.getdata()), dtype=torch.float32)
    img = img.view(image_size, image_size, 3).permute(2, 0, 1) / 255.0

    # Normalize (important for stability)
    mean = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    std = torch.tensor([0.5, 0.5, 0.5]).view(3, 1, 1)
    img = (img - mean) / std

    return img.unsqueeze(0)  # (1,3,H,W)


# -------------------------------
# Prediction
# -------------------------------
def predict(model, image_tensor, device):
    model.eval()
    image_tensor = image_tensor.to(device)

    with torch.no_grad():
        logits, confidence = model(image_tensor, return_confidence=True)
        probs = F.softmax(logits, dim=1)

    pred_class = torch.argmax(probs, dim=1).item()
    pred_prob = probs[0, pred_class].item()
    confidence = confidence.item()

    return pred_class, pred_prob, confidence


# -------------------------------
# Main
# -------------------------------
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=str, required=True, help="Path to image")
    parser.add_argument("--weights", type=str, default=None, help="Path to weights (.pth)")
    parser.add_argument("--image_size", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"\nUsing device: {device}")

    # Load model
    model = get_frequency_model_v5(
        image_size=args.image_size,
        weights_path=args.weights
    ).to(device)

    # Preprocess
    if not os.path.exists(args.image):
        print("❌ Image not found!")
        return

    img_tensor = preprocess_image(args.image, args.image_size)

    # Predict
    pred_class, prob, conf = predict(model, img_tensor, device)

    # Label mapping (edit if needed)
    class_names = ["REAL", "AI"]

    print("\n===== RESULT =====")
    print(f"Prediction : {class_names[pred_class]}")
    print(f"Probability: {prob:.4f}")
    print(f"Confidence : {conf:.4f}")

    # Extra insight
    if conf < 0.5:
        print("⚠️ Low confidence prediction (model unsure)")
    elif conf > 0.8:
        print("✅ High confidence prediction")


if __name__ == "__main__":
    main()
    