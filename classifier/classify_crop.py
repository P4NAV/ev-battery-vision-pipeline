"""Classify a single cropped battery module image as bad/good and Grade A/B/C triage."""

from __future__ import annotations

import argparse
from pathlib import Path

from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms


def build_resnet18(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def triage_from_p_bad(p_bad: float):
    if p_bad < 0.30:
        return "Grade A", "likely reusable"
    if p_bad < 0.70:
        return "Grade B", "manual review"
    return "Grade C", "likely damaged"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--weights", default="models/classifier/best_resnet18.pt")
    args = parser.parse_args()

    if not Path(args.weights).exists():
        raise FileNotFoundError(args.weights)

    checkpoint = torch.load(args.weights, map_location="cpu")
    state = checkpoint.get("model_state", checkpoint)
    classes = checkpoint.get("classes", ["bad", "good"])

    model = build_resnet18(len(classes))
    model.load_state_dict(state)
    model.eval()

    tfm = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    image = Image.open(args.image).convert("RGB")
    x = tfm(image).unsqueeze(0)

    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1).squeeze()

    p_bad = float(probs[classes.index("bad")]) if "bad" in classes else float(probs[0])
    pred_idx = int(torch.argmax(probs))
    grade, interpretation = triage_from_p_bad(p_bad)

    print(f"Prediction: {classes[pred_idx]} | conf={float(probs[pred_idx]):.3f}")
    print(f"p_bad={p_bad:.3f} | {grade} | {interpretation}")


if __name__ == "__main__":
    main()
