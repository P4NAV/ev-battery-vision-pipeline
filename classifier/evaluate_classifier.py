"""Evaluate the ResNet18 binary module condition classifier."""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


def build_resnet18(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_checkpoint(path: Path):
    checkpoint = torch.load(path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        return checkpoint["model_state"], checkpoint.get("classes", ["bad", "good"])
    return checkpoint, ["bad", "good"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate ResNet18 classifier")
    parser.add_argument("--data", default="data/classifier_sample")
    parser.add_argument("--weights", default="models/classifier/best_resnet18.pt")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--batch", type=int, default=16)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    split_dir = Path(args.data) / args.split
    weights_path = Path(args.weights)
    if not split_dir.exists():
        raise FileNotFoundError(f"Split folder not found: {split_dir}")
    if not weights_path.exists():
        raise FileNotFoundError(f"Weights not found: {weights_path}")

    tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    ds = datasets.ImageFolder(split_dir, transform=tfms)
    loader = DataLoader(ds, batch_size=args.batch, shuffle=False)

    state_dict, classes = load_checkpoint(weights_path)
    model = build_resnet18(num_classes=len(classes))
    model.load_state_dict(state_dict)
    model.eval()

    y_true, y_pred = [], []
    with torch.no_grad():
        for images, labels in loader:
            logits = model(images)
            preds = logits.argmax(1)
            y_true.extend(labels.tolist())
            y_pred.extend(preds.tolist())

    print("Classes:", classes)
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print("Weighted F1:", round(f1_score(y_true, y_pred, average="weighted"), 4))
    print("Confusion matrix:\n", confusion_matrix(y_true, y_pred))
    print(classification_report(y_true, y_pred, target_names=classes))


if __name__ == "__main__":
    main()
