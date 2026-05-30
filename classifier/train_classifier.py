"""Train a ResNet18 binary condition classifier for EV battery module crops.

Expected dataset layout follows torchvision.datasets.ImageFolder:

    data/classifier_sample/train/bad
    data/classifier_sample/train/good
    data/classifier_sample/val/bad
    data/classifier_sample/val/good

The final report used a manually refined crop-level dataset with class order
['bad', 'good'] and CPU training/inference.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms
from tqdm import tqdm


def build_resnet18(num_classes: int = 2, pretrained: bool = True) -> nn.Module:
    weights = models.ResNet18_Weights.DEFAULT if pretrained else None
    model = models.resnet18(weights=weights)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train ResNet18 condition classifier")
    parser.add_argument("--data", default="data/classifier_sample")
    parser.add_argument("--epochs", type=int, default=10)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--out", default="models/classifier/best_resnet18.pt")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def make_loaders(data_root: Path, batch_size: int):
    train_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tfms)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    return train_ds, val_ds, train_loader, val_loader


def class_weights_from_dataset(dataset: datasets.ImageFolder) -> torch.Tensor:
    counts = torch.zeros(len(dataset.classes), dtype=torch.float32)
    for _, label in dataset.samples:
        counts[label] += 1
    weights = 1.0 / torch.clamp(counts, min=1.0)
    weights = weights / weights.sum() * len(dataset.classes)
    return weights


def evaluate(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device):
    model.eval()
    total_loss, correct, total = 0.0, 0, 0
    with torch.no_grad():
        for images, labels in loader:
            images, labels = images.to(device), labels.to(device)
            logits = model(images)
            loss = criterion(logits, labels)
            total_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)
    return total_loss / max(total, 1), correct / max(total, 1)


def main() -> None:
    args = parse_args()
    data_root = Path(args.data)
    if not (data_root / "train").exists() or not (data_root / "val").exists():
        raise FileNotFoundError("Expected train/ and val/ folders with bad/ and good/ subfolders.")

    device = torch.device(args.device)
    train_ds, val_ds, train_loader, val_loader = make_loaders(data_root, args.batch)
    print("Classes:", train_ds.classes)

    model = build_resnet18(num_classes=len(train_ds.classes), pretrained=True).to(device)
    weights = class_weights_from_dataset(train_ds).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)

    best_acc = 0.0
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_loss = 0.0
        correct, total = 0, 0

        for images, labels in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            correct += (logits.argmax(1) == labels).sum().item()
            total += labels.size(0)

        train_loss = running_loss / max(total, 1)
        train_acc = correct / max(total, 1)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)

        print(
            f"epoch={epoch:02d} train_loss={train_loss:.4f} train_acc={train_acc:.3f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
        )

        if val_acc >= best_acc:
            best_acc = val_acc
            torch.save({
                "model_state": model.state_dict(),
                "classes": train_ds.classes,
                "img_size": 224,
                "best_val_acc": best_acc,
            }, out_path)
            print(f"Saved best model to {out_path}")


if __name__ == "__main__":
    main()
