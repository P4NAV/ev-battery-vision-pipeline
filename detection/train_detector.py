"""Train or fine-tune the YOLOv8 detector.

This script reflects the detector training strategy described in the final report:
YOLOv8n, 768 px input resolution, CPU-compatible training/inference, and an optional
checkpoint-based fine-tuning stage for busbar recall improvement.
"""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train YOLOv8 detector for EV battery modules and busbars")
    parser.add_argument("--data", default="detection/configs/dataset.yaml", help="Path to YOLO dataset YAML")
    parser.add_argument("--weights", default="yolov8n.pt", help="Starting weights, e.g. yolov8n.pt or previous best.pt")
    parser.add_argument("--epochs", type=int, default=100, help="Number of training epochs")
    parser.add_argument("--imgsz", type=int, default=768, help="Training image size")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--name", default="modules_busbar_yolov8n", help="Ultralytics run name")
    parser.add_argument("--project", default="runs/detect", help="Output project folder")
    parser.add_argument("--device", default="cpu", help="Device: cpu, 0, mps, etc.")
    parser.add_argument("--workers", type=int, default=2, help="Data-loader workers")
    parser.add_argument("--patience", type=int, default=15, help="Early stopping patience")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    data_path = Path(args.data)
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset config not found: {data_path}")

    model = YOLO(args.weights)

    model.train(
        data=str(data_path),
        epochs=args.epochs,
        imgsz=args.imgsz,
        batch=args.batch,
        patience=args.patience,
        workers=args.workers,
        pretrained=True,
        project=args.project,
        name=args.name,
        plots=True,
        cache=False,
        device=args.device,
        optimizer="AdamW",
        cos_lr=True,
        mosaic=0.5,
    )


if __name__ == "__main__":
    main()
