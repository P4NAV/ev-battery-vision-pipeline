"""Evaluate a trained YOLOv8 detector on train/val/test data."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate YOLOv8 detector")
    parser.add_argument("--weights", default="models/detector/best.pt")
    parser.add_argument("--data", default="detection/configs/dataset.yaml")
    parser.add_argument("--split", default="test", choices=["train", "val", "test"])
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--conf", type=float, default=0.001, help="Low threshold for metric calculation")
    parser.add_argument("--device", default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)
    metrics = model.val(
        data=args.data,
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        plots=True,
    )
    print(metrics)


if __name__ == "__main__":
    main()
