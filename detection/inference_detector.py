"""Run detector inference on an image, video, webcam, or folder."""

from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="YOLOv8 detector inference")
    parser.add_argument("--weights", default="models/detector/best.pt")
    parser.add_argument("--source", required=True, help="Image, folder, video, or webcam index")
    parser.add_argument("--conf", type=float, default=0.21, help="Detector confidence threshold")
    parser.add_argument("--imgsz", type=int, default=768)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--name", default="detector_predictions")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not Path(args.weights).exists():
        raise FileNotFoundError(f"Weights not found: {args.weights}")

    model = YOLO(args.weights)
    model.predict(
        source=args.source,
        conf=args.conf,
        imgsz=args.imgsz,
        device=args.device,
        save=True,
        project="results",
        name=args.name,
    )


if __name__ == "__main__":
    main()
