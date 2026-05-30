"""Integrated YOLOv8 + ResNet18 EV battery inspection pipeline.

The detector localises EV battery modules and busbars. Module detections are cropped
and passed to a ResNet18 binary condition classifier. The bad-class probability is mapped
onto Grade A/B/C triage thresholds, matching the interpretation used in the final report.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import cv2
from PIL import Image
import torch
import torch.nn as nn
from torchvision import models, transforms
from ultralytics import YOLO


DEFAULT_CLASSES = ["bad", "good"]


def build_classifier(num_classes: int = 2) -> nn.Module:
    model = models.resnet18(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    return model


def load_classifier(weights_path: str | Path):
    checkpoint = torch.load(weights_path, map_location="cpu")
    if isinstance(checkpoint, dict) and "model_state" in checkpoint:
        state_dict = checkpoint["model_state"]
        classes = checkpoint.get("classes", DEFAULT_CLASSES)
    else:
        state_dict = checkpoint
        classes = DEFAULT_CLASSES

    model = build_classifier(len(classes))
    model.load_state_dict(state_dict)
    model.eval()
    return model, classes


def classifier_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


def triage_from_p_bad(p_bad: float):
    if p_bad < 0.30:
        return "Grade A", "likely reusable", (0, 180, 0)
    if p_bad < 0.70:
        return "Grade B", "manual review", (0, 180, 180)
    return "Grade C", "likely damaged", (0, 0, 220)


def is_module_detection(class_id: int, detector: YOLO) -> bool:
    """Return True for module detections; fallback to True for single-class detectors."""
    try:
        names = detector.names
        class_name = str(names[class_id]).lower()
    except Exception:
        return True

    if len(names) == 1:
        return True
    return "module" in class_name


def classify_crop(crop_bgr, classifier, classes, tfm):
    crop_rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(crop_rgb)
    x = tfm(image).unsqueeze(0)
    with torch.no_grad():
        probs = torch.softmax(classifier(x), dim=1).squeeze()

    pred_idx = int(torch.argmax(probs))
    p_bad = float(probs[classes.index("bad")]) if "bad" in classes else float(probs[0])
    grade, interpretation, colour = triage_from_p_bad(p_bad)
    return {
        "pred_class": classes[pred_idx],
        "pred_conf": float(probs[pred_idx]),
        "p_bad": p_bad,
        "grade": grade,
        "interpretation": interpretation,
        "colour": colour,
    }


def run_pipeline(image_path: str | Path, detector_path: str | Path, classifier_path: str | Path,
                 output_path: str | Path, conf: float = 0.21, imgsz: int = 768):
    image_path = Path(image_path)
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    image = cv2.imread(str(image_path))
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    detector = YOLO(str(detector_path))
    classifier, classes = load_classifier(classifier_path)
    tfm = classifier_transform()

    results = detector.predict(source=image, conf=conf, imgsz=imgsz, device="cpu", verbose=False)
    annotated = image.copy()
    rows = []

    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
        det_conf = float(box.conf[0].item())
        class_id = int(box.cls[0].item())
        det_name = detector.names.get(class_id, str(class_id)) if isinstance(detector.names, dict) else str(class_id)

        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, image.shape[1] - 1), min(y2, image.shape[0] - 1)
        crop = image[y1:y2, x1:x2]
        if crop.size == 0:
            continue

        if is_module_detection(class_id, detector):
            cls_result = classify_crop(crop, classifier, classes, tfm)
            label = (
                f"module | {cls_result['grade']} | d={det_conf:.2f} | "
                f"p_bad={cls_result['p_bad']:.2f}"
            )
            colour = cls_result["colour"]
            rows.append({
                "det_class": det_name,
                "det_conf": det_conf,
                "bbox": [x1, y1, x2, y2],
                **{k: v for k, v in cls_result.items() if k != "colour"},
            })
        else:
            label = f"{det_name} | d={det_conf:.2f}"
            colour = (255, 120, 0)
            rows.append({"det_class": det_name, "det_conf": det_conf, "bbox": [x1, y1, x2, y2]})

        cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
        cv2.putText(annotated, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

    cv2.imwrite(str(output_path), annotated)
    return rows, output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run integrated detector + classifier pipeline")
    parser.add_argument("--image", required=True)
    parser.add_argument("--detector", default="models/detector/best.pt")
    parser.add_argument("--classifier", default="models/classifier/best_resnet18.pt")
    parser.add_argument("--out", default="results/sample_outputs/integrated_output.jpg")
    parser.add_argument("--conf", type=float, default=0.21)
    parser.add_argument("--imgsz", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows, out = run_pipeline(args.image, args.detector, args.classifier, args.out, args.conf, args.imgsz)
    print(f"Saved annotated output to: {out}")
    for i, row in enumerate(rows, 1):
        print(i, row)


if __name__ == "__main__":
    main()
