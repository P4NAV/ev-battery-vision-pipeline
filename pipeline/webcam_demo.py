"""Real-time webcam demo for EV module detection and condition triage."""

from __future__ import annotations

import argparse

import cv2
from ultralytics import YOLO

from integrated_pipeline import classifier_transform, classify_crop, is_module_detection, load_classifier


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Webcam demo: YOLOv8 + ResNet18")
    parser.add_argument("--detector", default="models/detector/best.pt")
    parser.add_argument("--classifier", default="models/classifier/best_resnet18.pt")
    parser.add_argument("--camera", type=int, default=0)
    parser.add_argument("--conf", type=float, default=0.21)
    parser.add_argument("--imgsz", type=int, default=768)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    detector = YOLO(args.detector)
    classifier, classes = load_classifier(args.classifier)
    tfm = classifier_transform()

    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. On macOS, check Terminal/IDE camera permissions.")

    while True:
        ok, frame = cap.read()
        if not ok:
            break

        results = detector.predict(source=frame, conf=args.conf, imgsz=args.imgsz, device="cpu", verbose=False)
        annotated = frame.copy()

        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
            class_id = int(box.cls[0].item())
            det_conf = float(box.conf[0].item())
            x1, y1 = max(x1, 0), max(y1, 0)
            x2, y2 = min(x2, frame.shape[1] - 1), min(y2, frame.shape[0] - 1)
            crop = frame[y1:y2, x1:x2]

            if crop.size > 0 and is_module_detection(class_id, detector):
                cls_result = classify_crop(crop, classifier, classes, tfm)
                label = f"module | {cls_result['grade']} | d={det_conf:.2f} | p_bad={cls_result['p_bad']:.2f}"
                colour = cls_result["colour"]
            else:
                label = f"busbar | d={det_conf:.2f}"
                colour = (255, 120, 0)

            cv2.rectangle(annotated, (x1, y1), (x2, y2), colour, 2)
            cv2.putText(annotated, label, (x1, max(y1 - 8, 20)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, colour, 2)

        cv2.imshow("EV Module Detection + Condition Assessment", annotated)
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
