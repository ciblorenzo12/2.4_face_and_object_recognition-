from __future__ import annotations

import argparse
import os
import pathlib
import time
from collections import Counter
from typing import List, Optional, Set, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


FaceBox = Tuple[int, int, int, int]
ObjectBox = Tuple[int, int, int, int, str, float]


def detect_faces(frame, face_cascade: cv2.CascadeClassifier) -> List[FaceBox]:
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(
        gray,
        scaleFactor=1.1,
        minNeighbors=5,
        minSize=(30, 30),
        flags=cv2.CASCADE_SCALE_IMAGE,
    )
    return [(int(x), int(y), int(w), int(h)) for (x, y, w, h) in faces]


def detect_objects(
    frame,
    model: YOLO,
    conf: float,
    allowed_labels: Optional[Set[str]] = None,
    class_conf_overrides: Optional[dict[str, float]] = None,
    unknown_guard_labels: Optional[Set[str]] = None,
    unknown_margin: float = 0.10,
    show_unknown: bool = True,
) -> List[ObjectBox]:
    results = model(frame, verbose=False, conf=conf)
    output: List[ObjectBox] = []
    class_conf_overrides = class_conf_overrides or {}
    unknown_guard_labels = unknown_guard_labels if unknown_guard_labels is not None else {"cell phone"}

    for result in results:
        if result.boxes is None:
            continue
        for box in result.boxes:
            cls_id = int(box.cls[0])
            label = model.names[cls_id]
            score = float(box.conf[0])
            if allowed_labels is not None and label not in allowed_labels:
                continue
            required_conf = class_conf_overrides.get(label, conf)
            if score < required_conf:
                continue
            if label in unknown_guard_labels and score < (required_conf + unknown_margin):
                if show_unknown:
                    label = "unknown"
                else:
                    continue
            x1, y1, x2, y2 = box.xyxy[0].tolist()
            output.append((int(x1), int(y1), int(x2), int(y2), label, score))

    return output


def draw_detections(frame, faces: List[FaceBox], objects: List[ObjectBox]):
    for (x, y, w, h) in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (255, 80, 80), 2)
        cv2.putText(
            frame,
            "Face",
            (x, max(20, y - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 80, 80),
            2,
        )

    for (x1, y1, x2, y2, label, score) in objects:
        color = (80, 255, 80) if label != "unknown" else (0, 210, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        caption = f"{label} {score:.2f}"
        cv2.putText(
            frame,
            caption,
            (x1, max(20, y1 - 8)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            color,
            2,
        )

    return frame


def load_face_cascade() -> cv2.CascadeClassifier:
    cascade_path = os.path.dirname(cv2.__file__) + "/data/haarcascade_frontalface_default.xml"
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise RuntimeError(f"Could not load face cascade at: {cascade_path}")
    return cascade


def process_frame(
    frame,
    model: YOLO,
    face_cascade: cv2.CascadeClassifier,
    conf: float,
    allowed_labels: Optional[Set[str]],
    class_conf_overrides: Optional[dict[str, float]],
    unknown_guard_labels: Optional[Set[str]],
    unknown_margin: float,
    show_unknown: bool,
):
    faces = detect_faces(frame, face_cascade)
    objects = detect_objects(
        frame,
        model,
        conf,
        allowed_labels,
        class_conf_overrides,
        unknown_guard_labels,
        unknown_margin,
        show_unknown,
    )
    return draw_detections(frame, faces, objects), faces, objects


def run_on_image(
    image_path: str,
    output_path: str,
    model: YOLO,
    face_cascade: cv2.CascadeClassifier,
    conf: float,
    allowed_labels: Optional[Set[str]] = None,
    class_conf_overrides: Optional[dict[str, float]] = None,
    unknown_guard_labels: Optional[Set[str]] = None,
    unknown_margin: float = 0.10,
    show_unknown: bool = True,
):
    frame = cv2.imread(image_path)
    if frame is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")

    annotated, faces, objects = process_frame(
        frame,
        model,
        face_cascade,
        conf,
        allowed_labels,
        class_conf_overrides,
        unknown_guard_labels,
        unknown_margin,
        show_unknown,
    )
    cv2.imwrite(output_path, annotated)
    print(f"Saved annotated image to: {output_path}")
    print(f"Detected {len(faces)} face(s) and {len(objects)} object(s)")


def run_on_webcam(
    camera_id: int,
    model: YOLO,
    face_cascade: cv2.CascadeClassifier,
    conf: float,
    allowed_labels: Optional[Set[str]] = None,
    class_conf_overrides: Optional[dict[str, float]] = None,
    unknown_guard_labels: Optional[Set[str]] = None,
    unknown_margin: float = 0.10,
    show_unknown: bool = True,
):
    window_name = "Face and Object Recognition 2.2 Assignment"
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera with id: {camera_id}")

    print("Running real-time recognition. Press 'q' or 'Esc' to quit.")
    while True:
        success, frame = cap.read()
        if not success:
            break

        annotated, _, _ = process_frame(
            frame,
            model,
            face_cascade,
            conf,
            allowed_labels,
            class_conf_overrides,
            unknown_guard_labels,
            unknown_margin,
            show_unknown,
        )
        cv2.putText(
            annotated,
            "Press Q or Esc to quit",
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break

        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break

    cap.release()
    cv2.destroyAllWindows()


def run_calibration(
    camera_id: int,
    model: YOLO,
    face_cascade: cv2.CascadeClassifier,
    conf: float,
    phone_conf: float,
    allowed_labels: Optional[Set[str]] = None,
    seconds: int = 120,
):
    window_name = "Face and Object Recognition - Calibration"
    cap = cv2.VideoCapture(camera_id)
    if not cap.isOpened():
        raise RuntimeError(f"Unable to open camera with id: {camera_id}")

    calibration_low_conf = 0.15
    all_scores: List[float] = []
    phone_scores: List[float] = []
    class_counts: Counter[str] = Counter()
    start = time.time()

    print("Calibration mode running. Show typical objects to your camera. Press 'q' or 'Esc' to finish early.")

    while True:
        success, frame = cap.read()
        if not success:
            break

        probe_objects = detect_objects(
            frame,
            model,
            calibration_low_conf,
            allowed_labels,
            class_conf_overrides=None,
            unknown_guard_labels=set(),
            unknown_margin=0.0,
            show_unknown=False,
        )

        for _, _, _, _, label, score in probe_objects:
            class_counts[label] += 1
            all_scores.append(score)
            if label == "cell phone":
                phone_scores.append(score)

        faces = detect_faces(frame, face_cascade)
        display_objects = detect_objects(
            frame,
            model,
            conf,
            allowed_labels,
            class_conf_overrides={"cell phone": phone_conf},
            unknown_guard_labels={"cell phone"},
            unknown_margin=0.10,
            show_unknown=True,
        )
        annotated = draw_detections(frame, faces, display_objects)

        elapsed = int(time.time() - start)
        remaining = max(0, seconds - elapsed)
        overlay = f"Calibrating: {remaining}s | conf={conf:.2f}, phone_conf={phone_conf:.2f}"
        cv2.putText(
            annotated,
            overlay,
            (10, 28),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
        )
        cv2.putText(
            annotated,
            "Press Q or Esc to quit",
            (10, 56),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.60,
            (255, 255, 255),
            2,
        )

        cv2.imshow(window_name, annotated)

        key = cv2.waitKey(1) & 0xFF
        if key in (ord("q"), ord("Q"), 27):
            break
        if cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1:
            break
        if elapsed >= seconds:
            break

    cap.release()
    cv2.destroyAllWindows()

    if all_scores:
        recommended_conf = float(np.clip(np.quantile(np.array(all_scores), 0.35), 0.25, 0.70))
    else:
        recommended_conf = conf

    if phone_scores:
        recommended_phone_conf = float(np.clip(np.quantile(np.array(phone_scores), 0.80), 0.55, 0.90))
    else:
        recommended_phone_conf = max(phone_conf, 0.70)

    report_path = pathlib.Path("outputs") / "calibration_report.txt"
    top_classes = class_counts.most_common(10)
    with report_path.open("w", encoding="utf-8") as report:
        report.write("Face and Object Recognition Calibration Report\n")
        report.write("=" * 48 + "\n")
        report.write(f"Frames/Object samples analyzed: {len(all_scores)}\n")
        report.write(f"Cell-phone samples analyzed: {len(phone_scores)}\n")
        report.write(f"Recommended --conf: {recommended_conf:.2f}\n")
        report.write(f"Recommended --phone-conf: {recommended_phone_conf:.2f}\n\n")
        report.write("Top observed classes:\n")
        for label, count in top_classes:
            report.write(f"- {label}: {count}\n")

    print(f"Saved calibration report to: {report_path}")
    print(
        "Suggested run command: "
        f"python app.py --mode webcam --camera-id {camera_id} --conf {recommended_conf:.2f} --phone-conf {recommended_phone_conf:.2f}"
    )


def parse_args():
    parser = argparse.ArgumentParser(description="Face and object recognition app")
    parser.add_argument(
        "--mode",
        choices=["webcam", "image", "calibrate"],
        default="webcam",
        help="Run real-time webcam mode, single image mode, or calibration mode",
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam ID")
    parser.add_argument("--image-path", type=str, help="Path to input image for image mode")
    parser.add_argument(
        "--output-path",
        type=str,
        default="outputs/recognized_output.jpg",
        help="Path to save annotated image in image mode",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="YOLO weights file (default downloads yolov8n.pt automatically)",
    )
    parser.add_argument("--conf", type=float, default=0.35, help="Object confidence threshold")
    parser.add_argument(
        "--phone-conf",
        type=float,
        default=0.65,
        help="Higher confidence threshold only for 'cell phone' class to reduce false positives",
    )
    parser.add_argument(
        "--only-classes",
        type=str,
        default="",
        help="Optional comma-separated class filter, e.g. 'bus,person,cell phone'",
    )
    parser.add_argument(
        "--unknown-margin",
        type=float,
        default=0.10,
        help="If phone confidence is within this margin above threshold, label as unknown",
    )
    parser.add_argument(
        "--hide-unknown",
        action="store_true",
        help="Hide uncertain phone detections instead of labeling them as unknown",
    )
    parser.add_argument(
        "--calibration-seconds",
        type=int,
        default=120,
        help="Duration for calibration mode",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    pathlib.Path("outputs").mkdir(parents=True, exist_ok=True)

    model = YOLO(args.weights)
    face_cascade = load_face_cascade()
    allowed_labels = None
    if args.only_classes.strip():
        allowed_labels = {x.strip() for x in args.only_classes.split(",") if x.strip()}

    class_conf_overrides = {"cell phone": args.phone_conf}
    unknown_guard_labels = {"cell phone"}
    show_unknown = not args.hide_unknown

    if args.mode == "image":
        if not args.image_path:
            raise ValueError("--image-path is required when --mode image")
        run_on_image(
            args.image_path,
            args.output_path,
            model,
            face_cascade,
            args.conf,
            allowed_labels,
            class_conf_overrides,
            unknown_guard_labels,
            args.unknown_margin,
            show_unknown,
        )
    elif args.mode == "webcam":
        run_on_webcam(
            args.camera_id,
            model,
            face_cascade,
            args.conf,
            allowed_labels,
            class_conf_overrides,
            unknown_guard_labels,
            args.unknown_margin,
            show_unknown,
        )
    else:
        run_calibration(
            args.camera_id,
            model,
            face_cascade,
            args.conf,
            args.phone_conf,
            allowed_labels,
            args.calibration_seconds,
        )


if __name__ == "__main__":
    main()
