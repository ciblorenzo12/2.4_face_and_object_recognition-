from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Set

from ultralytics import YOLO

from app import load_face_cascade, run_on_image


def parse_only_classes(value: str) -> Optional[Set[str]]:
    if not value.strip():
        return None
    labels = {item.strip() for item in value.split(",") if item.strip()}
    return labels or None


def build_output_path(input_path: Path, output_dir: Path, suffix: str) -> Path:
    extension = input_path.suffix if input_path.suffix else ".jpg"
    return output_dir / f"{input_path.stem}{suffix}{extension}"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate annotated screenshots for one or more images"
    )
    parser.add_argument(
        "images",
        nargs="+",
        help="One or more input image paths",
    )
    parser.add_argument(
        "--output-dir",
        default="outputs",
        help="Folder where annotated screenshots are saved",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Optional exact output path (only valid when one input image is provided)",
    )
    parser.add_argument(
        "--suffix",
        default="_recognized",
        help="Suffix added to each output filename when --output is not used",
    )
    parser.add_argument(
        "--weights",
        default="yolov8n.pt",
        help="YOLO weights path",
    )
    parser.add_argument(
        "--conf",
        type=float,
        default=0.45,
        help="Base object confidence threshold",
    )
    parser.add_argument(
        "--phone-conf",
        type=float,
        default=0.75,
        help="Confidence threshold for cell phone class",
    )
    parser.add_argument(
        "--unknown-margin",
        type=float,
        default=0.10,
        help="Relabel uncertain phone detections to unknown within this margin",
    )
    parser.add_argument(
        "--hide-unknown",
        action="store_true",
        help="Hide uncertain phone detections instead of labeling as unknown",
    )
    parser.add_argument(
        "--only-classes",
        default="",
        help="Optional comma-separated labels, e.g. 'person,bus,cell phone'",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    input_paths = [Path(path_str) for path_str in args.images]
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.output and len(input_paths) != 1:
        raise ValueError("--output can only be used when exactly one image path is provided")

    model = YOLO(args.weights)
    face_cascade = load_face_cascade()

    allowed_labels = parse_only_classes(args.only_classes)
    class_conf_overrides = {"cell phone": args.phone_conf}
    unknown_guard_labels = {"cell phone"}
    show_unknown = not args.hide_unknown

    success_count = 0
    failed_count = 0

    for input_path in input_paths:
        if not input_path.exists():
            print(f"[SKIP] Missing input file: {input_path}")
            failed_count += 1
            continue

        output_path = Path(args.output) if args.output else build_output_path(input_path, output_dir, args.suffix)

        try:
            run_on_image(
                str(input_path),
                str(output_path),
                model,
                face_cascade,
                args.conf,
                allowed_labels,
                class_conf_overrides,
                unknown_guard_labels,
                args.unknown_margin,
                show_unknown,
            )
            success_count += 1
        except Exception as error:
            print(f"[FAIL] {input_path}: {error}")
            failed_count += 1

    print(f"Done. Success: {success_count}, Failed: {failed_count}")


if __name__ == "__main__":
    main()
