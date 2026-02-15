from __future__ import annotations

import argparse
import pathlib
import subprocess
import sys


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="One-command setup and launch script for facial recognition"
    )
    parser.add_argument("--camera-id", type=int, default=0, help="Webcam ID")
    parser.add_argument(
        "--conf",
        type=float,
        default=0.35,
        help="Base object confidence threshold",
    )
    parser.add_argument(
        "--phone-conf",
        type=float,
        default=0.65,
        help="Cell phone specific confidence threshold",
    )
    parser.add_argument(
        "--unknown-margin",
        type=float,
        default=0.10,
        help="Unknown classification margin for uncertain phone detections",
    )
    parser.add_argument(
        "--hide-unknown",
        action="store_true",
        help="Hide uncertain phone detections instead of labeling them as unknown",
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="yolov8n.pt",
        help="YOLO weights path",
    )
    parser.add_argument(
        "--only-classes",
        type=str,
        default="",
        help="Optional comma-separated classes, e.g. 'person,bus,cell phone'",
    )
    parser.add_argument(
        "--skip-install",
        action="store_true",
        help="Skip installing dependencies from requirements.txt",
    )
    parser.add_argument(
        "--setup-only",
        action="store_true",
        help="Install dependencies and exit without launching webcam",
    )
    return parser.parse_args()


def ensure_dependencies(project_root: pathlib.Path) -> int:
    requirements_path = project_root / "requirements.txt"
    if not requirements_path.exists():
        print(f"requirements.txt not found at: {requirements_path}")
        return 1

    print("Installing project dependencies...")
    install_cmd = [
        sys.executable,
        "-m",
        "pip",
        "install",
        "-r",
        str(requirements_path),
    ]
    completed = subprocess.run(install_cmd, check=False)
    return completed.returncode


def main() -> int:
    args = parse_args()
    project_root = pathlib.Path(__file__).resolve().parent
    app_path = project_root / "app.py"

    if not app_path.exists():
        print(f"app.py not found at: {app_path}")
        return 1

    if not args.skip_install:
        install_code = ensure_dependencies(project_root)
        if install_code != 0:
            print("Dependency installation failed. Please check pip output above.")
            return install_code

    if args.setup_only:
        print("Setup completed. Run this script again without --setup-only to launch the widget.")
        return 0

    command = [
        sys.executable,
        str(app_path),
        "--mode",
        "webcam",
        "--camera-id",
        str(args.camera_id),
        "--conf",
        str(args.conf),
        "--phone-conf",
        str(args.phone_conf),
        "--unknown-margin",
        str(args.unknown_margin),
        "--weights",
        args.weights,
    ]

    if args.only_classes.strip():
        command.extend(["--only-classes", args.only_classes])

    if args.hide_unknown:
        command.append("--hide-unknown")

    print("Launching facial recognition widget...")
    completed = subprocess.run(command, check=False)
    return completed.returncode


if __name__ == "__main__":
    raise SystemExit(main())
