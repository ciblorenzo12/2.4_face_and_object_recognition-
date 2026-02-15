from __future__ import annotations

import pathlib
import urllib.request

from app import load_face_cascade, run_on_image
from ultralytics import YOLO


def main():
    data_dir = pathlib.Path("sample_data")
    out_dir = pathlib.Path("outputs")
    data_dir.mkdir(parents=True, exist_ok=True)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample_url = "https://ultralytics.com/images/bus.jpg"
    input_image = data_dir / "bus.jpg"
    output_image = out_dir / "sample_recognition_screenshot.jpg"

    if not input_image.exists():
        urllib.request.urlretrieve(sample_url, input_image)

    model = YOLO("yolov8n.pt")
    cascade = load_face_cascade()
    run_on_image(str(input_image), str(output_image), model, cascade, conf=0.25)


if __name__ == "__main__":
    main()
