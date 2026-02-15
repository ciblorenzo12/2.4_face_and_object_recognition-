# Face and Object Recognition Application (Python)

This project detects and recognizes:
- faces (using OpenCV Haar Cascade)
- common objects such as **bus**, **cell phone**, **person**, and more (using YOLOv8 on COCO classes)

## 1) Install

```bash
pip install -r requirements.txt
```

## 2) Run real-time webcam app

```bash
python app.py --mode webcam --camera-id 0
```

friendly one-command launcher (installs dependencies, then opens the facial recognition webcam widget):

```bash
python launch_facial_recognition.py --camera-id 0
```

Press `q` to quit.

## 3) Run on a single image

```bash
python app.py --mode image --image-path path/to/image.jpg --output-path outputs/recognized_output.jpg
```

If you see false phone detections (e.g., wallet or external drive marked as phone), use stricter thresholds:

```bash
python app.py --mode image --image-path path/to/image.jpg --output-path outputs/recognized_output.jpg --conf 0.45 --phone-conf 0.75
```

Uncertain phone detections are now converted to `unknown` by default. Adjust sensitivity with:

```bash
python app.py --mode image --image-path path/to/image.jpg --output-path outputs/recognized_output.jpg --conf 0.45 --phone-conf 0.75 --unknown-margin 0.10
```

To hide uncertain detections instead of showing `unknown`, add:

```bash
python app.py --mode image --image-path path/to/image.jpg --output-path outputs/recognized_output.jpg --hide-unknown
```

You can also restrict detections to only selected classes:

```bash
python app.py --mode image --image-path path/to/image.jpg --output-path outputs/recognized_output.jpg --only-classes "person,bus,cell phone"
```

PowerShell note: use `python app.py ...` (do not include markdown-style links such as `[app.py](...)`).

## 4) Generate sample screenshot artifact

```bash
python generate_sample_screenshot.py
```

This downloads a sample test image and saves:
- `outputs/sample_recognition_screenshot.jpg`

## 5) Calibrate thresholds for your own camera (recommended)

Run this and show typical objects for around 2 minutes:

```bash
python app.py --mode calibrate --camera-id 0 --calibration-seconds 120
```

The app saves recommendations in:
- `outputs/calibration_report.txt`

Then apply the suggested values in your normal webcam command.

## 6) Console app: generate screenshots from your own image paths

Use this helper to pass one or more image paths and auto-create annotated screenshots:

```bash
python make_screenshots.py "C:\path\to\img1.jpg" "C:\path\to\img2.jpg"
```

Outputs are saved in `outputs/` with `_recognized` suffix.

Single image with custom output filename:

```bash
python make_screenshots.py "C:\path\to\img1.jpg" --output "outputs\my_final_screenshot.jpg"
```

Stricter settings:

```bash
python make_screenshots.py "C:\path\to\img1.jpg" --conf 0.45 --phone-conf 0.75 --unknown-margin 0.10
```

Example used in this project for students + bus:

```bash
python make_screenshots.py sample_data/students_and_bus.jpg --output outputs/students_and_bus_screenshot.jpg --only-classes "person,bus" --conf 0.35
```

## 7) Assignment Submission Checklist

Required deliverables and where to find them:

- Source code:
  - `app.py`
  - `make_screenshots.py`
  - `generate_sample_screenshot.py`
  - `requirements.txt`
- Screenshot(s):
  - `outputs/students_and_bus_screenshot.jpg`
  - `outputs/sample_recognition_screenshot.jpg`


## Project Artifacts

- Source code:
  - `app.py`
  - `make_screenshots.py`
  - `generate_sample_screenshot.py`
  - `requirements.txt`
- Screenshot output :
 - `outputs/students_and_bus_screenshot.jpg`
 - `outputs/Screenshot 2026-02-15 092811.png`
    - `This file was taken by my camera to reflect the results on live cam and not just code `
         - `outputs/sample_recognition_screenshot.jpg`


