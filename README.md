# Paraglider Detection

A computer vision project for detecting paragliders in images and video streams using YOLOv8 and YOLOv11 models.

## Project Overview

This project implements paraglider detection across multiple platforms:
- **macOS**: Local development and testing
- **Jetson**: Edge deployment and inference examples (see `jetson/`)
- **RTSP Streaming**: Real-time video stream processing

## Quick Start

- Clone the repo and create a Python environment (recommended):

```bash
python3 -m venv .venv
source .venv/bin/activate
# For macOS / local development (recommended):
pip install -r requirements.txt
# For NVIDIA Jetson devices (edge deployment), use the Jetson-specific manifest:
pip install -r jetson/requirements.txt
```

- Create a `.env` file in the repository root and add your API keys and RTSP URL (this file is gitignored):

```env
ROBOFLOW_API_KEY=your_roboflow_api_key
QAI_HUB_API_TOKEN=your_qai_hub_token
RTSP_URL=rtsp://username:password@camera_ip:554/stream
```

## Dataset

The project uses the **paraglider_recognition-8** dataset from Roboflow:
- **Total Images**: 101 images
- **Annotation Format**: YOLOv8
- **Class**: Single class - Paraglider
- **License**: CC BY 4.0
- **Dataset URL**: https://universe.roboflow.com/initialtrial/paraglider_recognition/dataset/8

### Dataset Structure
```
paraglider_recognition-8/
├── train/
├── valid/
├── test/
├── data.yaml
└── yolov8n.pt
```

## Pre-trained Models

The project includes several pre-trained models:

- **yolov8n.pt**: YOLOv8 Nano (lightweight, fast)
- **yolov8s.pt**: YOLOv8 Small
- **yolo11n.pt**: YOLOv11 Nano (latest generation)

## Project Structure (current)

```
paraglider_detection/
├── .env                          # Environment variables (gitignored)
├── README.md
├── download_dataset.py           # helper to download dataset via Roboflow
├── jetson/                       # Jetson examples and helpers
│   ├── direct_inf.py
│   ├── on_container.py
│   ├── stream_inf.py
│   └── timed_run.py
├── mac_os/                       # macOS notebooks and examples
│   ├── inference_on_docker.ipynb
│   ├── rtsp_stream.ipynb
│   └── sample_inference.ipynb
├── paraglider_recognition-8/     # Dataset and baseline model files
│   ├── data.yaml
│   ├── README.dataset.txt
│   ├── README.roboflow.txt
│   ├── train/
│   ├── valid/
│   ├── test/
│   └── yolov8n.pt
├── training/                     # training artifacts and notebooks
│   ├── paraglider_cv-2.ipynb
│   ├── yolo11n.pt
│   ├── yolov8s.pt
│   └── job_j57xez9lg_optimized_onnx/
├── tflite_models/                # converted models and saved models
│   ├── convert_to_tflite.ipynb
│   └── yolov8s_saved_model/
└── venv/                         # (optional) local Python environment
```

## Features

- YOLOv8 and YOLOv11 object detection
- RTSP stream support for real-time video processing
- Multi-platform support (macOS, Jetson)
- ONNX model optimization for edge deployment

## Requirements

- Python 3.8+
- Dependency manifests included in the repo:
	- `requirements.txt` — for macOS/local development (root of repo)
	- `jetson/requirements.txt` — for Jetson / NVIDIA edge devices (device-specific packages)
- Example install commands:
	- macOS/local: `pip install -r requirements.txt`
	- Jetson/edge: `pip install -r jetson/requirements.txt`
- Notes:
	- The Jetson requirements include packages that depend on system drivers (e.g. TensorRT, Jetson.GPIO) and may require additional setup via `apt` or NVIDIA SDK Manager.
	- If you only plan to develop on macOS, install the root `requirements.txt`. For deployment to a Jetson device, prefer `jetson/requirements.txt` on that device.

## Usage

### RTSP Stream Processing

See `rtsp_stream.ipynb` for examples of connecting to and processing RTSP camera streams.

### Model Training

Train on the paraglider_recognition-8 dataset using the `data.yaml` configuration file.

### Edge Deployment

Use the ONNX models in `job_j57xez9lg_optimized_onnx/` for Jetson device deployment.

## References

- [Roboflow - Computer Vision Platform](https://roboflow.com)
- [YOLOv8 Documentation](https://docs.ultralytics.com)
- [YOLOv11 Documentation](https://docs.ultralytics.com)

## Converting to TFLite

I added a conversion script and notebook to convert `.pt` models to `.tflite`:

- `convert_to_tflite.py` — CLI script that programmatically converts models (may have issues on macOS)
- `convert_to_tflite_simple.py` — simpler CLI wrapper that calls `yolo export` (recommended on macOS)
- `convert_to_tflite.ipynb` — interactive notebook with step-by-step instructions

Install conversion dependencies (if not already installed):

```bash
# install ultralytics and tensorflow for conversion
pip install ultralytics tensorflow opencv-python
# or for the simple CLI method
pip install ultralytics
```

Convert a single model (recommended to use the simple CLI script on macOS):

```bash
# convert yolov8n
python convert_to_tflite_simple.py yolov8n

# convert all known models
python convert_to_tflite_simple.py all
```

Converted `.tflite` models are written to `tflite_models/` by default.

Notes:
- On macOS, GPU/CUDA is not available — conversion should be run with `device="cpu"`.
- If you encounter issues in programmatic export, use `convert_to_tflite_simple.py` which calls the `yolo` CLI.

## Running Inference

Examples and helper scripts live under `jetson/` and `mac_os/`:

- `jetson/inf.py`, `inf2.py`, `inf3.py` — Jetson-ready examples (updated to use `.env` variables)
- `mac_os/inference_on_docker.ipynb` — example using a local Docker inference server

Run inference — three quick examples

1) Run the Jetson/edge Python script (uses `ROBOFLOW_API_KEY` from `.env`)

```bash
# Ensure your .env has ROBOFLOW_API_KEY set and activate your env
source .venv/bin/activate
export $(cat .env | xargs)   # or set variables manually in your shell

# Run single-image inference using the helper script
python jetson/inf2.py
```

2) Use the macOS Docker inference notebook

- Open `mac_os/inference_on_docker.ipynb` and run the cells — the notebook shows how to connect to a local Docker inference server and call the model you loaded with the `InferenceHTTPClient`.

3) Run inference with a converted TFLite model (example snippet)

```python
from tflite_runtime.interpreter import Interpreter
import numpy as np
import cv2

# Load model and allocate tensors
interpreter = Interpreter(model_path='tflite_models/yolov8n.tflite')
interpreter.allocate_tensors()

# Prepare input
input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()
img = cv2.imread('paraglider_recognition-8/test/images/P_20231121_114926_jpg.rf.193318024b03b5a48f7ca5bb7c37ae7d.jpg')
img_resized = cv2.resize(img, (640, 640))
input_data = np.expand_dims(img_resized.astype(np.float32) / 255.0, axis=0)

# Set input tensor and invoke
interpreter.set_tensor(input_details[0]['index'], input_data)
interpreter.invoke()

# Get output tensors and post-process according to YOLOv8 output format
output_data = [interpreter.get_tensor(d['index']) for d in output_details]
print('Raw TFLite outputs:', [o.shape for o in output_data])

# Use your regular YOLO post-processing to get boxes, scores, classes
```

Notes:
- The Jetson scripts (`jetson/`) are examples for running with Roboflow/Inference SDK and have been updated to read credentials from `.env`.
- If you converted a model to `.tflite`, put it in `tflite_models/` and adapt the `model_path` above.
- For production/streaming use, prefer the `stream_inf.py` or `direct_inf.py` patterns in `jetson/`.

## Training

Use the `paraglider_recognition-8/data.yaml` configuration to train with Ultralytics YOLOv8:

```bash
# example training command
yolo task=detect mode=train model=yolov8n.pt data=paraglider_recognition-8/data.yaml epochs=25 imgsz=640
```

## Security

- API keys and credentials were moved to a `.env` file and `.gitignore` includes `.env` to avoid accidental commits.

## References

- Roboflow: https://roboflow.com
- Ultralytics / YOLO: https://docs.ultralytics.com
