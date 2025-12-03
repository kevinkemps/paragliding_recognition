# Paraglider Detection

A computer vision project for detecting paragliders in images and video streams using YOLOv8 and YOLOv11 models.

## Project Overview

This project implements paraglider detection across multiple platforms:
- **macOS**: Local development and testing
- **Jetson Device**: Edge deployment on NVIDIA Jetson hardware
- **RTSP Streaming**: Real-time video stream processing

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
├── train/          # Training images and labels
├── valid/          # Validation images and labels
├── test/           # Test images and labels
├── data.yaml       # Dataset configuration for YOLOv8
├── yolov8n.pt      # Pretrained YOLOv8 nano model
└── README files    # Dataset documentation
```

## Pre-trained Models

The project includes several pre-trained models:

- **yolov8n.pt**: YOLOv8 Nano (lightweight, fast)
- **yolov8s.pt**: YOLOv8 Small
- **yolo11n.pt**: YOLOv11 Nano (latest generation)

## Project Structure

```
paraglider_detection/
├── rtsp_stream.ipynb              # RTSP stream processing notebook
├── paraglider_recognition-8/      # YOLOv8 dataset and models
├── jetson_device/                 # Jetson deployment code
├── mac_os/                        # macOS specific implementations
├── job_j57xez9lg_optimized_onnx/  # ONNX optimized model
└── venv/                          # Python virtual environment
```

## Features

- YOLOv8 and YOLOv11 object detection
- RTSP stream support for real-time video processing
- Multi-platform support (macOS, Jetson)
- ONNX model optimization for edge deployment

## Requirements

- Python 3.8+
- OpenCV (cv2)
- YOLOv8 library
- NumPy, Pandas (for data processing)

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
