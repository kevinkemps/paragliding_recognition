# Rubik Pi 3 Runtime

This directory is the offline Rubik Pi deployment target for the paragliding recognition pipeline.

The first implementation slice is a local RTSP + TFLite runtime in [run_rtsp_tflite.py](run_rtsp_tflite.py). It loads a local `.tflite` model, runs inference on RTSP frames, annotates detections, and can save an annotated video plus periodic snapshots.

## Install

```bash
cd code_base/paragliding_recognition/rubikpi
pip install -r requirements.txt
```

If your device needs a headless OpenCV build, replace `opencv-python` with `opencv-python-headless` in `requirements.txt`.

## Run

```bash
python run_rtsp_tflite.py \
  --source "$RTSP_URL" \
  --model tflite_models/yolov8s_saved_model/yolov8s_float16.tflite \
  --output-dir output \
  --save-video \
  --snapshot-every 120
```

Optional flags:

- `--display` to show the live annotated stream.
- `--labels path/to/labels.txt` to override the default COCO label map.
- `--frame-stride N` to run inference every Nth frame.
- `--max-frames N` for smoke tests.

## Notes

- The current exported model metadata indicates a COCO-80 detect model at 640x640. If the production Rubik Pi target should be paraglider-only, export that checkpoint before final deployment.
- This runtime is intentionally offline. It does not depend on Roboflow API calls or a local inference server.
