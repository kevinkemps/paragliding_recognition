from __future__ import annotations

import argparse
import json
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

try:
    from tflite_runtime.interpreter import Interpreter
except ImportError:  # pragma: no cover - fallback for development hosts.
    import tensorflow as tf

    Interpreter = tf.lite.Interpreter


ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_MODEL_PATH = ROOT_DIR / "tflite_models" / "yolov8s_saved_model" / "yolov8s_float16.tflite"
DEFAULT_OUTPUT_DIR = ROOT_DIR / "output"
DEFAULT_IMAGE_SIZE = 640


COCO80_LABELS = [
    "person",
    "bicycle",
    "car",
    "motorcycle",
    "airplane",
    "bus",
    "train",
    "truck",
    "boat",
    "traffic light",
    "fire hydrant",
    "stop sign",
    "parking meter",
    "bench",
    "bird",
    "cat",
    "dog",
    "horse",
    "sheep",
    "cow",
    "elephant",
    "bear",
    "zebra",
    "giraffe",
    "backpack",
    "umbrella",
    "handbag",
    "tie",
    "suitcase",
    "frisbee",
    "skis",
    "snowboard",
    "sports ball",
    "kite",
    "baseball bat",
    "baseball glove",
    "skateboard",
    "surfboard",
    "tennis racket",
    "bottle",
    "wine glass",
    "cup",
    "fork",
    "knife",
    "spoon",
    "bowl",
    "banana",
    "apple",
    "sandwich",
    "orange",
    "broccoli",
    "carrot",
    "hot dog",
    "pizza",
    "donut",
    "cake",
    "chair",
    "couch",
    "potted plant",
    "bed",
    "dining table",
    "toilet",
    "tv",
    "laptop",
    "mouse",
    "remote",
    "keyboard",
    "cell phone",
    "microwave",
    "oven",
    "toaster",
    "sink",
    "refrigerator",
    "book",
    "clock",
    "vase",
    "scissors",
    "teddy bear",
    "hair drier",
    "toothbrush",
]


@dataclass(slots=True)
class Detection:
    class_id: int
    score: float
    x1: float
    y1: float
    x2: float
    y2: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline RTSP TFLite pipeline for Rubik Pi 3")
    parser.add_argument("--source", required=True, help="RTSP URL or camera source")
    parser.add_argument("--model", type=Path, default=DEFAULT_MODEL_PATH, help="Path to the .tflite model")
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR, help="Directory for output artifacts")
    parser.add_argument("--labels", type=Path, help="Optional newline-delimited class label file")
    parser.add_argument("--conf-threshold", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--iou-threshold", type=float, default=0.45, help="IoU threshold for NMS")
    parser.add_argument("--input-size", type=int, default=DEFAULT_IMAGE_SIZE, help="Square model input size")
    parser.add_argument("--num-threads", type=int, default=2, help="Interpreter CPU threads")
    parser.add_argument("--frame-stride", type=int, default=1, help="Run inference every Nth frame")
    parser.add_argument("--max-frames", type=int, default=0, help="Stop after this many frames; 0 means unlimited")
    parser.add_argument("--snapshot-every", type=int, default=0, help="Save an annotated frame every N frames; 0 disables")
    parser.add_argument("--display", action="store_true", help="Show annotated frames in a window")
    parser.add_argument("--save-video", action="store_true", help="Write an annotated video to disk")
    parser.add_argument("--verbose", action="store_true", help="Emit JSON events and timing details")
    return parser.parse_args()


def load_labels(labels_path: Path | None) -> list[str]:
    if labels_path is None:
        return COCO80_LABELS
    labels = [line.strip() for line in labels_path.read_text().splitlines() if line.strip()]
    return labels or COCO80_LABELS


def load_interpreter(model_path: Path, num_threads: int) -> Interpreter:
    if not model_path.exists():
        raise FileNotFoundError(f"Model file does not exist: {model_path}")

    interpreter = Interpreter(model_path=str(model_path), num_threads=num_threads)
    interpreter.allocate_tensors()
    return interpreter


def letterbox(image: np.ndarray, target_size: int) -> tuple[np.ndarray, float, int, int]:
    height, width = image.shape[:2]
    scale = min(target_size / width, target_size / height)
    resized_width = int(round(width * scale))
    resized_height = int(round(height * scale))

    resized = cv2.resize(image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR)
    canvas = np.full((target_size, target_size, 3), 114, dtype=np.uint8)
    pad_x = (target_size - resized_width) // 2
    pad_y = (target_size - resized_height) // 2
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    return canvas, scale, pad_x, pad_y


def preprocess(frame: np.ndarray, input_size: int) -> tuple[np.ndarray, float, int, int]:
    letterboxed, scale, pad_x, pad_y = letterbox(frame, input_size)
    rgb = cv2.cvtColor(letterboxed, cv2.COLOR_BGR2RGB)
    input_tensor = rgb.astype(np.float32) / 255.0
    input_tensor = np.expand_dims(input_tensor, axis=0)
    return input_tensor, scale, pad_x, pad_y


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-values))


def xywh_to_xyxy(boxes: np.ndarray) -> np.ndarray:
    output = np.empty_like(boxes)
    output[:, 0] = boxes[:, 0] - boxes[:, 2] / 2.0
    output[:, 1] = boxes[:, 1] - boxes[:, 3] / 2.0
    output[:, 2] = boxes[:, 0] + boxes[:, 2] / 2.0
    output[:, 3] = boxes[:, 1] + boxes[:, 3] / 2.0
    return output


def rescale_boxes(boxes: np.ndarray, scale: float, pad_x: int, pad_y: int, image_shape: tuple[int, int]) -> np.ndarray:
    boxes = boxes.copy()
    boxes[:, [0, 2]] = (boxes[:, [0, 2]] - pad_x) / scale
    boxes[:, [1, 3]] = (boxes[:, [1, 3]] - pad_y) / scale
    height, width = image_shape
    boxes[:, [0, 2]] = np.clip(boxes[:, [0, 2]], 0, width - 1)
    boxes[:, [1, 3]] = np.clip(boxes[:, [1, 3]], 0, height - 1)
    return boxes


def non_max_suppression(boxes: np.ndarray, scores: np.ndarray, conf_threshold: float, iou_threshold: float) -> list[int]:
    if boxes.size == 0:
        return []

    boxes_xywh = np.column_stack(
        [boxes[:, 0], boxes[:, 1], boxes[:, 2] - boxes[:, 0], boxes[:, 3] - boxes[:, 1]]
    ).tolist()
    keep = cv2.dnn.NMSBoxes(boxes_xywh, scores.tolist(), conf_threshold, iou_threshold)
    if keep is None or len(keep) == 0:
        return []

    keep_array = np.asarray(keep).reshape(-1)
    return [int(index) for index in keep_array]


def parse_yolo_outputs(
    outputs: Sequence[np.ndarray],
    scale: float,
    pad_x: int,
    pad_y: int,
    original_shape: tuple[int, int],
    conf_threshold: float,
    iou_threshold: float,
) -> list[Detection]:
    if not outputs:
        return []

    candidate = np.asarray(outputs[0])

    if candidate.ndim == 3:
        candidate = candidate[0]

    if candidate.ndim != 2:
        return []

    if candidate.shape[0] < candidate.shape[1] and candidate.shape[0] in {84, 85, 6}:
        candidate = candidate.T

    if candidate.shape[1] < 6:
        return []

    boxes = candidate[:, :4].astype(np.float32)
    class_logits = candidate[:, 4:].astype(np.float32)
    class_scores = sigmoid(class_logits)
    class_ids = np.argmax(class_scores, axis=1)
    scores = class_scores[np.arange(class_scores.shape[0]), class_ids]

    keep_mask = scores >= conf_threshold
    boxes = boxes[keep_mask]
    scores = scores[keep_mask]
    class_ids = class_ids[keep_mask]

    if boxes.size == 0:
        return []

    boxes = xywh_to_xyxy(boxes)
    boxes = rescale_boxes(boxes, scale, pad_x, pad_y, original_shape)
    keep = non_max_suppression(boxes, scores, conf_threshold, iou_threshold)

    detections: list[Detection] = []
    for index in keep:
        x1, y1, x2, y2 = boxes[index]
        detections.append(
            Detection(
                class_id=int(class_ids[index]),
                score=float(scores[index]),
                x1=float(x1),
                y1=float(y1),
                x2=float(x2),
                y2=float(y2),
            )
        )
    return detections


def parse_already_postprocessed_outputs(outputs: Sequence[np.ndarray], original_shape: tuple[int, int]) -> list[Detection]:
    candidate = np.asarray(outputs[0])
    if candidate.ndim == 3:
        candidate = candidate[0]

    if candidate.ndim != 2 or candidate.shape[1] < 6:
        return []

    detections: list[Detection] = []
    height, width = original_shape
    for row in candidate:
        x1, y1, x2, y2, score, class_id = row[:6]
        if score <= 0:
            continue
        detections.append(
            Detection(
                class_id=int(class_id),
                score=float(score),
                x1=float(np.clip(x1, 0, width - 1)),
                y1=float(np.clip(y1, 0, height - 1)),
                x2=float(np.clip(x2, 0, width - 1)),
                y2=float(np.clip(y2, 0, height - 1)),
            )
        )
    return detections


def run_inference(interpreter: Interpreter, frame: np.ndarray, input_size: int, conf_threshold: float, iou_threshold: float) -> list[Detection]:
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    input_tensor, scale, pad_x, pad_y = preprocess(frame, input_size)

    interpreter.set_tensor(input_details[0]["index"], input_tensor)
    interpreter.invoke()

    outputs = [interpreter.get_tensor(details["index"]) for details in output_details]
    original_shape = frame.shape[:2]

    if outputs and outputs[0].ndim in {2, 3} and outputs[0].shape[-1] >= 6 and outputs[0].shape[-1] <= 7:
        return parse_already_postprocessed_outputs(outputs, original_shape)

    return parse_yolo_outputs(outputs, scale, pad_x, pad_y, original_shape, conf_threshold, iou_threshold)


def load_video_writer(output_path: Path, frame_shape: tuple[int, int], fps: float) -> cv2.VideoWriter:
    height, width = frame_shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps if fps > 0 else 30.0, (width, height))
    if not writer.isOpened():
        raise RuntimeError(f"Could not open video writer for {output_path}")
    return writer


def annotate_frame(frame: np.ndarray, detections: Sequence[Detection], labels: Sequence[str]) -> np.ndarray:
    annotated = frame.copy()
    for detection in detections:
        label = labels[detection.class_id] if 0 <= detection.class_id < len(labels) else f"class_{detection.class_id}"
        caption = f"{label} {detection.score:.2f}"
        start_point = (int(detection.x1), int(detection.y1))
        end_point = (int(detection.x2), int(detection.y2))
        cv2.rectangle(annotated, start_point, end_point, (0, 255, 0), 2)
        cv2.putText(
            annotated,
            caption,
            (start_point[0], max(start_point[1] - 8, 20)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.55,
            (0, 255, 0),
            2,
            cv2.LINE_AA,
        )
    return annotated


def emit_event(verbose: bool, event: str, **payload: object) -> None:
    if not verbose:
        return
    print(json.dumps({"event": event, **payload}, sort_keys=True))


def main() -> int:
    args = parse_args()
    labels = load_labels(args.labels)
    args.output_dir.mkdir(parents=True, exist_ok=True)

    interpreter = load_interpreter(args.model, args.num_threads)
    input_details = interpreter.get_input_details()
    input_shape = input_details[0]["shape"]
    input_height, input_width = int(input_shape[1]), int(input_shape[2])
    if input_height != input_width:
        raise ValueError(f"Expected a square input tensor, got {input_shape}")

    capture = cv2.VideoCapture(args.source)
    if not capture.isOpened():
        raise RuntimeError(f"Could not open RTSP source: {args.source}")

    fps = capture.get(cv2.CAP_PROP_FPS)
    if not fps or math.isnan(fps) or fps <= 0:
        fps = 30.0

    timestamp = time.strftime("%Y%m%d_%H%M%S")
    video_path = args.output_dir / f"annotated_{timestamp}.mp4"
    snapshot_dir = args.output_dir / f"snapshots_{timestamp}"
    if args.snapshot_every > 0:
        snapshot_dir.mkdir(parents=True, exist_ok=True)

    writer: cv2.VideoWriter | None = None
    if args.save_video:
        frame_shape = (
            int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            int(capture.get(cv2.CAP_PROP_FRAME_WIDTH)),
        )
        writer = load_video_writer(video_path, frame_shape, fps)

    frame_index = 0
    processed_frames = 0
    last_detections: list[Detection] = []
    start_time = time.perf_counter()

    emit_event(args.verbose, "start", source=args.source, model=str(args.model), output_dir=str(args.output_dir))

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            frame_index += 1
            should_infer = args.frame_stride <= 1 or frame_index % args.frame_stride == 0
            if should_infer:
                infer_start = time.perf_counter()
                last_detections = run_inference(
                    interpreter,
                    frame,
                    int(input_width),
                    args.conf_threshold,
                    args.iou_threshold,
                )
                processed_frames += 1
                emit_event(
                    args.verbose,
                    "inference",
                    frame=frame_index,
                    detections=len(last_detections),
                    latency_ms=round((time.perf_counter() - infer_start) * 1000.0, 2),
                )

            annotated = annotate_frame(frame, last_detections, labels)

            if writer is not None:
                writer.write(annotated)

            if args.snapshot_every > 0 and frame_index % args.snapshot_every == 0:
                snapshot_path = snapshot_dir / f"frame_{frame_index:06d}.jpg"
                cv2.imwrite(str(snapshot_path), annotated)

            if args.display:
                cv2.imshow("rubikpi-tflite", annotated)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            if args.max_frames > 0 and frame_index >= args.max_frames:
                break
    finally:
        capture.release()
        if writer is not None:
            writer.release()
        if args.display:
            cv2.destroyAllWindows()

    elapsed = time.perf_counter() - start_time
    summary = {
        "frames_read": frame_index,
        "frames_inferred": processed_frames,
        "seconds": round(elapsed, 3),
        "fps": round(frame_index / elapsed, 2) if elapsed > 0 else 0.0,
        "output_video": str(video_path) if args.save_video else None,
    }
    emit_event(args.verbose, "summary", **summary)
    print(json.dumps(summary, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
