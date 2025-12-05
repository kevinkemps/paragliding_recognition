"""
Real-time Paraglider Detection from RTSP Stream

This script creates an inference pipeline that processes a live RTSP stream
from a camera and detects paragliders in real-time. The results are rendered
with bounding boxes overlaid on the video.

Requirements:
    - inference package
    - supervision package
    - ROBOFLOW_API_KEY environment variable set in .env file
    - RTSP stream URL accessible and valid

Usage:
    python inf.py
"""
import os
from dotenv import load_dotenv
from inference import InferencePipeline
from inference.core.interfaces.stream.sinks import render_boxes

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please set it in .env file.")

# Get RTSP stream URL from environment variable
rtsp_url = os.getenv("RTSP_URL")
if not rtsp_url:
    raise ValueError("RTSP_URL not found in environment variables. Please set it in .env file.")

# Create an inference pipeline for real-time processing
pipeline = InferencePipeline.init(
    model_id="yolov8n-640",  # YOLOv8 nano model with 640px input size (lightweight for edge)
    video_reference=rtsp_url,  # RTSP stream URL - can be file path, stream URL, or webcam (0)
    on_prediction=render_boxes,  # Render bounding boxes on detected objects
    api_key=api_key,  # Roboflow API key for loading the model
)

# Start the inference pipeline
pipeline.start()

# Wait for the pipeline to complete (runs until interrupted)
pipeline.join()
