"""
Paraglider Detection Inference on Single Image

This script loads the paraglider detection model and runs inference on a single image.
It demonstrates basic model loading and inference capabilities.

Requirements:
    - inference package
    - ROBOFLOW_API_KEY environment variable set in .env file
    - Test image file path

Usage:
    python inf2.py
"""
import os
from dotenv import load_dotenv
import inference

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please set it in .env file.")

# Define path to test image
test_image = "./paraglider_recognition-8/test/images/P_20231121_114926_jpg.rf.193318024b03b5a48f7ca5bb7c37ae7d.jpg"

# Load the paraglider detection model (version 8)
model = inference.load_roboflow_model("paraglider_recognition/8", api_key=api_key)

# Run inference on the test image
results = model.infer(image=test_image)
