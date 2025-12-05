"""
Paraglider Detection Performance Benchmark

This script benchmarks the paraglider detection model by running inference
multiple times on a single image and measuring performance metrics.
Useful for evaluating edge device performance and optimization.

Requirements:
    - inference package
    - supervision package
    - opencv-python (cv2)
    - ROBOFLOW_API_KEY environment variable set in .env file
    - Test image file path

Usage:
    python inf3.py
"""
import os
from dotenv import load_dotenv
from inference import get_model
import supervision as sv
import cv2
import time

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please set it in .env file.")

# Define path to test image for benchmarking
image_file = "./paraglider_recognition-8/test/images/P_20231121_114926_jpg.rf.193318024b03b5a48f7ca5bb7c37ae7d.jpg"

# Load the image using OpenCV
image = cv2.imread(image_file)

# Load the paraglider detection model (version 8)
model = get_model(model_id="paraglider_recognition/8", api_key=api_key)

# Benchmark inference performance by running 10 iterations
time_list = []
for i in range(10):
    start = time.time()
    results = model.infer(image)[0]
    end = time.time()
    time_list.append(end - start)

# Calculate average inference time
average = sum(time_list) / len(time_list)

# Display results
print(f"Result: {results}")
print(f"Last inference took {end - start:.4f} seconds")
print(f"Average inference time (10 runs): {average:.4f} seconds")
