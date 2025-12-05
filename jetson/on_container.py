"""
Example: Running Workflows with Roboflow Inference SDK

This script demonstrates how to run Roboflow workflows using the Inference SDK.
It connects to a local inference server and executes a workflow that compares
two different YOLO models on the same image.

Requirements:
    - inference-sdk package
    - Local inference server running on http://localhost:9001

Usage:
    python ex.py
"""
from inference_sdk import InferenceHTTPClient

# Initialize inference client pointing to local inference server
# The local server runs models efficiently on edge devices
client = InferenceHTTPClient(
    api_url="http://localhost:9001",  # Local inference server endpoint
    # api_key="" # Optional: for accessing private data and models
)

# Run a workflow that compares two YOLO models
# Workflows allow complex inference pipelines with multiple models
result = client.run_workflow(
    workspace_name="roboflow-docs",  # Roboflow workspace
    workflow_id="model-comparison",  # Workflow identifier
    images={
        "image": "https://media.roboflow.com/workflows/examples/bleachers.jpg"
    },
    parameters={
        "model1": "yolov8n-640",  # YOLOv8 nano model
        "model2": "yolov11n-640"  # YOLOv11 nano model
    }
)

# Display the inference results
print(result)

