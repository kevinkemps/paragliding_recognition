"""
Download Paraglider Dataset from Roboflow

This script downloads the paraglider_recognition dataset (version 8) from Roboflow
in YOLOv8 format for local use on Jetson devices.

Requirements:
    - roboflow package
    - ROBOFLOW_API_KEY environment variable set in .env file

Usage:
    python download_dataset.py
"""
import os
from dotenv import load_dotenv
from roboflow import Roboflow

# Load environment variables from .env file
load_dotenv()

# Get API key from environment variable
api_key = os.getenv("ROBOFLOW_API_KEY")
if not api_key:
    raise ValueError("ROBOFLOW_API_KEY not found in environment variables. Please set it in .env file.")

# Initialize Roboflow with API key
rf = Roboflow(api_key=api_key)

# Get workspace and project
project = rf.workspace("initialtrial").project("paraglider_recognition")

# Download dataset version 8 in YOLOv8 format
dataset = project.version(8).download("yolov8")
