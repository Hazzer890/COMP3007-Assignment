#!/usr/bin/env python3
"""
Train a custom YOLO model for building number detection
using the provided dataset.
"""

import os
from ultralytics import YOLO

def train_building_number_detector():
    """Train a YOLO model for building number detection."""
    
    print("Starting training process...")
    # Load a YOLO11 model
    model = YOLO('yolov5su.pt')  # Start from pretrained model
    print("Training custom YOLO model for building number detection...")
    
    # Train the model
    results = model.train(
        data='/home/20607591/COMP3007-Assignment/mp-assignment-12/data.yaml',  # path to dataset YAML
        epochs=10,  # number of training epochs
        imgsz=640,  # training image size
        batch=16,   # batch size
        device='cuda',  # use CPU (change to 'cuda' if GPU available)
        project='myModel',  # project directory
        name='building_numbers',  # experiment name
        save=True,
        verbose=True
    )
    
    print(f"Training completed. Best model saved to: {results.save_dir}")
    return results.save_dir

def load_yolo_model(model_path):
    """Load a YOLO model from the specified path."""
    try:
        model = YOLO(model_path)
        print(f"Loaded YOLO model from {model_path}")
        return model
    except Exception as e:
        print(f"Error loading YOLO model: {e}")
        return None


if __name__ == "__main__":
    train_building_number_detector()