Repository to create data for object detection. It creates a training set with circle and squares and it creates a test set with circle with sort of corners and a square with more curvy corners.

# Simulation Data Generation 2D

A Python-based simulation framework for generating synthetic training and test data for object detection tasks. This repository creates annotated datasets of shapes (circles and squares) with various physical dynamics, suitable for YOLO-based object detection models.

## Overview

The simulator generates:
- **Training set**: Perfect circles and squares with different physical behaviors
- **Test set**: Circles with rounded corners and squares with curved edges to test model generalization

Each simulation includes:
- Video files of object motion
- YOLO-format bounding box annotations
- Position, velocity, and acceleration data (ground truth and noisy measurements)
- Physics parameters and settings

## Repository Structure   
simulation_data_generation_2D/
├── Data2_fixed_ylim/ # Generated data output directory (gitignored)
│ └── Annotations/
│ ├── test/ # Test set annotations and videos
│ ├── train/ # Training set annotations and videos
│ └── YOLOCuratedData/ # YOLO-formatted dataset structure
├── data_processing/ # Data processing utilities
│ ├── copy_images_and_annotations.py # Converts to YOLOv6 structure
│ ├── pixel_to_world_axis_transformation.py
│ ├── video_tracking.py
│ ├── YOLO_annotations_visualize.py
│ └── YOLO_image_annotations_visualize.py
└── simulation/ # Main simulation code
├── main.py # Main script to generate data
├── ShapeDynamicsSimulator.py # Shape classes with physics
├── YOLO_annotations_extract2.py # YOLO annotation extraction
├── utils.py # Helper functions
├── model_accuracy.py # Noise modeling for measurements
├── folder_rename.py # Dataset organization utilities
└── test.py # Experimental shape creation tests




## Installation

1. Clone the repository:
```bash
git clone https://github.com/yourusername/simulation_data_generation_2D.git
cd simulation_data_generation_2D
```
## Install requirements
pip install -r requirements.txt


## Usage

Generate Sample Data
To generate a small sample dataset:


