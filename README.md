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
simulation_data_generation
|--Data -> the folder where the generated data annotations are saved. 
    |-- Annotations -> the same as above
            |-- test -> contains the folders with the annotations for the test set. The structure 				 in each of the folders contained in the test are according to Mathieu's 				 thesis structure, so that the data are applicable to his code
            | --train ->contains the folders with the annotations for the train set. The structure 				 in each of the folders contained in the test are according to Mathieu's 				 thesis structure, so that the data are applicable to his code
            |--YOLOCuratedData -> contains the structure needed for the annotations in YOLOv6 					   repository
|--data_processing ->
    |--copy_images_and_annotations.py -> copies the images and annotations to the folders 						according to YOLOv6 structure (images, labels , train, val 						etc.)
|--simulation -> contains the scripts for generating the data
    |-- folder_rename.py -> renames the folders in the test and train folder to 0,1,2 etc. to 				     comply with the naming of the respective folders in mathieu's 				     implementation
    |--YOLO_anotations_extract2.py -> contains the class to extract YOLO labels during generating 				               the data annotations
    |-- utils.py _> contains methods to process the data
    |-- test.py -> an initial script that I used to experiment with the libraries to create shapes
    |--model_accuracy.py -> script that defines the std of noise that can be added to the 				     simulation siglans to simulate the "measured signals" 
    |-- ShapeDynamicsSimulator.py --> contains the class that creates the shapes (circle, 						square..etc.) which follow a specific motion (gravity, 						damper-spring etc.)
    
    
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
