
import os
import shutil
from random import sample

# Define the base directory for 'train' and output directories for images and annotations
base_dir = '/home/ailioudi/Documents/PythonProjects/github_projects/simulation_data_generation/Data2_fixed_ylim/Annotations/test_more_rounded_cornerns_square'
output_images_dir = '/home/ailioudi/Documents/PythonProjects/github_projects/simulation_data_generation/Data2_fixed_ylim/Annotations/YOLO_curated_data_test_more_rounded_cornerns_square/images/val'  # TODO: User should set the correct path
output_annotations_dir = '/home/ailioudi/Documents/PythonProjects/github_projects/simulation_data_generation/Data2_fixed_ylim/Annotations/YOLO_curated_data_test_more_rounded_cornerns_square/labels/val'  # TODO: User should set the correct path

# Create output directories if they do not exist
os.makedirs(output_images_dir, exist_ok=True)
os.makedirs(output_annotations_dir, exist_ok=True)

# Loop over each subfolder in the train directory
for folder_name in os.listdir(base_dir):
    folder_path = os.path.join(base_dir, folder_name)
    
    # Check if it's a directory
    if os.path.isdir(folder_path):
        # Define the path to the YOLO_annotations folder within the subfolder
        annotations_path = os.path.join(folder_path, 'YOLO_annotations')
        
        # List all jpg images in the subfolder
        images = [img for img in os.listdir(folder_path) if img.endswith('.jpg')]
        
        # Randomly select 20 images (or all if there are less than 20)
        selected_images = sample(images, min(20, len(images)))
        
        # Copy selected images to the output directory and their corresponding annotations if they exist
        for img_name in selected_images:
            # Copy image
            src_img_path = os.path.join(folder_path, img_name)
            dest_img_path = os.path.join(output_images_dir, img_name)
            shutil.copy(src_img_path, dest_img_path)
            
            # Check and copy annotation file if it exists
            annotation_name = os.path.splitext(img_name)[0] + '.txt'
            src_annotation_path = os.path.join(annotations_path, annotation_name)
            if os.path.isfile(src_annotation_path):
                dest_annotation_path = os.path.join(output_annotations_dir, annotation_name)
                shutil.copy(src_annotation_path, dest_annotation_path)

print('Copying completed.')
