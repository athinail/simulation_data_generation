import shutil
import numpy as np
import os
import glob
import re
import random
import cv2
import json
from scipy.integrate import odeint
import matplotlib.pyplot as plt

def copy_jpg_files(source_dir, dest_dir):
    # Ensure destination directory exists
    if not os.path.exists(dest_dir):
        os.makedirs(dest_dir)

    # List all files in the source directory
    for filename in os.listdir(source_dir):
        if filename.endswith(".jpg"):
            # Construct full file path
            source_file = os.path.join(source_dir, filename)
            dest_file = os.path.join(dest_dir, filename)

            # Copy file
            shutil.copy2(source_file, dest_file)

def ndarray_to_list(data):
    if isinstance(data, np.ndarray):
        return data.tolist()
    elif isinstance(data, dict):
        return {key: ndarray_to_list(value) for key, value in data.items()}
    elif isinstance(data, list):
        return [ndarray_to_list(item) for item in data]
    else:
        return data
    
# def get_next_video_id(data_directory):
#     """finds the last number in the filename of the video (sample id) and adds +1"""
#     # Find all directories with the pattern "class_*_*_*_*_[number]"
#     pattern = os.path.join(data_directory, "Annotations", "*", "class_*_*_*_*_[0-9]*")
#     directories = glob.glob(pattern)

#     max_id = 0
#     for dir in directories:
#         # Extract the last part of the directory name and try to convert it to an integer
#         try:
#             video_id = int(dir.split('_')[-1])
#             max_id = max(max_id, video_id)
#         except ValueError:
#             continue

#     # Return the next available id
#     return max_id + 1

def get_next_video_id(data_directory):
    """Finds the highest number at the end of the directory names and returns the next number."""
    # Regex pattern to match directory names ending with a number
    pattern = re.compile(r'class_\d+_posy_\d+_vel_\d+_centerz_\d+_(\d+)$')

    max_id = 0
    search_path = os.path.join(data_directory, "Annotations", "*", "")
    for dir_path in glob.glob(search_path):
        dir_name = os.path.basename(dir_path.rstrip(os.sep))  # Remove trailing separator
        match = pattern.search(dir_name)
        if match:
            video_id = int(match.group(1))
            max_id = max(max_id, video_id)

    # Return the next available id
    return max_id + 1