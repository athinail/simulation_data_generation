import numpy as np
import os
import glob
import random
import cv2
import json
from scipy.integrate import odeint
import matplotlib
matplotlib.use('Agg')  
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from pathlib import Path
from ShapeDynamicsSimulator import CircleSimulationGravity, FancyBoxSimulationGravitySpringDamper, FancyBoxSimulationGravity
import matplotlib.colors as mcolors
from YOLO_annotations_extract2 import VideoAnnotationProcessor
from utils import copy_jpg_files, ndarray_to_list, get_next_video_id

PROJECT_ROOT = Path(__file__).resolve().parent.parent
data_directory = PROJECT_ROOT / 'Data2_fixed_ylim'
os.makedirs(data_directory, exist_ok=True)

def save_data_to_annotations_json(file_name, bounding_box_coordinates, position_data, velocity_data, acceleration_data, frameTime_data):
    annotations = {
        "boundingBoxCoordinates": ndarray_to_list(bounding_box_coordinates),
        "position": ndarray_to_list(position_data),
        "velocity": ndarray_to_list(velocity_data),
        "acceleration": ndarray_to_list(acceleration_data),
        "objectAngularAngle": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in frameTime_data],
        "objectAngularSpeed": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in frameTime_data],
        "frameTime": [float(time) for time in frameTime_data]
    }

    with open(file_name, 'w') as f:
        json.dump(annotations, f, indent=4, ensure_ascii=False)

def save_settings_json(file_name, class_category):
    settings = {
        "waterCurrentStrength": [{"x": 0.0, "y": 0.0, "z": 0.1654302179813385} for _ in range(10)],
        "lightIntensity": 0.9258633255958557,
        "lightDirection": {"x": 109.0, "y": 164.0},
        "m": 0.7433659434318543,
        "I_x": 0.0025634118355810644,
        "I_y": 0.002482375595718622,
        "I_z": 0.000709395797457546,
        "CG_x": 0.0,
        "CG_y": 0.0,
        "CG_z": 0.0,
        "V": 0.0009486281778663397,
        "A_x": 0.014943252317607403,
        "A_y": 0.01332345325499773,
        "A_z": 0.005688331555575132,
        "cd_x": 0.8199999928474426,
        "cd_y": 0.8199999928474426,
        "cd_z": 1.0299999713897706,
        "mu_s": 0.4000000059604645,
        "mu_d": 0.30000001192092898,
        "g": 9.800000190734864,
        "rho_w": 997.0,
        "dampCoefficient": 0.0000019999999949504856,
        "objectType": "Plastic Water Bottle" if class_category == 0 else "Fish",
        "swimForceVector": {"x": 0.0, "y": 0.0, "z": 0.0}
    }

    with open(file_name, 'w') as f:
        json.dump(settings, f, indent=4, ensure_ascii=False)

def save_all_data_to_json(file_name, video_filename, class_category, bounding_box_coordinates, position_data, velocity_data, acceleration_data, position_data_noisy, velocity_data_noisy, acceleration_data_noisy, frameTime_data):
    annotations_all = {
        "boundingBoxCoordinates": ndarray_to_list(bounding_box_coordinates),
        "position": ndarray_to_list(position_data),
        "velocity": ndarray_to_list(velocity_data),
        "acceleration": ndarray_to_list(acceleration_data),
        "objectAngularAngle": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in frameTime_data],
        "objectAngularSpeed": [{"x": 0.0, "y": 0.0, "z": 0.0} for _ in frameTime_data],
        "frameTime": [float(time) for time in frameTime_data],
        "position_data_noisy" : ndarray_to_list(position_data_noisy),
        "velocity_data_noisy" : ndarray_to_list(velocity_data_noisy), 
        "acceleration_data_noisy" : ndarray_to_list(acceleration_data_noisy),
        "class_category" : class_category,
        "video_filename" : video_filename

    }

    with open(file_name, 'w') as f:
        json.dump(annotations_all, f, indent=4, ensure_ascii=False)

def run_simulation(set, class_category, shape_params, init_conditions, timesteps, color, args):
    if set == "train":
        # Circle - gravity
        if class_category == 0:
            sim = CircleSimulationGravity(m=args[0], k1=args[1], k2=args[2], c1=args[3], c2=args[4], init_conditions=init_conditions, time_steps=timesteps, shape_params=shape_params, color=color, noise_mean=0, noise_std_dev=0.1)
        # Square - gravity spring damper
        elif class_category == 1:
            sim = FancyBoxSimulationGravitySpringDamper(m=args[0], k1=args[1], k2=args[2], c1=args[3], c2=args[4], init_conditions=init_conditions, time_steps=timesteps, shape_params=shape_params, color=color, noise_mean=0, noise_std_dev=0.1)
    elif set == "test":
        # Circle with corners - gravity
        if class_category == 0:
            sim = FancyBoxSimulationGravity(m=args[0], k1=args[1], k2=args[2], c1=args[3], c2=args[4], init_conditions=init_conditions, time_steps=timesteps, shape_params=shape_params, color=color, noise_mean=0, noise_std_dev=0.1)
        # Square with rounded corners - gravity spring damper
        elif class_category == 1:
            sim = FancyBoxSimulationGravitySpringDamper(m=args[0], k1=args[1], k2=args[2], c1=args[3], c2=args[4], init_conditions=init_conditions, time_steps=timesteps, shape_params=shape_params, color=color, noise_mean=0, noise_std_dev=0.1)
    return sim


def main(generate_samples=False):
    # Mapping for folder names
    set_folder_mapping = {
        "train": "train_try",
        "test": "test_try"
    }
    
    sets = ["train", "test"]
    class_categories = [0 , 1] # 0 - circle (gravity), 1 - square (mass, spring, damping)
    timesteps = np.linspace(0, 10, 301)
    color = "green"
    
    initial_positions_x = [13, 14, 15, 16, 17, 18]
    initial_positions_y = [-1, 0, 1, 2, 3, 4]
    initial_velocities_x = [0, 0, 0, 0, 0]
    initial_velocities_y = [0, 0, 0, 0, 0]
    args = (1.0, 3.0, 15.0, 0.3, 0.7)

    initial_conditions_range = [(x, vx, y, vy) for x in initial_positions_x for vx in initial_velocities_x for y in initial_positions_y for vy in initial_velocities_y]

    video_id = get_next_video_id(data_directory)
    for set_range in sets:
        if generate_samples:
            # Create only a few samples
            random_initial_conditions = random.sample(initial_conditions_range, 2)
        else:
            # Use all initial conditions
            random_initial_conditions = initial_conditions_range

        for init_cond in random_initial_conditions:
            for class_category in class_categories:
                if set_range == "train":
                    if class_category == 0: # circle
                        sim_params = {'width': 4.5, 'height': 4.5}
                    elif class_category == 1: # square
                        sim_params = {'width': 5.5, 'height': 4.5, 'pad': 0.0, 'color': 'green'}
                elif set_range == "test":
                    if class_category == 0: # circle with corners
                        sim_params = {'width': 1, 'height': 0.5, 'pad': 2}
                    elif class_category == 1: # square with rounded corners
                        sim_params = {'width': 2, 'height': 1, 'pad': 2, 'color': 'green'}
                
                sim = run_simulation(set=set_range, class_category=class_category, shape_params=sim_params, init_conditions=init_cond, timesteps=timesteps, color=color, args=args)
                sim.animate()
                
                file = f"class_{class_category}_initcond_{init_cond}_{video_id}"
                # Use the mapped folder name (train_try or test_try)
                folder_name = set_folder_mapping[set_range]
                videofile_directory_name = os.path.join(data_directory, "Annotations", folder_name, file)
                os.makedirs(videofile_directory_name, exist_ok=True)
                YOLO_annotations_directory = os.path.join(videofile_directory_name, "YOLO_annotations")
                os.makedirs(YOLO_annotations_directory, exist_ok=True)

                sim.save_animation(os.path.join(videofile_directory_name, f"{file}.mp4"))

                position_data = sim.position_data
                velocity_data = sim.velocity_data
                acceleration_data = sim.acceleration_data
                position_data_measured = sim.position_data_noisy
                velocity_data_measured = sim.velocity_data_noisy
                acceleration_data_measured = sim.acceleration_data_noisy
                frameTime_data = timesteps

                video_path = os.path.join(videofile_directory_name, f"{file}.mp4")
                processor = VideoAnnotationProcessor(video_path, YOLO_annotations_directory, frames_dir=videofile_directory_name, object_class_id=class_category, video_id=video_id)
                processor.process_video()
                bounding_box_coordinates = processor.get_bounding_box_coordinates()

                annotations_path = os.path.join(videofile_directory_name, "annotations.json")
                save_data_to_annotations_json(annotations_path, bounding_box_coordinates, position_data_measured, velocity_data_measured, acceleration_data_measured, frameTime_data)
                save_all_data_to_json(os.path.join(videofile_directory_name, "annotations_all.json"), f"{file}.mp4", class_category, bounding_box_coordinates, position_data, velocity_data, acceleration_data, position_data_measured, velocity_data_measured, acceleration_data_measured, frameTime_data)

                settings_path = os.path.join(videofile_directory_name, "settings.json")
                save_settings_json(settings_path, class_category)

                copy_jpg_files(YOLO_annotations_directory, videofile_directory_name)

                video_id += 1

if __name__ == "__main__":
    main(generate_samples=True)