import numpy as np
import os
import glob
import random
import cv2
import json
from scipy.integrate import odeint
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.animation import FuncAnimation, FFMpegWriter
from ShapeDynamicsSimulator import CircleSimulationGravity, FancyBoxSimulationGravitySpringDamper, FancyBoxSimulationGravity
import matplotlib.colors as mcolors
from YOLO_annotations_extract2 import VideoAnnotationProcessor
from utils import copy_jpg_files, ndarray_to_list, get_next_video_id

data_directory = '/home/ailioudi/Documents/PythonProjects/github_projects/simulation_data_generation/Data2_fixed_ylim'
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

def run_simulation( set, class_category, shape_params, init_conditions, timesteps, color):
    if set == "train":
        # Circle - gravity
        if class_category == 0:
            sim = CircleSimulationGravity(m=1.0, k=3.0, c=0.3, init_conditions=init_conditions, time_steps=timesteps, shape_params=shape_params, color=color, noise_mean=0, noise_std_dev=14.2)
        # Square - gravity spring damper
        elif class_category == 1:
            sim = FancyBoxSimulationGravitySpringDamper(m=1.0, k=3.0, c=0.3, init_conditions=init_conditions, time_steps=timesteps, shape_params=shape_params, color=color, noise_mean=0, noise_std_dev=0.7)
    elif set == "test":
        # Circle with corners - gravity
        if class_category == 0:
            sim = FancyBoxSimulationGravity(m=1.0, k=3.0, c=0.3, init_conditions=init_conditions, time_steps=timesteps, shape_params=shape_params, color=color, noise_mean=0, noise_std_dev=14.2)
        # Square with rounded corners - gravity spring damper
        elif class_category == 1:
            sim = FancyBoxSimulationGravitySpringDamper(m=1.0, k=3.0, c=0.3, init_conditions=init_conditions, time_steps=timesteps, shape_params=shape_params, color=color, noise_mean=0, noise_std_dev=0.7)
    return sim



def main():
    sets = ["train", "test"]
    class_categories = [0 , 1] # 0 - circle (gravity), 1 - sqare (mass,sprinf,damping)
    timesteps = np.linspace(0, 10, 301)
    color = "green"
    # object_colors = ["green", "indigo", "blue", "turquoise",  "fuchsia", "gold", "darkred" ,"lightblue", "peru", "tomato"]
    # background_colors = ["white", "silver", "greenyellow", "khaki"]
    
    # Range of initial conditions
    initial_positions_y = [2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    initial_velocities = [0, 10, 12, 14, 15]
    center_z_range = [2,4,6,8,10,12,14,16,18,20]

    # Run Circle Simulation
    # Run Square Simulation

    video_id = get_next_video_id(data_directory)
    for set_range in sets:
        #remove the following three lines when generating the whole dataset
        random_positions_y = random.sample(initial_positions_y, 5)
        random_velocities = random.sample(initial_velocities, 5)
        random_center_z_range = random.sample(center_z_range, 5)
        for init_pos_y in random_positions_y:
            for init_vel in random_velocities:
                for center_z in random_center_z_range:
                    init_conditions = [init_pos_y, init_vel]
                
                    # sim_params = {'center_z': center_z, 'center_y': init_conditions[0], 'width': 4.5, 'height': 4.5}
                    # shape_params_square = {'center_z': center_z, 'center_y': init_conditions[0], 'width': 4.5, 'height': 3.3, 'pad': 0.0}

                    for class_category in class_categories:
                        # Define file name for saving the animation
                        file = f"class_{class_category}_posy_{init_pos_y}_vel_{init_vel}_centerz_{center_z}_{video_id}"
                        videofile_directory_name = os.path.join(data_directory, "Annotations", f"{set_range}", file)
                        os.makedirs(videofile_directory_name, exist_ok=True)
                        YOLO_annotations_directory = os.path.join(videofile_directory_name, "YOLO_annotations")
                        os.makedirs(YOLO_annotations_directory, exist_ok=True)
                        if set_range == "train":
                            if class_category == 0: #circle
                                sim_params = {'center_z': center_z, 'center_y': init_conditions[0], 'width': 4.5, 'height': 4.5}
                            elif class_category == 1: #square
                                sim_params = {'center_z': center_z, 'center_y': init_conditions[0], 'width': 4.5, 'height': 3.3, 'pad': 0.0}
                        elif  set_range == "test": 
                            if class_category == 0: #circle with corners
                                sim_params = {'center_z': 5, 'center_y': -0.05, 'width': 1, 'height': 0.5, 'pad': 2}
                            elif class_category ==1: #square with rounded corners
                                sim_params = {'center_z': 5, 'center_y': -0.05, 'width': 3, 'height': 2, 'pad': 0.8}
                        sim = run_simulation(set=set_range, class_category=class_category, shape_params=sim_params, init_conditions=init_conditions, timesteps=timesteps, color=color)
                        sim.animate()
                        sim.save_animation(os.path.join(videofile_directory_name, f"{file}.mp4"))

                        # Extract data from the simulation
                        position_data = sim.position_data
                        velocity_data = sim.velocity_data
                        acceleration_data = sim.acceleration_data
                        position_data_measured = sim.position_data_noisy
                        velocity_data_measured = sim.velocity_data_noisy
                        acceleration_data_measured = sim.acceleration_data_noisy
                        frameTime_data = timesteps

                        # Process the video to get bounding box coordinates
                        
                        video_path = os.path.join(videofile_directory_name, f"{file}.mp4")
                        processor = VideoAnnotationProcessor(video_path, YOLO_annotations_directory , frames_dir = videofile_directory_name, object_class_id = class_category, video_id= video_id)
                        processor.process_video()
                        bounding_box_coordinates = processor.get_bounding_box_coordinates()

                        # Save all data to JSON
                        annotations_path = os.path.join(videofile_directory_name, "annotations.json")
                        save_data_to_annotations_json(annotations_path, bounding_box_coordinates, position_data_measured, velocity_data_measured, acceleration_data_measured, frameTime_data)
                        save_all_data_to_json(os.path.join(videofile_directory_name, "annotations_all.json"), f"{file}.mp4", class_category, bounding_box_coordinates, position_data, velocity_data, acceleration_data, position_data_measured, velocity_data_measured, acceleration_data_measured, frameTime_data)

                        # Call the function to save settings.json
                        settings_path = os.path.join(videofile_directory_name, "settings.json")
                        save_settings_json(settings_path, class_category)

                        # copy jpg to the same folder with settings.json and annotations.json to get the same structure required for thesis-mathieu
                        copy_jpg_files(YOLO_annotations_directory, videofile_directory_name)

                        video_id += 1


if __name__ == "__main__":
    main()