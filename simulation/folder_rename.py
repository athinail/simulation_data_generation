import os

# Directory containing the folders
directory = '/home/ailioudi/Documents/Data/SimulationData/noise_std_07sdm_14m_imbalanced_125class1_31class0'  # Replace with the path to your directory

# List all folders in the directory
folders = [f for f in os.listdir(directory) if os.path.isdir(os.path.join(directory, f))]

# Rename each folder
for folder in folders:
    # Extract the last number from the folder name
    new_name = folder.split('_')[-1]

    # Full path for the current and new folder name
    current_folder_path = os.path.join(directory, folder)
    new_folder_path = os.path.join(directory, new_name)

    # Rename the folder
    os.rename(current_folder_path, new_folder_path)
    print(f"Renamed '{folder}' to '{new_name}'")

print("Renaming complete.")
