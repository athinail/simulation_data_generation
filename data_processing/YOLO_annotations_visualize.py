import cv2
import os
import glob

# Paths
video_path = 'circle_corners_gravity.mp4'
annotations_dir = '/home/ailioudi/Documents/PythonProjects/github_projects/simulation_data_generation/Data/Annotations'
output_dir = '/home/ailioudi/Documents/PythonProjects/github_projects/simulation_data_generation/Data/Output'
os.makedirs(output_dir, exist_ok=True)

# Open the video
cap = cv2.VideoCapture(video_path)
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

# Loop through the frames and annotation files
for annotation_file in sorted(glob.glob(os.path.join(annotations_dir, '*.txt'))):
    # Extract the frame index from the file name
    frame_idx = int(os.path.basename(annotation_file).split('_')[1].split('.')[0])

    # Set the video to the correct frame
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
    ret, frame = cap.read()
    if not ret:
        print(f"Failed to grab frame {frame_idx}")
        continue

    # Read the annotation
    with open(annotation_file, 'r') as file:
        for line in file:
            class_id, x_center_norm, y_center_norm, w_norm, h_norm = map(float, line.split())

            # Convert normalized positions back to pixel positions
            x_center = int(x_center_norm * frame_width)
            y_center = int(y_center_norm * frame_height)
            w = int(w_norm * frame_width)
            h = int(h_norm * frame_height)

            # Calculate the top left corner of the bounding box
            x = int(x_center - w / 2)
            y = int(y_center - h / 2)

            # Draw the bounding box on the frame
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(frame, f'Class {int(class_id)}', (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

    # Display the frame with the bounding box
    cv2.imshow('Frame with Bounding Box', frame)
    cv2.waitKey(100)  # Wait 100ms or until a key is pressed

    # Save the frame with the bounding box
    output_frame_path = os.path.join(output_dir, f'frame_{frame_idx}_annotated.jpg')
    cv2.imwrite(output_frame_path, frame)

# Release the video capture object
cap.release()
cv2.destroyAllWindows()
