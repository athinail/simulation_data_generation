import cv2
import numpy as np

video_path = 'Data2_fixed_ylim/Annotations/train/class_1_posy_18_vel_0_centerz_2_62/class_1_posy_18_vel_0_centerz_2_62.mp4'
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)  # Frame rate of the video

positions = []  # List to store positions
velocities = []  # List to store velocities
times = []  # List to store time of each frame

frame_idx = 0  # Index of the frame
previous_position = None

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert to HSV color space
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    
    # Define the range for green color and create a mask
    lower_green = np.array([40, 40, 40])
    upper_green = np.array([70, 255, 255])
    mask = cv2.inRange(hsv_frame, lower_green, upper_green)

    # Find contours
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if contours:
        # Assuming the largest contour is the object
        c = max(contours, key=cv2.contourArea)
        (x, y), radius = cv2.minEnclosingCircle(c)
        current_position = np.array([x, y])
        positions.append(current_position)  # Save position
        times.append(frame_idx / fps)  # Save time
        
        # If we have a previous position, calculate velocity
        if previous_position is not None:
            velocity = (current_position - previous_position) * fps  # pixels per second
            velocities.append(velocity)
        
        previous_position = current_position

    frame_idx += 1  # Increment frame index

cap.release()

# Save positions and velocities to files
np.savetxt('positions.txt', positions, delimiter=',')
np.savetxt('velocities.txt', velocities, delimiter=',')
np.savetxt('times.txt', times, delimiter=',')

# 'positions' contains the position of the object in pixels for each frame
# 'velocities' contains the velocity of the object in pixels/second for each frame
# 'times' contains the timestamp for each frame in seconds
