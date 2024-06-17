import cv2
import numpy as np
import os

class VideoAnnotationProcessor:
    def __init__(self, video_path, annotations_dir, frames_dir, object_class_id, video_id):
        self.video_path = video_path
        self.annotations_dir = annotations_dir
        self.frames_dir  = frames_dir
        self.object_class_id = object_class_id
        self.video_id = video_id
        os.makedirs(annotations_dir, exist_ok=True)

    def process_video(self):
        cap = cv2.VideoCapture(self.video_path)
        frame_idx = 0

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Process frame for bounding box and save frame and annotation
            self.process_frame(frame, frame_idx)
            frame_idx += 1

        cap.release()

    def process_frame(self, frame, frame_idx):
        hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
        lower_green = np.array([40, 40, 40]) # Adjust these values for your video
        upper_green = np.array([70, 255, 255]) # Adjust these values for your video
        mask = cv2.inRange(hsv_frame, lower_green, upper_green)

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        if contours:
            rect_contour = max(contours, key=cv2.contourArea)
            x, y, w, h = cv2.boundingRect(rect_contour)

            # Normalize the bounding box coordinates for YOLO format
            frame_width, frame_height = frame.shape[1], frame.shape[0]
            x_center_norm = (x + w / 2) / frame_width
            y_center_norm = (y + h / 2) / frame_height
            w_norm = w / frame_width
            h_norm = h / frame_height

            # Save the frame and annotation
            self.save_frame(frame, frame_idx)
            self.save_annotation(frame_idx, x_center_norm, y_center_norm, w_norm, h_norm)

    def save_frame(self, frame, frame_idx):
        frame_path = os.path.join(self.annotations_dir, f'{self.video_id}_{frame_idx}.jpg')
        cv2.imwrite(frame_path, frame)

    def save_annotation(self, frame_idx, x_center, y_center, width, height):
        annotation_path = os.path.join(self.annotations_dir, f'{self.video_id}_{frame_idx}.txt')
        with open(annotation_path, 'w') as f:
            f.write(f'{self.object_class_id} {x_center} {y_center} {width} {height}\n')

    def get_bounding_box_coordinates(self):
        cap = cv2.VideoCapture(self.video_path)
        bounding_box_coordinates = []

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            lower_green = np.array([40, 40, 40])
            upper_green = np.array([70, 255, 255])
            mask = cv2.inRange(hsv_frame, lower_green, upper_green)

            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            if contours:
                rect_contour = max(contours, key=cv2.contourArea)
                x, y, w, h = cv2.boundingRect(rect_contour)
                bounding_box_coordinates.append({
                    "xMin": float(x),
                    "yMin": float(y),
                    "xMax": float(x + w),
                    "yMax": float(y + h)
                })

        cap.release()
        return bounding_box_coordinates


# # Example Usage
# video_path = "circle_corners_gravity.mp4"
# annotations_dir = "/home/ailioudi/Documents/PythonProjects/github_projects/simulation_data_generation/Data/Annotations"
# frames_dir = "/home/ailioudi/Documents/PythonProjects/github_projects/simulation_data_generation/Data/Annotations"
# processor = VideoAnnotationProcessor(video_path, annotations_dir, frames_dir, object_class_id=0, video_id=1)
# processor.process_video()