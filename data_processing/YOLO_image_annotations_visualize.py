import os
from PIL import Image, ImageDraw
import matplotlib.pyplot as plt

def draw_annotations(image_path, annotation_path):
    """
    Draws YOLO annotations on the image.

    Parameters:
    - image_path: The path to the image file.
    - annotation_path: The path to the YOLO annotation file.
    """
    # Load the image
    with Image.open(image_path) as img:
        img_width, img_height = img.size
        draw = ImageDraw.Draw(img)

        # Read the annotations
        with open(annotation_path, 'r') as file:
            for line in file.readlines():
                class_id, x_center, y_center, width, height = map(float, line.split(' '))

                # Convert YOLO annotations to bounding box coordinates
                x_center *= img_width
                y_center *= img_height
                width *= img_width
                height *= img_height

                # Calculate the corners of the bounding box
                left = x_center - (width / 2)
                top = y_center - (height / 2)
                right = x_center + (width / 2)
                bottom = y_center + (height / 2)

                # Draw the bounding box
                draw.rectangle([left, top, right, bottom], outline="red", width=2)

        # Display the image
        plt.imshow(img)
        plt.axis('off')  # Turn off axis numbers and ticks
        plt.show()

def main():
    image_path = 'Data/Annotations/test/class_0_posy_2_vel_0_centerz_2_273/YOLO_annotations/273_0.jpg'  # Replace with your actual image path
    annotation_path = 'Data/Annotations/test/class_0_posy_2_vel_0_centerz_2_273/YOLO_annotations/273_0.txt'  # Replace with your actual annotation path
    draw_annotations(image_path, annotation_path)

if __name__ == "__main__":
    main()
