import os
import cv2
import numpy as np
from ultralytics import YOLO
import imutils
import pandas as pd

# Load YOLOv8 model
model = YOLO('yolov8x.pt')  # or 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt' depending on your needs

# Constants
NMS_THRESHOLD = 0.3
MIN_CONFIDENCE = 0.3

def pedestrian_detection(image):
    results = model(image ,conf=MIN_CONFIDENCE, classes=0)
    detections = results[0].boxes.xyxy.numpy()
    confidences = results[0].boxes.conf.numpy()
    class_ids = results[0].boxes.cls.numpy()
    
    boxes = []
    confidences_list = []
    for i, (box, confidence, class_id) in enumerate(zip(detections, confidences, class_ids)):
        if int(class_id) == 0 and confidence > MIN_CONFIDENCE:  # Assuming 'person' class is 0
            x1, y1, x2, y2 = box
            boxes.append([int(x1), int(y1), int(x2), int(y2)])
            confidences_list.append(float(confidence))

    return boxes, confidences_list


# Directory containing images
image_directory = "Farol"
output_base_directory = "output"

# List to store results
results_list = []

# Iterate over each image in the directory
for filename in os.listdir(image_directory):
    if filename.endswith(".JPG") or filename.endswith(".jpeg"):
        image_path = os.path.join(image_directory, filename)
        image = cv2.imread(image_path)

        if image is None:
            print(f"Error loading image: {image_path}")
            continue

        # Create a directory for the current image
        output_dir = os.path.join(output_base_directory, os.path.splitext(filename)[0])
        os.makedirs(output_dir, exist_ok=True)

        # Get dimensions of the image
        height = image.shape[0]
        width = image.shape[1]

        # Divide the image into 10 parts (2 rows and 5 columns)
        height_cutoff = height // 2
        width_cutoff = width // 5

        parts = [
            image[:height_cutoff, :width_cutoff],
            image[:height_cutoff, width_cutoff:2*width_cutoff],
            image[:height_cutoff, 2*width_cutoff:3*width_cutoff],
            image[:height_cutoff, 3*width_cutoff:4*width_cutoff],
            image[:height_cutoff, 4*width_cutoff:],
            image[height_cutoff:, :width_cutoff],
            image[height_cutoff:, width_cutoff:2*width_cutoff],
            image[height_cutoff:, 2*width_cutoff:3*width_cutoff],
            image[height_cutoff:, 3*width_cutoff:4*width_cutoff],
            image[height_cutoff:, 4*width_cutoff:]
        ]

        # Perform pedestrian detection on each part
        total_pedestrian_count = 0

        for idx, img in enumerate(parts):
            boxes, confidences = pedestrian_detection(img)
            pedestrian_count = len(boxes)
            
            for box in boxes:
                x1, y1, x2, y2 = box
                cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

            # Display the number of pedestrians
            cv2.putText(img, f"Pedestrians: {pedestrian_count}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 0), 2)
            print(f"Pedestrians in part {idx+1} of {filename}: {pedestrian_count}")
            
            total_pedestrian_count += pedestrian_count
            
            # Save the output image for each part in the respective directory
            output_path = os.path.join(output_dir, f"part_{idx+1}.jpg")
            cv2.imwrite(output_path, img)

            
        print(f"Total Pedestrians in {filename}: {total_pedestrian_count}")

        # Save results to list
        results_list.append({
            "Image Name": filename,
            "Total Pedestrian Count": total_pedestrian_count
        })

# Save results to a DataFrame
results_df = pd.DataFrame(results_list)

# Save DataFrame to an Excel file
results_df.to_excel("pedestrian_detection_results.xlsx", index=False)
