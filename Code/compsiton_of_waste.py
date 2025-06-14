# ==============================================================================
# YOLOv8 Waste Composition Analysis - Kaggle Notebook
# ==============================================================================
# This script guides you through training a YOLOv8 object detection model
# on a custom waste dataset and then using it to analyze the composition
# of waste in an image.
#
# Follow these steps in your Kaggle Notebook:
# 1. New Notebook: Create a new Kaggle Notebook.
# 2. Add Data:
#    - On the right-hand side, click "+ Add data".
#    - Click "Upload" and select the .zip file you exported from Roboflow.
# 3. Enable GPU:
#    - In the right-hand menu, under "Accelerator", select "GPU".
# 4. Unzip Data:
#    - The dataset will be available at a path like "/kaggle/input/your-dataset-name/".
#    - We will use the !unzip command to extract it into the working directory.
# ==============================================================================

# Step 1: Install Required Libraries
# ------------------------------------------------------------------------------
# The 'ultralytics' package contains the YOLOv8 implementation.
# The -q flag installs it quietly.
pip install ultralytics -q

# Step 2: Unzip Your Dataset
# ------------------------------------------------------------------------------
# IMPORTANT: Replace 'mixed-waste-10' with the actual name of your uploaded
# zip file directory. You can see the name in the "/kaggle/input/" directory.
# This command extracts your dataset into the current working directory.
# !unzip /kaggle/input/mixed-waste-10/archive.zip -d /kaggle/working/

# print("Dataset unzipped successfully!")


# Step 3: Train the YOLOv8 Model
# ------------------------------------------------------------------------------
import os
from ultralytics import YOLO

# Load a pretrained YOLOv8 model. 'yolov8n.pt' is the smallest and fastest.
# You can also use 'yolov8s.pt', 'yolov8m.pt', etc., for better accuracy
# at the cost of speed.
model = YOLO('yolov8n.pt')

# Define the path to your dataset's YAML file. This file was created by Roboflow
# and contains information about your training/validation data and class names.
# After unzipping, it should be in the working directory.
data_yaml_path = '/kaggle/input/wastewise-dataset/data.yaml'

print(f"Starting training with data from: {data_yaml_path}")

# Train the model.
# - data: Path to your .yaml file.
# - epochs: Number of times to loop through the entire dataset. 50-100 is a good start.
# - imgsz: The image size to train on. 640 is standard for YOLOv8.
# - project: The directory where training results will be saved.
# - name: The specific sub-directory for this training run.
results = model.train(
    data=data_yaml_path,
    epochs=75,
    imgsz=640,
    project='waste-detection-training',
    name='run1'
)

print("Training finished!")
# The best trained model will be saved at:
# 'waste-detection-training/run1/weights/best.pt'


# Step 4: Inference and Waste Composition Analysis
# ------------------------------------------------------------------------------
# Now, we'll use the newly trained model to analyze an image.

# Load your custom-trained model weights.
# IMPORTANT: Make sure this path matches where the training results were saved.
trained_model_path = '/kaggle/working/waste-detection-training/run1/weights/best.pt'
trained_model = YOLO(trained_model_path)

print(f"Loaded trained model from: {trained_model_path}")

def analyze_waste_composition(image_path):
    """
    Analyzes a single image to detect waste items and calculate the
    percentage composition based on the area of their bounding boxes.
    """
    print(f"\n--- Analyzing Image: {image_path} ---")

    # Run inference on the image
    inference_results = trained_model(image_path)

    # Dictionary to hold the total area for each detected class
    class_areas = {}
    total_waste_area = 0

    # The result object contains detected boxes, masks, etc.
    # We iterate through the boxes found in the first (and only) image result.
    for r in inference_results:
        # The 'names' dict maps class IDs (e.g., 0, 1, 2) to class names (e.g., 'plastic')
        class_names = r.names

        for box in r.boxes:
            # Get class name
            class_id = int(box.cls[0])
            class_name = class_names[class_id]

            # Get bounding box coordinates (xywh format)
            x, y, w, h = box.xywh[0]
            bbox_area = w * h

            # Aggregate area by class
            if class_name not in class_areas:
                class_areas[class_name] = 0
            class_areas[class_name] += float(bbox_area)

            # Add to total area
            total_waste_area += float(bbox_area)

    # Calculate and display the composition percentage
    print("\nWaste Composition Results:")
    if total_waste_area == 0:
        print("No waste detected in the image.")
        return

    # Sort classes by area for a cleaner report
    sorted_classes = sorted(class_areas.items(), key=lambda item: item[1], reverse=True)

    for class_name, area in sorted_classes:
        percentage = (area / total_waste_area) * 100
        print(f"- {class_name.capitalize()}: {percentage:.2f}%")
