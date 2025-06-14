# ==============================================================================
#
#           End-to-End Lithium Battery Detection with YOLOv8
#
# This script provides a complete pipeline to:
# 1. Set up the environment for Google Colab.
# 2. Download the specified "l-battery" dataset from Roboflow.
# 3. Train a YOLOv8 object detection model using transfer learning.
# 4. Validate the trained model's performance.
# 5. Run inference on new images to detect batteries.
#
# Environment: This script is designed to be run in a Google Colab notebook.
# To run: Open a new notebook at https://colab.research.google.com/,
# and paste this entire code block into a cell.
#
# ==============================================================================

# Step 1: Environment Setup
# -------------------------
# We start by installing the 'ultralytics' and 'roboflow' libraries,
# and checking for GPU availability. A GPU will significantly speed up training.

print("Step 1: Setting up the environment...")
# Install the Ultralytics and Roboflow packages
!pip install ultralytics roboflow -q

# Check if a GPU is available. Training is much faster on a GPU.
!nvidia-smi

import os
import glob
import yaml
from IPython.display import display, Image
from ultralytics import YOLO
from roboflow import Roboflow

print("\nEnvironment setup complete.")

# ==============================================================================
# Step 2: Dataset Acquisition and Preparation
# -------------------------------------------
# Using the official Roboflow library to download the specific "l-battery" dataset.
# This is the most reliable method for accessing Roboflow datasets.

print("\nStep 2: Downloading and preparing the l-battery dataset...")

DATASET_YAML_PATH = None
try:
    # Initialize the Roboflow client. This key is for the public l-battery dataset.
    rf = Roboflow(api_key="XpJD5pijjMYQr4LrQlwy")
    project = rf.workspace("tokyo-university-7pakh").project("l-battery")
    dataset = project.version(1).download("yolov8")

    DATASET_YAML_PATH = os.path.join(dataset.location, 'data.yaml')
    print(f"Dataset downloaded and prepared. YAML path: {DATASET_YAML_PATH}")
except Exception as e:
    print(f"An error occurred while downloading the dataset: {e}")
    print("Please check the Roboflow workspace/project ID and ensure the dataset is public.")


if DATASET_YAML_PATH:
    # Let's inspect the data.yaml file to understand the dataset structure
    with open(DATASET_YAML_PATH, 'r') as f:
        data_yaml = yaml.safe_load(f)
        print("\nDataset configuration (data.yaml):")
        print(data_yaml)

# ==============================================================================
# Step 3: Model Training
# ----------------------
# Now, we will train our object detection model on the battery dataset.
if DATASET_YAML_PATH:
    print("\nStep 3: Starting model training...")

    # Load a pre-trained YOLOv8 model (yolov8n is the smallest and fastest)
    model = YOLO('yolov8n.pt')

    # Train the model on our custom dataset
    results = model.train(
        data=DATASET_YAML_PATH,
        epochs=50,
        imgsz=640,
        project='runs/detect',
        name='l_battery_detection_run',
        exist_ok=True # Overwrite previous runs if they exist
    )

    print("\nModel training complete.")
    print("Training results and model weights are saved in 'runs/detect/l_battery_detection_run'")
    TRAINING_SUCCESSFUL = True
else:
    print("\nSkipping training due to dataset download failure.")
    TRAINING_SUCCESSFUL = False

# ==============================================================================
# Step 4: Model Validation
# ------------------------
# After training, it's essential to evaluate the model's performance on the
# validation set.
if TRAINING_SUCCESSFUL:
    print("\nStep 4: Validating the model...")

    # Find the path to the best trained model weights
    BEST_MODEL_PATH = os.path.join('runs/detect/l_battery_detection_run/weights/best.pt')

    # Load the best trained model
    model = YOLO(BEST_MODEL_PATH)

    # Run validation using the same dataset yaml file
    metrics = model.val(data=DATASET_YAML_PATH)
    print("\nValidation metrics:")
    print(f"  - mAP50-95: {metrics.box.map:.4f}")
    print(f"  - mAP50: {metrics.box.map50:.4f}")
    print(f"  - mAP75: {metrics.box.map75:.4f}")
else:
    print("\nSkipping validation because training did not complete.")

# ==============================================================================
# Step 5: Inference - Detecting Batteries in New Images
# -------------------------------------------------
# Using our trained model to detect batteries in the test images.
if TRAINING_SUCCESSFUL:
    print("\nStep 5: Running inference on test images...")

    # Path to the folder with test images, derived from the dataset location
    TEST_IMAGES_DIR = os.path.join(dataset.location, 'test/images')
    INFERENCE_RESULTS_DIR = 'runs/detect/inference_results_battery'
    
    # Ensure the test directory exists before running prediction
    if os.path.exists(TEST_IMAGES_DIR) and len(os.listdir(TEST_IMAGES_DIR)) > 0:
        predict_results = model.predict(source=TEST_IMAGES_DIR, save=True, project=INFERENCE_RESULTS_DIR, name='run1', exist_ok=True)
        print(f"\nInference complete. Results saved in '{INFERENCE_RESULTS_DIR}/run1'")

        # Display a few of the result images
        print("\nDisplaying a few prediction results:")
        result_images = glob.glob(f'{INFERENCE_RESULTS_DIR}/run1/*.jpg')
        for img_path in result_images[:5]: # Display the first 5 results
            display(Image(filename=img_path))
    else:
        print(f"Test images directory is empty or not found at: {TEST_IMAGES_DIR}")
else:
    print("\nSkipping inference because training did not complete.")