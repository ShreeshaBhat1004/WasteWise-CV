# WasteWise-CV Project
### 1. Project Overview
The WasteWise-CV project utilizes computer vision to automate the process of waste management through a two-pronged approach. The primary goal is to identify and categorize different types of waste materials to facilitate recycling and proper disposal. This is achieved by employing two distinct YOLOv8 object detection models:

Waste Composition Analysis Model: This model is trained to identify and classify various types of general waste, such as paper, plastic, and biodegradable materials. It then calculates the percentage composition of each waste type within a given image.

Hazardous Object Detection Model: A specialized model trained specifically to detect hazardous materials, with an initial focus on identifying discarded batteries. This is crucial for ensuring safety and preventing environmental contamination.

The project is implemented as a series of Python scripts within a Jupyter Notebook, designed to be executed in a cloud environment like Kaggle or Google Colab, leveraging GPU acceleration for efficient model training.

### 2. Waste Composition Analysis
Methodology
The first part of the project focuses on analyzing the composition of general waste. The process is as follows:

Model Training: A YOLOv8n model, a small and fast version of the YOLO (You Only Look Once) architecture, is used as a base. This pre-trained model is then fine-tuned on a custom dataset named "WasteWise-dataset". The training is conducted for 75 epochs with an image size of 640x640 pixels.

Class Name Mapping: A notable feature of this model is its use of a CLASS_NAME_MAP dictionary. The model is initially trained on coded class names (e.g., '1A', '2B'). After detection, these codes are translated into human-readable, descriptive names (e.g., 'Paper Cardboard', 'Plastic'). This approach allows for a more organized and flexible class management system.

Inference and Analysis: Once the model is trained, it can be used for inference on new images. The analyze_waste_composition_with_mapping function takes an image path, performs object detection, and then calculates the area of each detected waste item's bounding box. The areas of items belonging to the same category are aggregated. The final output is a percentage breakdown of the different waste materials in the image.

##### Results
The model was trained on a dataset containing 705 training images and 201 validation images. After 75 epochs, the best-performing model achieved the following on the validation set:

mAP50-95: 0.727

mAP50: 0.937

An inference test on the image 000784_jpg.rf.66f5f0c22536a76f97981417a3505a13.jpg resulted in a 100% plastic composition, demonstrating the model's ability to correctly identify and quantify waste in a sample image.

### 3. Hazardous Object Detection
Methodology
The second component of the project is a dedicated model for identifying hazardous objects, specifically batteries. The workflow for this model is:

Dataset Acquisition: The model is trained on the "l-battery" dataset, which is downloaded from Roboflow, a platform for managing and annotating computer vision datasets.

Model Training: Similar to the waste composition model, a pre-trained YOLOv8n model is fine-tuned on this specialized dataset. The training is performed for 50 epochs.

Validation and Inference: After training, the model's performance is evaluated on a validation set. The validated model is then used to run inference on a directory of test images, with the results saved for review.

##### Results
The hazardous object detection model was trained on a dataset from Roboflow. The validation metrics for the trained model were:

mAP50-95: 0.1912

mAP50: 0.3508

mAP75: 0.1753

The inference was run on 18 test images, and the model was able to successfully detect batteries in several of them. The results, including bounding boxes drawn around the detected batteries, are saved as image files.

4. Conclusion
The WasteWise-CV project successfully demonstrates the potential of using YOLOv8 for automated waste management. The dual-model approach allows for both a broad-spectrum analysis of waste composition and a targeted detection of hazardous materials. The use of class name mapping provides a flexible and user-friendly way to manage waste categories.

Future improvements could include expanding the range of detectable hazardous materials, further optimizing the models for speed and accuracy, and integrating the system with physical sorting mechanisms for a fully automated waste processing pipeline.
