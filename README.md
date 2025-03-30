# Traffic Sign Recognition

## Overview

This project focuses on building a deep learning model to recognize traffic signs using the **German Traffic Sign Recognition Benchmark (GTSRB)** dataset. The trained model is then integrated into a **Tkinter GUI** application, allowing users to upload images and receive predictions for traffic signs. The model is saved as `best_model.h5` and can be reused for further testing.

## Project Structure

- `traffic.py` – Script to train a CNN model on the GTSRB dataset.
- `best_model.h5` – Trained neural network model.
- `images/` – A folder containing 10 different sample traffic sign images for testing.
- `predict_sign.py` – A **Tkinter-based GUI** that allows users to upload images and predict traffic signs.

## Experimentation Process

1. **Dataset Preparation**: We used the **GTSRB dataset**, which consists of images categorized into 43 classes. The dataset was preprocessed to resize images and normalize pixel values.
2. **Model Training**: We trained a Convolutional Neural Network (CNN) using TensorFlow/Keras. The architecture was fine-tuned by adjusting the number of layers, filter sizes, and activation functions.
3. **Testing & Evaluation**: The trained model was evaluated on a separate test set to check accuracy. A Tkinter GUI was implemented to allow real-time image predictions.

## What Worked Well

- The CNN model successfully learned to classify most traffic signs with **high accuracy**.
- The GUI is functional, allowing users to upload images and get **real-time predictions**.
- The program correctly **saves predicted images** in the `predicted_images/` folder with filenames that include the predicted category.
- Proper error handling ensures smooth execution, even when users upload unsupported image formats.

## What Did Not Work Well

- **Blurred Images & Low-Quality Predictions**: The dataset contained many high-quality images, but real-world test images uploaded through the GUI were often **blurry, low-resolution, or captured at odd angles**, making predictions less accurate.
- **Lighting Conditions Affect Predictions**: Images with **excessive brightness or shadows** caused misclassification.
- **Size and Positioning of Signs**: The model struggles with images where the traffic sign is **too small or partially occluded**.

## What Was Noticed

-The model struggled with blurry or low-quality images, leading to incorrect classifications.

-Some signs with similar shapes and colors were sometimes misclassified due to limited differentiation in the dataset.

-The prediction time varied depending on image size, and resizing images before prediction improved performance.

-When saving test images, we ensured that their filenames started with their category number to match the dataset's structure.



## How to Run

1. **Train the Model** (if not already trained):
   ```bash
   python traffic.py ..\data\gtsrb best_model.h5
   ```
2. **Run the GUI Application:**
   ```bash
   python predict_sign.py best_model.h5
   ```
3. **Upload an image** of a traffic sign and view the predicted label.



 
