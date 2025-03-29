import pygame
import tkinter as tk
from tkinter import filedialog
import tensorflow as tf
import numpy as np
import cv2

# Load the trained model
MODEL_PATH = "traffic_signs_model.h5"  # Ensure your trained model is named 'model.h5'
model = tf.keras.models.load_model(MODEL_PATH)

# Define label mappings for GTSRB dataset
LABELS = [
    "Speed Limit 20 km/h", "Speed Limit 30 km/h", "Speed Limit 50 km/h",
    "Speed Limit 60 km/h", "Speed Limit 70 km/h", "Speed Limit 80 km/h",
    "End of Speed Limit 80 km/h", "Speed Limit 100 km/h", "Speed Limit 120 km/h",
    "No passing", "No passing for vehicles > 3.5 tons", "Right-of-way at next intersection",
    "Priority road", "Yield", "Stop", "No vehicles", "Vehicles > 3.5 tons prohibited",
    "No entry", "General caution", "Dangerous curve left", "Dangerous curve right",
    "Double curve", "Bumpy road", "Slippery road", "Road narrows on the right",
    "Road work", "Traffic signals", "Pedestrians", "Children crossing",
    "Bicycles crossing", "Beware of ice/snow", "Wild animals crossing",
    "End of all speed & passing limits", "Turn right ahead", "Turn left ahead",
    "Ahead only", "Go straight or right", "Go straight or left",
    "Keep right", "Keep left", "Roundabout mandatory", "End of no passing",
    "End of no passing for vehicles > 3.5 tons"
]

# Initialize Pygame
pygame.init()
WIDTH, HEIGHT = 600, 400
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Traffic Sign Recognition")

# Font settings
font = pygame.font.Font(None, 30)
message = "Upload an image to classify"

def classify_image(image_path):
    """ Load and classify an image """
    global message
    try:
        # Load and preprocess image
        img = cv2.imread(image_path)
        img = cv2.resize(img, (30, 30))  # Resize to match the model input
        img = img.astype("float32") / 255.0  # Normalize
        img = np.expand_dims(img, axis=0)  # Add batch dimension

        # Predict
        prediction = model.predict(img)
        predicted_label = np.argmax(prediction)  # Get the highest probability class

        # Display result
        message = f"Prediction: {LABELS[predicted_label]}"
    except Exception as e:
        message = f"Error: {str(e)}"

def upload_image():
    """ Open file dialog to select an image """
    root = tk.Tk()
    root.withdraw()  # Hide the root window
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        classify_image(file_path)

# Main loop
running = True
while running:
    screen.fill((50, 50, 50))  # Dark gray background

    # Display message
    text_surface = font.render(message, True, (255, 255, 255))
    screen.blit(text_surface, (20, 150))

    # Event handling
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_u:  # Press "U" to upload
                upload_image()

    # Display instructions
    instruction_text = font.render("Press 'U' to Upload Image", True, (255, 255, 0))
    screen.blit(instruction_text, (20, 250))

    pygame.display.flip()

pygame.quit()
