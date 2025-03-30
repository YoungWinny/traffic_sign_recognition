import tkinter as tk
from tkinter import filedialog, Label, Button
import numpy as np
import tensorflow as tf
from PIL import Image, ImageTk
import os

# Define label mappings based on GTSRB dataset
LABELS = {
            0:'Speed limit (20km/h)',
            1:'Speed limit (30km/h)', 
            2:'Speed limit (50km/h)', 
            3:'Speed limit (60km/h)', 
            4:'Speed limit (70km/h)', 
            5:'Speed limit (80km/h)', 
            6:'End of speed limit (80km/h)', 
            7:'Speed limit (100km/h)', 
            8:'Speed limit (120km/h)', 
            9:'No passing', 
            10:'No passing veh over 3.5 tons', 
            11:'Right-of-way at intersection', 
            12:'Priority road', 
            13:'Yield', 
            14:'Stop', 
            15:'No vehicles', 
            16:'Veh > 3.5 tons prohibited', 
            17:'No entry', 
            18:'General caution', 
            19:'Dangerous curve left', 
            20:'Dangerous curve right', 
            21:'Double curve', 
            22:'Bumpy road', 
            23:'Slippery road', 
            24:'Road narrows on the right', 
            25:'Road work', 
            26:'Traffic signals', 
            27:'Pedestrians', 
            28:'Children crossing', 
            29:'Bicycles crossing', 
            30:'Beware of ice/snow',
            31:'Wild animals crossing', 
            32:'End speed + passing limits', 
            33:'Turn right ahead', 
            34:'Turn left ahead', 
            35:'Ahead only', 
            36:'Go straight or right', 
            37:'Go straight or left', 
            38:'Keep right', 
            39:'Keep left', 
            40:'Roundabout mandatory', 
            41:'End of no passing', 
            42:'End no passing veh > 3.5 tons' }


# Load the trained model
MODEL_PATH = "best_model.h5"
model = tf.keras.models.load_model(MODEL_PATH)

# Create directory for saving images
SAVE_DIR = "images"
os.makedirs(SAVE_DIR, exist_ok=True)

def classify_image(image_path):
    """Load and classify an image, then return the predicted label."""
    image = Image.open(image_path).convert("RGB")
    image = image.resize((30, 30))  # Resize to match model input size
    image_array = np.array(image) / 255.0  # Normalize
    image_array = np.expand_dims(image_array, axis=0)  # Expand dimensions for model input
    
    # Make prediction
    prediction = model.predict(image_array)
    class_id = np.argmax(prediction)  # Get the index of the highest probability
    probability = np.max(prediction)  # Get the highest probability
    label = LABELS.get(class_id, "Unknown")  # Get the label corresponding to the predicted class
    
    # Save image with category number as prefix in its filename
    image_name = f"{class_id}_{os.path.basename(image_path)}"
    image.save(os.path.join(SAVE_DIR, image_name))
    
    return class_id, label, probability  # Return class ID, label, and probability

def upload_image():
    """Handle image selection and classification."""
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.webp")])
    if not file_path:
        return
    
    class_id, label, probability = classify_image(file_path)
    
    # Display image
    img = Image.open(file_path)
    img.thumbnail((200, 200))
    img = ImageTk.PhotoImage(img)
    image_label.config(image=img)
    image_label.image = img
    
    # Display result (prediction)
    result_label.config(text=f"Prediction: {label} ({class_id}), Probability: {probability:.3f}")

# Initialize GUI
root = tk.Tk()
root.title("Traffic Sign Recognition")
root.geometry("800x800")

# Upload button
upload_btn = Button(root, text="Upload Image", command=upload_image)
upload_btn.pack()

# Image display
image_label = Label(root)
image_label.pack()

# Result display
result_label = Label(root, text="Prediction: ")
result_label.pack()

# Run GUI
root.mainloop()

