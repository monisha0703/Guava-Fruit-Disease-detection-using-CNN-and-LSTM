from __future__ import division, print_function
import os
import numpy as np
import tensorflow as tf
from tensorflow.compat.v1 import ConfigProto
from tensorflow.compat.v1 import InteractiveSession
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tkinter import filedialog, Tk, Label, Button
from PIL import Image, ImageTk

# TensorFlow GPU configuration
config = ConfigProto()
config.gpu_options.per_process_gpu_memory_fraction = 0.5
config.gpu_options.allow_growth = True
session = InteractiveSession(config=config)

# Load the model
model = load_model('Guava_fruit_disease_model.h5', compile=False)

# Prediction function
def model_predict(img_path, model):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = x / 255.0  # Normalize the image
    x = np.expand_dims(x, axis=0)

    preds = model.predict(x)  # Get prediction probabilities
    confidence = np.max(preds)  # Get the highest probability (confidence)
    preds = np.argmax(preds, axis=1)  # Get the class index

    labels = ["Canker","Disease Free","Dot","Mummification","Phytopthora","RedRust","Rust","Scab","Styler & Root"]



    
    
    label = labels[preds[0]]

    return label, confidence

# Function to handle image display and result output
def show_image_and_result(img_path, result, confidence):
    img = Image.open(img_path)
    img = img.resize((400, 400), Image.Resampling.LANCZOS)
    img_tk = ImageTk.PhotoImage(img)

    img_label.config(image=img_tk)
    img_label.image = img_tk  # Keep a reference to avoid garbage collection

    result_text = f"Predicted Result: {result}"
    result_label.config(text=result_text)

# Function to select an image and display it along with the result
def select_image():
    img_path = filedialog.askopenfilename(title="Select Image", filetypes=[("Image files", "*.jpg; *.jpeg; *.png")])
    if img_path:
        result, confidence = model_predict(img_path, model)
        show_image_and_result(img_path, result, confidence)
    else:
        print("No image selected.")

# Initialize Tkinter
root = Tk()
root.title("Guava Fruit Disease Detection")
root.geometry("450x600")

# Image display label
img_label = Label(root)
img_label.pack()

# Prediction result label
result_label = Label(root, text="", font=("Helvetica", 16))
result_label.pack(pady=20)

# Button to select image
select_button = Button(root, text="Select Image", command=select_image, font=("Helvetica", 14))
select_button.pack(pady=10)

# Run the Tkinter event loop
root.mainloop()
