import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
import sys
sys.stdout.reconfigure(encoding='utf-8')
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from unet_model import build_unet

# Define the image size and path
IMG_HEIGHT, IMG_WIDTH = 256, 256
IMAGE_DIR = 'D:/Hackons_BigBasket/product-images/'  # Path to product images
MASK_DIR = IMAGE_DIR  # Using the same images for masks (temporary solution)

def load_data(image_dir):
    images = []  # List to hold images and masks

    # Load product images and use them as masks as well
    for folder in os.listdir(image_dir):
        folder_path = os.path.join(image_dir, folder)
        for img_file in sorted(os.listdir(folder_path)):
            img_path = os.path.join(folder_path, img_file)

            # Load the image in RGB
            img = load_img(img_path, target_size=(IMG_HEIGHT, IMG_WIDTH), color_mode='rgb')
            img_array = img_to_array(img)
            images.append(img_array)

    images = np.array(images)

    # Normalize the images
    images = images / 255.0

    # Convert images to grayscale for masks
    masks = np.mean(images, axis=-1, keepdims=True)

    # Split data into training and validation sets
    split_index = int(0.8 * len(images))
    x_train, x_val = images[:split_index], images[split_index:]
    y_train, y_val = masks[:split_index], masks[split_index:]

    return (x_train, y_train), (x_val, y_val)

(x_train, y_train), (x_val, y_val) = load_data(IMAGE_DIR)

# Build and train the model
model = build_unet(input_shape=(IMG_HEIGHT, IMG_WIDTH, 3))

history = model.fit(
    x_train, y_train,
    epochs=10,
    batch_size=8,
    validation_data=(x_val, y_val),
    verbose = 1
)

# Save the trained model
model.save('unet_model.h5')

# Evaluate the model on validation data after training
loss, accuracy = model.evaluate(x_val, y_val, verbose=1)
print(f"Validation Loss: {loss}")
print(f"Validation Accuracy: {accuracy}")
