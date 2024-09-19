#For Testing U-NET Model
import sys
sys.stdout.reconfigure(encoding='utf-8')

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import numpy as np
import os
import matplotlib.pyplot as plt

# Define paths
MODEL_PATH = 'D:/Hackons_BigBasket/PromoAI-VideoBanner/unet_model.h5'
TEST_IMAGE_DIR = 'D:/Hackons_BigBasket/test_images/'

# Load the trained U-Net model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load test images
def load_test_data(test_image_dir):
    images = []
    for img_file in sorted(os.listdir(test_image_dir)):
        img_path = os.path.join(test_image_dir, img_file)
        img = load_img(img_path, target_size=(256, 256), color_mode='rgb')
        img_array = img_to_array(img)
        images.append(img_array)
    images = np.array(images)
    images = images / 255.0  # Normalize images
    return images

# Load test images
test_images = load_test_data(TEST_IMAGE_DIR)
print(f"Loaded {len(test_images)} test images.")

# Make predictions
predicted_masks = model.predict(test_images)
print("Prediction complete!")

# Display original images and their predicted masks
num_display = min(5, len(test_images))  # Display up to 5 images
for i in range(num_display):
    plt.figure(figsize=(10, 5))

    # Display original image
    plt.subplot(1, 2, 1)
    plt.imshow(test_images[i])
    plt.title("Original Image")
    plt.axis('off')

    # Display predicted mask
    plt.subplot(1, 2, 2)
    plt.imshow(predicted_masks[i].squeeze(), cmap='gray')
    plt.title("Predicted Mask")
    plt.axis('off')

    plt.show()
