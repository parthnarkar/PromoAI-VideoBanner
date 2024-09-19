import numpy as np
from tensorflow.keras.metrics import MeanIoU
from tensorflow.keras.models import load_model
import os
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import sys
import io

# Set default encoding to UTF-8
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

# Define the paths to the model and test images
MODEL_PATH = 'D:/Hackons_BigBasket/PromoAI-VideoBanner/unet_model.h5'
TEST_IMAGE_DIR = 'D:/Hackons_BigBasket/test_images'

def load_test_data(test_image_dir):
    images = []
    
    # Load test images
    for img_file in sorted(os.listdir(test_image_dir)):
        img_path = os.path.join(test_image_dir, img_file)

        # Load the image in RGB
        img = load_img(img_path, target_size=(256, 256), color_mode='rgb')
        img_array = img_to_array(img)
        images.append(img_array)

    images = np.array(images)

    # Normalize the images
    images = images / 255.0
    
    return images

def evaluate_model(predictions, ground_truth):
    # Convert predictions to binary (0 or 1)
    predictions = (predictions > 0.5).astype(np.uint8)

    # Flatten arrays for IoU calculation
    predictions = predictions.flatten()
    ground_truth = ground_truth.flatten()

    # MeanIoU calculation
    iou_metric = MeanIoU(num_classes=2)
    iou_metric.update_state(ground_truth, predictions)
    iou = iou_metric.result().numpy()

    print(f"Mean IoU: {iou}")
    return iou

def visualize_results(test_images, predicted_masks, num_images=2):
    for i in range(num_images):
        plt.figure(figsize=(12, 6))

        # Original Image
        plt.subplot(1, 3, 1)
        plt.title('Original Image')
        plt.imshow(test_images[i])
        plt.axis('off')

        # Predicted Mask
        plt.subplot(1, 3, 2)
        plt.title('Predicted Mask')
        plt.imshow(predicted_masks[i, :, :, 0], cmap='gray')
        plt.axis('off')

        # Save the results to files
        plt.savefig(f'predicted_mask_{i}.png')

        plt.show()

# Load the model
model = load_model(MODEL_PATH)
print("Model loaded successfully!")

# Load test images
test_images = load_test_data(TEST_IMAGE_DIR)
print(f"Loaded {len(test_images)} test images.")

# Predict masks for test images
try:
    predicted_masks = model.predict(test_images)
except Exception as e:
    print(f"Error during model prediction: {e}")
    raise

# Assume ground truth masks are available in a similar directory
# For this example, we'll use predicted masks as placeholders for ground truth masks
ground_truth_masks = predicted_masks  # Placeholder for actual masks

# Evaluate the model
evaluate_model(predicted_masks, ground_truth_masks)

# Visualize results
visualize_results(test_images, predicted_masks)
