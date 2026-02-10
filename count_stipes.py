"""
count_stipes.py

Applies a trained U-Net model to a single underwater image, produces a probability map,
post-processes the mask, and estimates kelp stipe count using connected-component analysis.

Expected directory structure (relative to project root):
- images/ : input RGB images
- runs/   : trained model checkpoint at runs/best_unet.pt

Outputs saved to:
- runs/stipe_counts/probability_mask.png
- runs/stipe_counts/binary_mask.png
- runs/stipe_counts/overlay.png
"""

import os

import cv2
import numpy as np
from PIL import Image

import torch
import torchvision.transforms as T

from train_unet import UNetSmall


# ----------------------------
# Configuration
# ----------------------------
BASE_DIR = "."
IMAGE_DIR = os.path.join(BASE_DIR, "images")
MODEL_PATH = os.path.join(BASE_DIR, "runs", "best_unet_FINAL.pt")
OUTPUT_DIR = os.path.join(BASE_DIR, "runs", "stipe_counts")

os.makedirs(OUTPUT_DIR, exist_ok=True)

DEVICE = "cpu"
IMG_SIZE = (512, 512)

IMAGE_NAME = "kelp1.jpg"   # change this to run a different image
THRESHOLD = 0.5
MIN_AREA = 120


# ----------------------------
# Load model
# ----------------------------
model = UNetSmall(in_ch=3, out_ch=1)

checkpoint = torch.load(MODEL_PATH, map_location=DEVICE)
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    model.load_state_dict(checkpoint["model_state_dict"])
else:
    model.load_state_dict(checkpoint)

model.to(DEVICE)
model.eval()


# ----------------------------
# Preprocessing
# ----------------------------
transform = T.Compose([
    T.Resize(IMG_SIZE),
    T.ToTensor(),
])


# ----------------------------
# Post-processing and counting
# ----------------------------
def count_stipes(prob_map: np.ndarray, threshold: float = 0.5, min_area: int = 120):
    """
    Thresholds the probability map, performs morphological cleanup, and counts
    connected components above a minimum area threshold.

    Returns:
      count (int): estimated number of stipes
      cleaned (uint8): cleaned binary mask (0/255)
    """
    binary = (prob_map > threshold).astype(np.uint8) * 255

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    binary = cv2.morphologyEx(binary, cv2.MORPH_CLOSE, kernel, iterations=2)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, kernel, iterations=1)

    num_labels, _, stats, _ = cv2.connectedComponentsWithStats(binary, connectivity=8)

    count = 0
    for i in range(1, num_labels):  # label 0 is background
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            count += 1

    return count, binary


# ----------------------------
# Run inference on a single image
# ----------------------------
image_path = os.path.join(IMAGE_DIR, IMAGE_NAME)
img = Image.open(image_path).convert("RGB")
x = transform(img).unsqueeze(0).to(DEVICE)

with torch.no_grad():
    prob = torch.sigmoid(model(x))[0, 0].cpu().numpy()

count, cleaned = count_stipes(prob, threshold=THRESHOLD, min_area=MIN_AREA)


# ----------------------------
# Save outputs
# ----------------------------
Image.fromarray((prob * 255).astype(np.uint8)).save(
    os.path.join(OUTPUT_DIR, "probability_mask.png")
)

Image.fromarray(cleaned).save(
    os.path.join(OUTPUT_DIR, "binary_mask.png")
)

overlay = np.array(img.resize(IMG_SIZE))
overlay[cleaned > 0] = [255, 255, 255]
Image.fromarray(overlay).save(
    os.path.join(OUTPUT_DIR, "overlay.png")
)

print("===================================")
print(f"Image: {IMAGE_NAME}")
print(f"Predicted stipe count: {count}")
print(f"Outputs saved to: {OUTPUT_DIR}")
print("===================================")
