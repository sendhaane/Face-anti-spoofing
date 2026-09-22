import os
import cv2
import numpy as np

BASE_DIR = r"D:\MinifasnetV1"

FACE_DIR = os.path.join(BASE_DIR, "calibration_data", "faces")
OUTPUT_DIR = os.path.join(BASE_DIR, "test_cind")

os.makedirs(OUTPUT_DIR, exist_ok=True)

# Take the first image from the calibration faces folder
files = [
    f for f in os.listdir(FACE_DIR)
    if f.lower().endswith((".jpg", ".jpeg", ".png"))
]

if not files:
    raise RuntimeError("No face images found.")

image_path = os.path.join(FACE_DIR, files[0])

print("Using:", image_path)

img = cv2.imread(image_path)

if img is None:
    raise RuntimeError("Could not read image.")

print("Original shape:", img.shape)
print("Original dtype:", img.dtype)
print("Original range:", img.min(), img.max())

# HWC -> CHW
img = img.transpose(2, 0, 1)

# Add batch dimension
img = np.expand_dims(img, axis=0)

# Match PyTorch preprocessing
img = img.astype(np.float32)

output_path = os.path.join(OUTPUT_DIR, "input.npy")

np.save(output_path, img)

print()
print("Saved:", output_path)
print("Shape:", img.shape)
print("Dtype:", img.dtype)
print("Range:", img.min(), img.max())