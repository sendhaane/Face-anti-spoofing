import os
import cv2
import numpy as np

# ============================================================
# Configuration
# ============================================================

DATASET_PATH = "test_images"
OUTPUT_DIR = "conversion_data"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    "representative_images.npy"
)

IMAGE_SIZE = 224

# Number of images from each class
IMAGES_PER_CLASS = 50

CLASS_FOLDERS = [
    "real",
    "spoof"
]

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# Create output directory
# ============================================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# Load and preprocess images
# ============================================================

images = []

for class_name in CLASS_FOLDERS:

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    # Use first N images from each class
    image_files = image_files[:IMAGES_PER_CLASS]

    print(f"\n{class_name.upper()}")
    print(f"Using {len(image_files)} images")

    for filename in image_files:

        image_path = os.path.join(
            folder_path,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            print(f"WARNING: Could not read {image_path}")
            continue

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        image = cv2.resize(
            image,
            (IMAGE_SIZE, IMAGE_SIZE),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # uint8 -> float32
        # Normalize to [0, 1]
        # ----------------------------------------------------

        image = image.astype(
            np.float32
        ) / 255.0

        # ----------------------------------------------------
        # HWC -> CHW
        # ----------------------------------------------------

        image = np.transpose(
            image,
            (2, 0, 1)
        )

        images.append(image)


# ============================================================
# Convert to numpy array
# ============================================================

images = np.stack(images, axis=0)

print("\n" + "=" * 60)
print("REPRESENTATIVE DATA")
print("=" * 60)

print("Shape :", images.shape)
print("Dtype :", images.dtype)
print("Min   :", images.min())
print("Max   :", images.max())


# ============================================================
# Save
# ============================================================

np.save(
    OUTPUT_FILE,
    images
)

print("\nSaved:")
print(OUTPUT_FILE)