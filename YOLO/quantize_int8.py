import os
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# Configuration
# ============================================================

SAVED_MODEL = r"tflite"

OUTPUT_MODEL = r"tflite/best_int8.tflite"

CALIBRATION_DATASET = r"test_images"

IMAGE_SIZE = 224

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# Collect calibration images
# ============================================================

image_paths = []

for class_name in ["real", "spoof"]:

    folder = os.path.join(
        CALIBRATION_DATASET,
        class_name
    )

    for filename in sorted(os.listdir(folder)):

        if filename.lower().endswith(
            SUPPORTED_EXTENSIONS
        ):

            image_paths.append(
                os.path.join(folder, filename)
            )


print("=" * 70)
print("INT8 POST-TRAINING QUANTIZATION")
print("=" * 70)

print("SavedModel       :", SAVED_MODEL)
print("Calibration data :", CALIBRATION_DATASET)
print("Images found     :", len(image_paths))


# ============================================================
# Preprocessing
# ============================================================

def preprocess_image(image_path):

    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(
            f"Could not read image: {image_path}"
        )

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (IMAGE_SIZE, IMAGE_SIZE),
        interpolation=cv2.INTER_LINEAR
    )

    # Same normalization used during FP32 validation
    image = image.astype(
        np.float32
    ) / 255.0

    # HWC -> NCHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(np.float32)


# ============================================================
# Representative dataset
# ============================================================

def representative_dataset():

    for image_path in image_paths:

        image = preprocess_image(
            image_path
        )

        yield [image]


# ============================================================
# Create converter
# ============================================================

print("\nLoading SavedModel...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    SAVED_MODEL
)


# ============================================================
# Enable quantization
# ============================================================

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = (
    representative_dataset
)


# ============================================================
# Force full INT8
# ============================================================

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]

converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


# ============================================================
# Convert
# ============================================================

print("\nStarting INT8 conversion...")
print("Using all", len(image_paths), "calibration images.")

tflite_model = converter.convert()


# ============================================================
# Save
# ============================================================

os.makedirs(
    os.path.dirname(OUTPUT_MODEL),
    exist_ok=True
)

with open(
    OUTPUT_MODEL,
    "wb"
) as f:

    f.write(tflite_model)


# ============================================================
# Result
# ============================================================

model_size_kb = (
    os.path.getsize(OUTPUT_MODEL)
    / 1024
)

print("\n" + "=" * 70)
print("INT8 CONVERSION COMPLETE")
print("=" * 70)

print("Output model :", OUTPUT_MODEL)
print(f"Model size   : {model_size_kb:.2f} KB")