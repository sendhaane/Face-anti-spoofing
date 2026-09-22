
import os
import glob
import numpy as np
import tensorflow as tf
from PIL import Image


# ============================================================
# 1. Paths
# ============================================================

BASE_DIR = r"D:\MinifasnetV1"

# ONNX2TF generated SavedModel
SAVED_MODEL = os.path.join(
    BASE_DIR,
    "tensorflow"
)

# Calibration dataset
CALIBRATION_DIR = os.path.join(
    BASE_DIR,
    "calibration_data",
    "faces"
)

# Output INT8 model
OUTPUT_MODEL = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_int8.tflite"
)


# ============================================================
# 2. Find calibration images
# ============================================================

image_paths = []

for ext in ["*.jpg", "*.jpeg", "*.png", "*.bmp"]:
    image_paths.extend(
        glob.glob(
            os.path.join(
                CALIBRATION_DIR,
                ext
            )
        )
    )

# Sort for reproducibility
image_paths = sorted(image_paths)

print("==========================================")
print("INT8 QUANTIZATION")
print("==========================================")

print("\nSavedModel:")
print(SAVED_MODEL)

print("\nCalibration directory:")
print(CALIBRATION_DIR)

print("\nCalibration images found:", len(image_paths))


if len(image_paths) == 0:
    raise RuntimeError(
        "No calibration images found!"
    )


# ============================================================
# 3. Representative dataset
# ============================================================

def representative_dataset():

    for image_path in image_paths:

        # Read image
        image = Image.open(
            image_path
        ).convert("RGB")

        # Ensure exactly 80 x 80
        image = image.resize(
            (80, 80),
            Image.Resampling.BILINEAR
        )

        # Convert to NumPy
        image = np.asarray(
            image,
            dtype=np.float32
        )

        # IMPORTANT:
        # Do NOT divide by 255.
        #
        # MiniFASNet preprocessing uses
        # pixel values in approximately 0-255.
        #
        # Shape before expand:
        # (80, 80, 3)

        # Add batch dimension
        image = np.expand_dims(
            image,
            axis=0
        )

        # Shape:
        # (1, 80, 80, 3)

        yield [image]


# ============================================================
# 4. Create TFLite converter
# ============================================================

print("\nLoading SavedModel...")

converter = tf.lite.TFLiteConverter.from_saved_model(
    SAVED_MODEL
)


# ============================================================
# 5. Enable quantization
# ============================================================

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]


# ============================================================
# 6. Provide calibration data
# ============================================================

converter.representative_dataset = (
    representative_dataset
)


# ============================================================
# 7. Force all supported operations to INT8
# ============================================================

converter.target_spec.supported_ops = [
    tf.lite.OpsSet.TFLITE_BUILTINS_INT8
]


# ============================================================
# 8. Force INT8 input
# ============================================================
converter.inference_input_type = tf.int8
converter.inference_output_type = tf.int8


# ============================================================
# 10. Convert
# ============================================================

print("\nStarting INT8 conversion...")

tflite_model = converter.convert()


# ============================================================
# 11. Save INT8 model
# ============================================================

with open(
    OUTPUT_MODEL,
    "wb"
) as f:

    f.write(
        tflite_model
    )


# ============================================================
# 12. Display results
# ============================================================

model_size_kb = (
    os.path.getsize(OUTPUT_MODEL)
    / 1024
)

model_size_mb = (
    model_size_kb
    / 1024
)

print("\n==========================================")
print("INT8 CONVERSION COMPLETE")
print("==========================================")

print("\nOutput model:")
print(OUTPUT_MODEL)

print(
    "\nModel size:"
)

print(
    f"{model_size_kb:.2f} KB"
)

print(
    f"{model_size_mb:.2f} MB"
)

print("\nCalibration images used:")
print(len(image_paths))

print("\nInput type:")
print("INT8")

print("\nOutput type:")
print("INT8")

print("\nQuantization complete!")

