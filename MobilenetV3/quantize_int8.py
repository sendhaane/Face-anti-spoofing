import os
import glob
import numpy as np
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

FP32_MODEL = "models/mn3_tflite/MN3_antispoof_float32.tflite"

OUTPUT_MODEL = "models/MN3_antispoof_int8.tflite"

CALIBRATION_DIR = "calibration_data"


# ============================================================
# Load calibration files
# ============================================================

calibration_files = []

calibration_files.extend(
    glob.glob(
        os.path.join(
            CALIBRATION_DIR,
            "live",
            "*.npy"
        )
    )
)

calibration_files.extend(
    glob.glob(
        os.path.join(
            CALIBRATION_DIR,
            "spoof",
            "*.npy"
        )
    )
)

# Shuffle calibration samples
rng = np.random.default_rng(42)
rng.shuffle(calibration_files)

print("=" * 60)
print("CALIBRATION DATASET")
print("=" * 60)

print("LIVE samples :", len(
    glob.glob(
        os.path.join(CALIBRATION_DIR, "live", "*.npy")
    )
))

print("SPOOF samples:", len(
    glob.glob(
        os.path.join(CALIBRATION_DIR, "spoof", "*.npy")
    )
))

print("Total samples:", len(calibration_files))


# ============================================================
# Representative dataset
# ============================================================

def representative_dataset():

    for file_path in calibration_files:

        data = np.load(file_path)

        # Make sure the calibration sample is float32
        data = data.astype(np.float32)

        yield [data]


# ============================================================
# Load FP32 TFLite model
# ============================================================

print("\n" + "=" * 60)
print("Loading FP32 TFLite model")
print("=" * 60)

converter = tf.lite.TFLiteConverter.from_saved_model(
    "models/mn3_tflite"
)

# ============================================================
# Enable INT8 quantization
# ============================================================

converter.optimizations = [
    tf.lite.Optimize.DEFAULT
]

converter.representative_dataset = representative_dataset


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

tflite_model = converter.convert()


# ============================================================
# Save
# ============================================================

with open(OUTPUT_MODEL, "wb") as f:
    f.write(tflite_model)


print("\n" + "=" * 60)
print("INT8 CONVERSION COMPLETE")
print("=" * 60)

print("Output:", OUTPUT_MODEL)

print(
    "Size:",
    os.path.getsize(OUTPUT_MODEL) / 1024,
    "KB"
)