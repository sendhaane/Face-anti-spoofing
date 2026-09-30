import os
import cv2
import numpy as np
import random

from src.face_detector import FaceDetector


# ============================================================
# Configuration
# ============================================================

IMAGE_ROOT = "test_images_organized"
OUTPUT_ROOT = "calibration_data"

DETECTOR_MODEL = "models/scrfd_int8.tflite"

MAX_PER_CLASS = 600

SEED = 42


# MobileNetV3 normalization
MEAN = np.array(
    [0.5931, 0.4690, 0.4229],
    dtype=np.float32
)

STD = np.array(
    [0.2471, 0.2214, 0.2157],
    dtype=np.float32
)


# ============================================================
# Reproducibility
# ============================================================

random.seed(SEED)
np.random.seed(SEED)


# ============================================================
# Create directories
# ============================================================

os.makedirs(
    os.path.join(OUTPUT_ROOT, "live"),
    exist_ok=True
)

os.makedirs(
    os.path.join(OUTPUT_ROOT, "spoof"),
    exist_ok=True
)


# ============================================================
# Load SCRFD
# ============================================================

print("=" * 60)
print("Loading SCRFD detector")
print("=" * 60)

detector = FaceDetector(DETECTOR_MODEL)

print("SCRFD loaded successfully")


# ============================================================
# MobileNetV3 preprocessing
# ============================================================

def preprocess_face(face):
    """
    Same preprocessing used by the validated
    PyTorch / ONNX / Float32 TFLite pipeline.

    BGR
      -> RGB
      -> resize 128x128
      -> /255
      -> mean/std normalization
      -> NHWC
    """

    # BGR -> RGB
    face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)

    # Resize
    face = cv2.resize(
        face,
        (128, 128),
        interpolation=cv2.INTER_CUBIC
    )

    # float32 [0, 1]
    face = face.astype(np.float32) / 255.0

    # Normalize
    face = (face - MEAN) / STD

    # TFLite expects NHWC
    face = np.expand_dims(face, axis=0)

    return face.astype(np.float32)


# ============================================================
# Process one class
# ============================================================

def process_class(class_name):

    input_dir = os.path.join(
        IMAGE_ROOT,
        class_name
    )

    output_dir = os.path.join(
        OUTPUT_ROOT,
        class_name
    )

    images = []

    for filename in os.listdir(input_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            images.append(filename)

    # Shuffle so calibration data is not ordered
    random.shuffle(images)

    print("\n" + "=" * 60)
    print(f"Processing {class_name.upper()}")
    print("=" * 60)

    print("Images found:", len(images))

    processed = 0
    no_face = 0
    invalid = 0

    for filename in images:

        if processed >= MAX_PER_CLASS:
            break

        image_path = os.path.join(
            input_dir,
            filename
        )

        image = cv2.imread(image_path)

        if image is None:
            invalid += 1
            continue

        # ----------------------------------------------------
        # SCRFD detection
        # ----------------------------------------------------

        detections = detector.detect_faces(image)

        if not detections:
            no_face += 1
            continue

        # Highest confidence face
        best_detection = max(
            detections,
            key=lambda d: d[4]
        )

        x1 = int(best_detection[0])
        y1 = int(best_detection[1])
        x2 = int(best_detection[2])
        y2 = int(best_detection[3])

        # Clamp coordinates
        h, w = image.shape[:2]

        x1 = max(0, min(x1, w - 1))
        y1 = max(0, min(y1, h - 1))
        x2 = max(0, min(x2, w))
        y2 = max(0, min(y2, h))

        if x2 <= x1 or y2 <= y1:
            invalid += 1
            continue

        # ----------------------------------------------------
        # Crop face
        # ----------------------------------------------------

        face = image[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:
            invalid += 1
            continue

        # ----------------------------------------------------
        # MobileNet preprocessing
        # ----------------------------------------------------

        processed_face = preprocess_face(face)

        # ----------------------------------------------------
        # Save
        # ----------------------------------------------------

        output_name = (
            f"{class_name}_{processed + 1:04d}.npy"
        )

        output_path = os.path.join(
            output_dir,
            output_name
        )

        np.save(
            output_path,
            processed_face
        )

        processed += 1

        if processed % 100 == 0:
            print(
                f"Processed: {processed}/{MAX_PER_CLASS}"
            )

    print("\nFinished:", class_name.upper())
    print("Saved       :", processed)
    print("No face     :", no_face)
    print("Invalid     :", invalid)

    return processed, no_face, invalid


# ============================================================
# Main
# ============================================================

live_result = process_class("live")
spoof_result = process_class("spoof")


# ============================================================
# Summary
# ============================================================

print("\n" + "=" * 60)
print("CALIBRATION DATASET SUMMARY")
print("=" * 60)

print(
    "LIVE  saved:",
    live_result[0]
)

print(
    "SPOOF saved:",
    spoof_result[0]
)

print(
    "TOTAL saved:",
    live_result[0] + spoof_result[0]
)

print("\nCalibration data location:")
print(OUTPUT_ROOT)