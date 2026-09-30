import os
import cv2
import numpy as np
import tensorflow as tf
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    roc_curve
)

from src.face_detector import FaceDetector


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_ROOT = "test_images_organized"

DETECTOR_MODEL = "models/scrfd_int8.tflite"

INT8_MODEL = "models/MN3_antispoof_int8.tflite"

MAX_IMAGES_PER_CLASS = 600

# Same MobileNetV3 normalization
MEAN = np.array(
    [0.5931, 0.4690, 0.4229],
    dtype=np.float32
)

STD = np.array(
    [0.2471, 0.2214, 0.2157],
    dtype=np.float32
)


# ============================================================
# LOAD SCRFD
# ============================================================

print("=" * 70)
print("Loading SCRFD detector")
print("=" * 70)

detector = FaceDetector(DETECTOR_MODEL)

print("SCRFD loaded successfully")


# ============================================================
# LOAD INT8 TFLITE MODEL
# ============================================================

print("\n" + "=" * 70)
print("Loading INT8 anti-spoofing model")
print("=" * 70)

interpreter = tf.lite.Interpreter(
    model_path=INT8_MODEL
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()[0]
output_details = interpreter.get_output_details()[0]

print("Input shape :", input_details["shape"])
print("Input dtype :", input_details["dtype"])

print(
    "Input scale :",
    input_details["quantization"][0]
)

print(
    "Input zero point :",
    input_details["quantization"][1]
)

print("Output shape:", output_details["shape"])
print("Output dtype:", output_details["dtype"])

print(
    "Output scale :",
    output_details["quantization"][0]
)

print(
    "Output zero point :",
    output_details["quantization"][1]
)


# ============================================================
# QUANTIZATION PARAMETERS
# ============================================================

input_scale = input_details["quantization"][0]
input_zero_point = input_details["quantization"][1]

output_scale = output_details["quantization"][0]
output_zero_point = output_details["quantization"][1]


if input_scale == 0:
    raise RuntimeError(
        "Invalid INT8 input scale"
    )

if output_scale == 0:
    raise RuntimeError(
        "Invalid INT8 output scale"
    )


# ============================================================
# PREPROCESSING
# ============================================================

def preprocess_face(face):

    # BGR -> RGB
    face = cv2.cvtColor(
        face,
        cv2.COLOR_BGR2RGB
    )

    # 128x128
    face = cv2.resize(
        face,
        (128, 128),
        interpolation=cv2.INTER_CUBIC
    )

    # [0, 1]
    face = face.astype(
        np.float32
    ) / 255.0

    # MobileNetV3 normalization
    face = (
        face - MEAN
    ) / STD

    # Add batch dimension
    face = np.expand_dims(
        face,
        axis=0
    )

    return face.astype(
        np.float32
    )


# ============================================================
# INT8 INPUT QUANTIZATION
# ============================================================

def quantize_input(x):

    x = np.round(
        x / input_scale
        + input_zero_point
    )

    if input_details["dtype"] == np.int8:

        x = np.clip(
            x,
            -128,
            127
        )

    elif input_details["dtype"] == np.uint8:

        x = np.clip(
            x,
            0,
            255
        )

    return x.astype(
        input_details["dtype"]
    )


# ============================================================
# INT8 OUTPUT DEQUANTIZATION
# ============================================================

def dequantize_output(output):

    return (
        output.astype(np.float32)
        - output_zero_point
    ) * output_scale


# ============================================================
# EVALUATION
# ============================================================

y_true = []
y_pred = []
y_scores = []

total_images = 0
successful_images = 0
no_face = 0
invalid_images = 0

class_counts = {
    "live": {
        "total": 0,
        "success": 0
    },
    "spoof": {
        "total": 0,
        "success": 0
    }
}


# ============================================================
# PROCESS EACH CLASS
# ============================================================

for class_name in ["live", "spoof"]:

    class_dir = os.path.join(
        IMAGE_ROOT,
        class_name
    )

    image_files = []

    for filename in os.listdir(class_dir):

        if filename.lower().endswith(
            (".jpg", ".jpeg", ".png", ".bmp")
        ):
            image_files.append(filename)

    # Deterministic ordering
    image_files.sort()

    # Limit
    image_files = image_files[
        :MAX_IMAGES_PER_CLASS
    ]

    actual_class = (
        0 if class_name == "live"
        else 1
    )

    print("\n" + "=" * 70)
    print(
        f"Evaluating {class_name.upper()} "
        f"({len(image_files)} images)"
    )
    print("=" * 70)

    class_counts[class_name]["total"] = len(
        image_files
    )

    for index, filename in enumerate(
        image_files,
        start=1
    ):

        image_path = os.path.join(
            class_dir,
            filename
        )

        image = cv2.imread(
            image_path
        )

        total_images += 1

        if image is None:

            invalid_images += 1

            continue

        # ----------------------------------------------------
        # SCRFD
        # ----------------------------------------------------

        detections = detector.detect_faces(
            image
        )

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

        x1 = max(
            0,
            min(x1, w - 1)
        )

        y1 = max(
            0,
            min(y1, h - 1)
        )

        x2 = max(
            0,
            min(x2, w)
        )

        y2 = max(
            0,
            min(y2, h)
        )

        if x2 <= x1 or y2 <= y1:

            invalid_images += 1

            continue

        # ----------------------------------------------------
        # FACE CROP
        # ----------------------------------------------------

        face = image[
            y1:y2,
            x1:x2
        ]

        if face.size == 0:

            invalid_images += 1

            continue

        # ----------------------------------------------------
        # PREPROCESS
        # ----------------------------------------------------

        input_float = preprocess_face(
            face
        )

        # ----------------------------------------------------
        # QUANTIZE INPUT
        # ----------------------------------------------------

        input_int8 = quantize_input(
            input_float
        )

        # ----------------------------------------------------
        # INT8 INFERENCE
        # ----------------------------------------------------

        interpreter.set_tensor(
            input_details["index"],
            input_int8
        )

        interpreter.invoke()

        output_int8 = interpreter.get_tensor(
            output_details["index"]
        )

        # ----------------------------------------------------
        # DEQUANTIZE OUTPUT
        # ----------------------------------------------------

        logits = dequantize_output(
            output_int8
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

        predicted_class = int(
            np.argmax(
                logits,
                axis=1
            )[0]
        )

        # Class 1 = SPOOF
        spoof_score = float(
            logits[0, 1]
        )

        y_true.append(
            actual_class
        )

        y_pred.append(
            predicted_class
        )

        y_scores.append(
            spoof_score
        )

        successful_images += 1

        class_counts[class_name]["success"] += 1

        # Progress
        if index % 100 == 0:

            print(
                f"Processed "
                f"{index}/{len(image_files)}"
            )


# ============================================================
# CONVERT TO NUMPY
# ============================================================

y_true = np.array(
    y_true,
    dtype=np.int32
)

y_pred = np.array(
    y_pred,
    dtype=np.int32
)

y_scores = np.array(
    y_scores,
    dtype=np.float32
)


# ============================================================
# BASIC METRICS
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_scores
)


# ============================================================
# CONFUSION MATRIX
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

tn, fp, fn, tp = cm.ravel()


# ============================================================
# APCER / BPCER / ACER
# ============================================================

# LIVE = class 0
# SPOOF = class 1

APCER = (
    fp / (tn + fp)
    if (tn + fp) > 0
    else 0
)

BPCER = (
    fn / (fn + tp)
    if (fn + tp) > 0
    else 0
)

ACER = (
    (APCER + BPCER) / 2
)


# ============================================================
# CLASS ACCURACY
# ============================================================

live_accuracy = (
    tn / (tn + fp)
    if (tn + fp) > 0
    else 0
)

spoof_accuracy = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)




# ============================================================
# RESULTS
# ============================================================

print("\n\n" + "=" * 70)
print("INT8 ANTI-SPOOFING EVALUATION")
print("=" * 70)

print(
    f"\nTotal images              : "
    f"{total_images}"
)

print(
    f"Successfully evaluated    : "
    f"{successful_images}"
)

print(
    f"No face detected          : "
    f"{no_face}"
)

print(
    f"Invalid images            : "
    f"{invalid_images}"
)

print("\nClass-wise processing:")
print(
    f"LIVE  : "
    f"{class_counts['live']['success']}/"
    f"{class_counts['live']['total']}"
)

print(
    f"SPOOF : "
    f"{class_counts['spoof']['success']}/"
    f"{class_counts['spoof']['total']}"
)


print("\n" + "-" * 70)
print("Classification Metrics")
print("-" * 70)

print(
    f"Accuracy                  : "
    f"{accuracy:.4f}"
)

print(
    f"Precision                 : "
    f"{precision:.4f}"
)

print(
    f"Recall                    : "
    f"{recall:.4f}"
)

print(
    f"F1 Score                  : "
    f"{f1:.4f}"
)

print(
    f"ROC-AUC                   : "
    f"{roc_auc:.4f}"
)


print("\n" + "-" * 70)
print("Class Performance")
print("-" * 70)

print(
    f"LIVE accuracy             : "
    f"{live_accuracy:.4f}"
)

print(
    f"SPOOF accuracy            : "
    f"{spoof_accuracy:.4f}"
)


print("\n" + "-" * 70)
print("Face Anti-Spoofing Metrics")
print("-" * 70)

print(
    f"APCER                     : "
    f"{APCER:.4f}"
)

print(
    f"BPCER                     : "
    f"{BPCER:.4f}"
)

print(
    f"ACER                      : "
    f"{ACER:.4f}"
)




print("\n" + "-" * 70)
print("Confusion Matrix")
print("-" * 70)

print(
    f"Actual LIVE  -> "
    f"LIVE: {tn}, SPOOF: {fp}"
)

print(
    f"Actual SPOOF -> "
    f"LIVE: {fn}, SPOOF: {tp}"
)

print("\n" + "=" * 70)