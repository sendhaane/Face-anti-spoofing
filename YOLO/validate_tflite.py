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
    classification_report
)


# ============================================================
# Configuration
# ============================================================

MODEL_PATH = "tflite/best_float32.tflite"
DATASET_PATH = "test_images"

CLASS_MAPPING = {
    "real": 0,
    "spoof": 1
}

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# Load TFLite model
# ============================================================

print("Loading TFLite model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("\nModel information")
print("-" * 70)

print("Input name  :", input_details[0]["name"])
print("Input shape :", input_details[0]["shape"])
print("Input dtype :", input_details[0]["dtype"])

print("Output name :", output_details[0]["name"])
print("Output shape:", output_details[0]["shape"])
print("Output dtype:", output_details[0]["dtype"])


# ============================================================
# Input information
# ============================================================

input_shape = input_details[0]["shape"]
input_dtype = input_details[0]["dtype"]

input_scale, input_zero_point = input_details[0]["quantization"]

print("Quantization:", input_scale, input_zero_point)


# ============================================================
# Preprocessing
# ============================================================

def preprocess(image):

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    # Float32 normalization
    image = image.astype(np.float32) / 255.0

    # Determine expected layout
    if input_shape[1] == 3:

        # NHWC -> NCHW
        image = np.transpose(
            image,
            (2, 0, 1)
        )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    # Handle input dtype
    if input_dtype == np.float32:

        image = image.astype(np.float32)

    elif input_dtype == np.uint8:

        image = image / input_scale + input_zero_point
        image = np.clip(
            image,
            0,
            255
        ).astype(np.uint8)

    elif input_dtype == np.int8:

        image = image / input_scale + input_zero_point
        image = np.clip(
            image,
            -128,
            127
        ).astype(np.int8)

    else:

        raise ValueError(
            f"Unsupported input dtype: {input_dtype}"
        )

    return image


# ============================================================
# Evaluation
# ============================================================

y_true = []
y_pred = []
y_scores = []

total_images = 0
successful = 0
invalid_images = 0


for folder_name, true_class in CLASS_MAPPING.items():

    folder_path = os.path.join(
        DATASET_PATH,
        folder_name
    )

    print("\n" + "=" * 70)
    print(f"Evaluating: {folder_name.upper()}")
    print("=" * 70)

    image_files = sorted([
        f
        for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ])

    print("Images:", len(image_files))

    for image_name in image_files:

        image_path = os.path.join(
            folder_path,
            image_name
        )

        total_images += 1

        image = cv2.imread(image_path)

        if image is None:

            print(
                "Could not read:",
                image_path
            )

            invalid_images += 1
            continue

        # Preprocess
        input_data = preprocess(image)

        # Run inference
        interpreter.set_tensor(
            input_details[0]["index"],
            input_data
        )

        interpreter.invoke()

        output = interpreter.get_tensor(
            output_details[0]["index"]
        )[0]

        # ----------------------------------------------------
        # Output handling
        # ----------------------------------------------------

        output = output.astype(np.float32)

        # If output is logits, convert to probabilities.
        # For a normal Float32 YOLO classifier this should
        # normally already represent classification scores.
        exp_output = np.exp(
            output - np.max(output)
        )

        probabilities = (
            exp_output /
            np.sum(exp_output)
        )

        predicted_class = int(
            np.argmax(probabilities)
        )

        spoof_probability = float(
            probabilities[1]
        )

        # Store
        y_true.append(true_class)
        y_pred.append(predicted_class)
        y_scores.append(spoof_probability)

        successful += 1


# ============================================================
# Metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1
)

roc_auc = roc_auc_score(
    y_true,
    y_scores
)

cm = confusion_matrix(
    y_true,
    y_pred
)


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 70)
print("YOLOv8 FACE ANTI-SPOOFING TFLITE EVALUATION")
print("=" * 70)

print(f"Model                  : {MODEL_PATH}")
print(f"Dataset                : {DATASET_PATH}")

print()
print(f"Total images           : {total_images}")
print(f"Successfully evaluated : {successful}")
print(f"Invalid images         : {invalid_images}")

print("\n" + "-" * 70)
print("Classification Metrics")
print("-" * 70)

print(f"Accuracy       : {accuracy:.4f} ({accuracy * 100:.2f}%)")
print(f"Precision      : {precision:.4f} ({precision * 100:.2f}%)")
print(f"Recall         : {recall:.4f} ({recall * 100:.2f}%)")
print(f"F1 Score       : {f1:.4f} ({f1 * 100:.2f}%)")
print(f"ROC-AUC        : {roc_auc:.4f} ({roc_auc * 100:.2f}%)")


print("\n" + "-" * 70)
print("Confusion Matrix")
print("-" * 70)

print(
    "                 Predicted"
)

print(
    "              REAL    SPOOF"
)

print(
    f"Actual REAL     {cm[0][0]:4d}    {cm[0][1]:4d}"
)

print(
    f"Actual SPOOF    {cm[1][0]:4d}    {cm[1][1]:4d}"
)


# ============================================================
# Class accuracy
# ============================================================

real_accuracy = (
    cm[0][0] /
    (cm[0][0] + cm[0][1])
)

spoof_accuracy = (
    cm[1][1] /
    (cm[1][0] + cm[1][1])
)

print("\n" + "-" * 70)
print("Class Accuracy")
print("-" * 70)

print(
    f"REAL accuracy  : "
    f"{real_accuracy:.4f} "
    f"({real_accuracy * 100:.2f}%)"
)

print(
    f"SPOOF accuracy : "
    f"{spoof_accuracy:.4f} "
    f"({spoof_accuracy * 100:.2f}%)"
)


# ============================================================
# Classification report
# ============================================================

print("\n" + "-" * 70)
print("Classification Report")
print("-" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        target_names=["REAL", "SPOOF"],
        digits=2
    )
)

print("=" * 70)