import os
import cv2
import numpy as np

from ultralytics import YOLO

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

MODEL_PATH = "best.pt"
DATASET_PATH = "test_images"

SUPPORTED_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# Load model
# ============================================================

print("Loading YOLO classification model...")

model = YOLO(MODEL_PATH)

print("\nModel information")
print("-" * 60)
print("Task   :", model.task)
print("Classes:", model.names)

print("\nExpected:")
print("0 -> real")
print("1 -> spoof")


# ============================================================
# Dataset class mapping
# ============================================================

CLASS_MAPPING = {
    "real": 0,
    "spoof": 1
}


# ============================================================
# Evaluation storage
# ============================================================

y_true = []
y_pred = []

# Probability of SPOOF
y_scores = []

total_images = 0
successful = 0
invalid_images = 0


# ============================================================
# Evaluate dataset
# ============================================================

for folder_name, true_class in CLASS_MAPPING.items():

    folder_path = os.path.join(
        DATASET_PATH,
        folder_name
    )

    if not os.path.isdir(folder_path):

        print(
            f"\nWARNING: Folder not found: "
            f"{folder_path}"
        )

        continue

    image_files = [
        f for f in os.listdir(folder_path)
        if f.lower().endswith(SUPPORTED_EXTENSIONS)
    ]

    print("\n" + "=" * 60)
    print(f"Evaluating: {folder_name.upper()}")
    print(f"Images    : {len(image_files)}")
    print("=" * 60)

    for image_name in image_files:

        image_path = os.path.join(
            folder_path,
            image_name
        )

        total_images += 1

        # ----------------------------------------------------
        # Read image
        # ----------------------------------------------------

        image = cv2.imread(image_path)

        if image is None:

            print(
                f"Could not read: {image_path}"
            )

            invalid_images += 1
            continue

        # ----------------------------------------------------
        # YOLO classification inference
        # ----------------------------------------------------

        results = model(
            image,
            verbose=False
        )

        result = results[0]

        if result.probs is None:

            print(
                f"No classification output: "
                f"{image_path}"
            )

            invalid_images += 1
            continue

        # ----------------------------------------------------
        # Get probabilities
        # ----------------------------------------------------

        probabilities = (
            result.probs.data
            .cpu()
            .numpy()
        )

        real_probability = float(
            probabilities[0]
        )

        spoof_probability = float(
            probabilities[1]
        )

        predicted_class = int(
            np.argmax(probabilities)
        )

        # ----------------------------------------------------
        # Store results
        # ----------------------------------------------------

        y_true.append(true_class)
        y_pred.append(predicted_class)

        # Probability of positive class = SPOOF
        y_scores.append(spoof_probability)

        successful += 1


# ============================================================
# Basic results
# ============================================================

print("\n")
print("=" * 70)
print("YOLOv8 FACE ANTI-SPOOFING EVALUATION")
print("=" * 70)

print(f"Model                  : {MODEL_PATH}")
print(f"Dataset                : {DATASET_PATH}")

print()
print(f"Total images           : {total_images}")
print(f"Successfully evaluated : {successful}")
print(f"Invalid images         : {invalid_images}")


if successful == 0:

    print("\nNo valid predictions were produced.")
    exit()


# ============================================================
# Classification metrics
# ============================================================

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    pos_label=1,
    zero_division=0
)


print("\n" + "-" * 70)
print("Classification Metrics")
print("-" * 70)

print(
    f"Accuracy       : "
    f"{accuracy:.4f} "
    f"({accuracy * 100:.2f}%)"
)

print(
    f"Precision      : "
    f"{precision:.4f} "
    f"({precision * 100:.2f}%)"
)

print(
    f"Recall         : "
    f"{recall:.4f} "
    f"({recall * 100:.2f}%)"
)

print(
    f"F1 Score       : "
    f"{f1:.4f} "
    f"({f1 * 100:.2f}%)"
)


# ============================================================
# ROC-AUC
# ============================================================

if len(set(y_true)) == 2:

    roc_auc = roc_auc_score(
        y_true,
        y_scores
    )

    print(
        f"ROC-AUC        : "
        f"{roc_auc:.4f} "
        f"({roc_auc * 100:.2f}%)"
    )


# ============================================================
# Confusion Matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

print("\n" + "-" * 70)
print("Confusion Matrix")
print("-" * 70)

print("                 Predicted")
print("              REAL    SPOOF")
print()
print(
    f"Actual REAL    "
    f"{cm[0][0]:5d}   "
    f"{cm[0][1]:5d}"
)

print(
    f"Actual SPOOF   "
    f"{cm[1][0]:5d}   "
    f"{cm[1][1]:5d}"
)


# ============================================================
# Class-wise accuracy
# ============================================================

real_total = cm[0][0] + cm[0][1]
spoof_total = cm[1][0] + cm[1][1]

real_accuracy = (
    cm[0][0] / real_total
    if real_total > 0
    else 0
)

spoof_accuracy = (
    cm[1][1] / spoof_total
    if spoof_total > 0
    else 0
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
# Classification Report
# ============================================================

print("\n" + "-" * 70)
print("Classification Report")
print("-" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=["REAL", "SPOOF"],
        zero_division=0
    )
)


print("=" * 70)