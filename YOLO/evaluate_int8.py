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

MODEL_PATH = "tflite/best_int8.tflite"
DATASET_PATH = "test_images"

IMAGE_SIZE = 224

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
# Load INT8 TFLite model
# ============================================================

print("=" * 70)
print("YOLOv8 INT8 TFLITE ANTI-SPOOFING EVALUATION")
print("=" * 70)

print("\nLoading model...")

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_info = input_details[0]
output_info = output_details[0]

input_index = input_info["index"]
output_index = output_info["index"]

input_shape = input_info["shape"]
input_dtype = input_info["dtype"]

output_shape = output_info["shape"]
output_dtype = output_info["dtype"]

input_scale, input_zero_point = input_info["quantization"]
output_scale, output_zero_point = output_info["quantization"]


# ============================================================
# Print model information
# ============================================================

print("\nModel Information")
print("-" * 70)

print("Model          :", MODEL_PATH)
print("Input shape    :", input_shape)
print("Input dtype    :", input_dtype)
print("Input scale    :", input_scale)
print("Input zero pt  :", input_zero_point)

print("Output shape   :", output_shape)
print("Output dtype   :", output_dtype)
print("Output scale   :", output_scale)
print("Output zero pt :", output_zero_point)


# ============================================================
# Collect images
# ============================================================

image_paths = []

for class_name, class_id in CLASS_MAPPING.items():

    folder_path = os.path.join(
        DATASET_PATH,
        class_name
    )

    files = sorted(os.listdir(folder_path))

    for filename in files:

        if filename.lower().endswith(
            SUPPORTED_EXTENSIONS
        ):

            image_paths.append(
                (
                    os.path.join(
                        folder_path,
                        filename
                    ),
                    class_id
                )
            )


print("\nDataset Information")
print("-" * 70)

print("Dataset        :", DATASET_PATH)
print("Total images   :", len(image_paths))
print("REAL images    :", sum(
    1 for _, label in image_paths if label == 0
))
print("SPOOF images   :", sum(
    1 for _, label in image_paths if label == 1
))


# ============================================================
# Preprocessing
# ============================================================

def preprocess_image(image):

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

    # Normalize exactly like FP32 model
    image = image.astype(
        np.float32
    ) / 255.0

    # --------------------------------------------------------
    # TFLite model was converted with -k images
    # so input is expected to be NCHW:
    #
    # [1, 3, 224, 224]
    # --------------------------------------------------------

    if len(input_shape) == 4 and input_shape[1] == 3:

        image = np.transpose(
            image,
            (2, 0, 1)
        )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image


# ============================================================
# Quantize input
# ============================================================

def quantize_input(image):

    if input_dtype == np.int8:

        image = (
            image / input_scale
        ) + input_zero_point

        image = np.round(image)

        image = np.clip(
            image,
            -128,
            127
        )

        return image.astype(np.int8)

    elif input_dtype == np.uint8:

        image = (
            image / input_scale
        ) + input_zero_point

        image = np.round(image)

        image = np.clip(
            image,
            0,
            255
        )

        return image.astype(np.uint8)

    elif input_dtype == np.float32:

        return image.astype(np.float32)

    else:

        raise ValueError(
            f"Unsupported input dtype: {input_dtype}"
        )


# ============================================================
# Dequantize output
# ============================================================

def dequantize_output(output):

    if output_dtype in [
        np.int8,
        np.uint8
    ]:

        output = (
            output.astype(np.float32)
            - output_zero_point
        ) * output_scale

    else:

        output = output.astype(
            np.float32
        )

    return output


# ============================================================
# Run inference
# ============================================================

y_true = []
y_pred = []
y_scores = []

invalid_images = 0


print("\nRunning inference...\n")

for count, (image_path, true_class) in enumerate(
    image_paths,
    start=1
):

    image = cv2.imread(image_path)

    if image is None:

        print(
            "Could not read:",
            image_path
        )

        invalid_images += 1
        continue

    # Preprocess
    image = preprocess_image(image)

    # Quantize
    input_data = quantize_input(image)

    # Set input
    interpreter.set_tensor(
        input_index,
        input_data
    )

    # Inference
    interpreter.invoke()

    # Get output
    output = interpreter.get_tensor(
        output_index
    )

    output = output[0]

    # Dequantize
    output = dequantize_output(output)

    # --------------------------------------------------------
    # Convert output to probabilities
    # --------------------------------------------------------

    # Stable softmax
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

    if count % 50 == 0:

        print(
            f"Processed {count}/{len(image_paths)} images"
        )


# ============================================================
# Convert to NumPy
# ============================================================

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_scores = np.array(y_scores)


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

roc_auc = roc_auc_score(
    y_true,
    y_scores
)


# ============================================================
# Confusion matrix
# ============================================================

cm = confusion_matrix(
    y_true,
    y_pred,
    labels=[0, 1]
)

TN = cm[0, 0]
FP = cm[0, 1]
FN = cm[1, 0]
TP = cm[1, 1]


# ============================================================
# Anti-spoofing metrics
# ============================================================

# APCER:
# Attack (SPOOF) classified as Bona Fide (REAL)

APCER = (
    FN /
    (FN + TP)
    if (FN + TP) > 0
    else 0
)


# BPCER:
# Bona Fide (REAL) classified as Attack (SPOOF)

BPCER = (
    FP /
    (TN + FP)
    if (TN + FP) > 0
    else 0
)


# ACER

ACER = (
    APCER + BPCER
) / 2


# ============================================================
# Class accuracy
# ============================================================

real_accuracy = (
    TN /
    (TN + FP)
    if (TN + FP) > 0
    else 0
)

spoof_accuracy = (
    TP /
    (TP + FN)
    if (TP + FN) > 0
    else 0
)


# ============================================================
# Print results
# ============================================================

print("\n\n")
print("=" * 70)
print("INT8 TFLITE EVALUATION RESULTS")
print("=" * 70)

print("\nDataset")
print("-" * 70)

print("Total images           :", len(image_paths))
print("Successfully evaluated :", len(y_true))
print("Invalid images         :", invalid_images)


print("\nClassification Metrics")
print("-" * 70)

print(
    f"Accuracy       : "
    f"{accuracy:.4f} ({accuracy * 100:.2f}%)"
)

print(
    f"Precision      : "
    f"{precision:.4f} ({precision * 100:.2f}%)"
)

print(
    f"Recall         : "
    f"{recall:.4f} ({recall * 100:.2f}%)"
)

print(
    f"F1 Score       : "
    f"{f1:.4f} ({f1 * 100:.2f}%)"
)

print(
    f"ROC-AUC        : "
    f"{roc_auc:.4f} ({roc_auc * 100:.2f}%)"
)


# ============================================================
# Confusion matrix
# ============================================================

print("\nConfusion Matrix")
print("-" * 70)

print(
    "                 Predicted"
)

print(
    "              REAL    SPOOF"
)

print(
    f"Actual REAL     "
    f"{TN:4d}    {FP:4d}"
)

print(
    f"Actual SPOOF    "
    f"{FN:4d}    {TP:4d}"
)


# ============================================================
# Class accuracy
# ============================================================

print("\nClass Accuracy")
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
# Anti-spoofing metrics
# ============================================================

print("\nAnti-Spoofing Metrics")
print("-" * 70)

print(
    f"APCER : "
    f"{APCER:.4f} ({APCER * 100:.2f}%)"
)

print(
    f"BPCER : "
    f"{BPCER:.4f} ({BPCER * 100:.2f}%)"
)

print(
    f"ACER  : "
    f"{ACER:.4f} ({ACER * 100:.2f}%)"
)


# ============================================================
# Classification report
# ============================================================

print("\nClassification Report")
print("-" * 70)

print(
    classification_report(
        y_true,
        y_pred,
        labels=[0, 1],
        target_names=[
            "REAL",
            "SPOOF"
        ],
        digits=4,
        zero_division=0
    )
)


print("=" * 70)
print("Evaluation complete.")
print("=" * 70)