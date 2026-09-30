import os
import glob
import numpy as np
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

FP32_MODEL = "models/mn3_tflite/MN3_antispoof_float32.tflite"
INT8_MODEL = "models/MN3_antispoof_int8.tflite"

CALIBRATION_DIR = "calibration_data"


# ============================================================
# Load models
# ============================================================

print("=" * 70)
print("Loading models")
print("=" * 70)

fp32_interpreter = tf.lite.Interpreter(
    model_path=FP32_MODEL
)
fp32_interpreter.allocate_tensors()

int8_interpreter = tf.lite.Interpreter(
    model_path=INT8_MODEL
)
int8_interpreter.allocate_tensors()


fp32_input = fp32_interpreter.get_input_details()[0]
fp32_output = fp32_interpreter.get_output_details()[0]

int8_input = int8_interpreter.get_input_details()[0]
int8_output = int8_interpreter.get_output_details()[0]


print("\nFP32 input:")
print(" shape :", fp32_input["shape"])
print(" dtype :", fp32_input["dtype"])

print("\nFP32 output:")
print(" shape :", fp32_output["shape"])
print(" dtype :", fp32_output["dtype"])

print("\nINT8 input:")
print(" shape :", int8_input["shape"])
print(" dtype :", int8_input["dtype"])
print(" scale:", int8_input["quantization"][0])
print(" zero :", int8_input["quantization"][1])

print("\nINT8 output:")
print(" shape :", int8_output["shape"])
print(" dtype :", int8_output["dtype"])
print(" scale:", int8_output["quantization"][0])
print(" zero :", int8_output["quantization"][1])


# ============================================================
# Get calibration images
# ============================================================

files = []

files.extend(
    glob.glob(
        os.path.join(
            CALIBRATION_DIR,
            "live",
            "*.npy"
        )
    )
)

files.extend(
    glob.glob(
        os.path.join(
            CALIBRATION_DIR,
            "spoof",
            "*.npy"
        )
    )
)

rng = np.random.default_rng(42)
rng.shuffle(files)

# Validate all 1200 samples
print("\nTotal validation samples:", len(files))


# ============================================================
# Quantization helper
# ============================================================

def quantize_input(x, scale, zero_point, dtype):

    q = np.round(
        x / scale + zero_point
    )

    info = np.iinfo(dtype)

    q = np.clip(
        q,
        info.min,
        info.max
    )

    return q.astype(dtype)


# ============================================================
# Dequantization helper
# ============================================================

def dequantize_output(x, scale, zero_point):

    return (
        x.astype(np.float32) - zero_point
    ) * scale


# ============================================================
# Evaluation
# ============================================================

total = 0
prediction_matches = 0

max_difference = 0.0
sum_difference = 0.0
num_values = 0

# Confusion matrix
# Actual LIVE  = class 0
# Actual SPOOF = class 1

tp = 0
tn = 0
fp = 0
fn = 0

results = []


for file_path in files:

    # --------------------------------------------------------
    # Load preprocessed sample
    # --------------------------------------------------------

    x = np.load(file_path).astype(np.float32)

    # --------------------------------------------------------
    # FP32 inference
    # --------------------------------------------------------

    fp32_interpreter.set_tensor(
        fp32_input["index"],
        x
    )

    fp32_interpreter.invoke()

    fp32_logits = fp32_interpreter.get_tensor(
        fp32_output["index"]
    )

    # --------------------------------------------------------
    # Quantize input for INT8 model
    # --------------------------------------------------------

    int8_x = quantize_input(
        x,
        int8_input["quantization"][0],
        int8_input["quantization"][1],
        int8_input["dtype"]
    )

    # --------------------------------------------------------
    # INT8 inference
    # --------------------------------------------------------

    int8_interpreter.set_tensor(
        int8_input["index"],
        int8_x
    )

    int8_interpreter.invoke()

    int8_raw = int8_interpreter.get_tensor(
        int8_output["index"]
    )

    # --------------------------------------------------------
    # Dequantize output
    # --------------------------------------------------------

    int8_logits = dequantize_output(
        int8_raw,
        int8_output["quantization"][0],
        int8_output["quantization"][1]
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    fp32_class = int(
        np.argmax(fp32_logits, axis=1)[0]
    )

    int8_class = int(
        np.argmax(int8_logits, axis=1)[0]
    )

    if fp32_class == int8_class:
        prediction_matches += 1

    # --------------------------------------------------------
    # Numerical difference
    # --------------------------------------------------------

    difference = np.abs(
        fp32_logits - int8_logits
    )

    max_difference = max(
        max_difference,
        float(np.max(difference))
    )

    sum_difference += float(
        np.sum(difference)
    )

    num_values += difference.size

    # --------------------------------------------------------
    # Ground truth from directory
    # --------------------------------------------------------

    if os.path.sep + "live" + os.path.sep in file_path:
        actual_class = 0
    else:
        actual_class = 1

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    if actual_class == 0 and int8_class == 0:
        tn += 1

    elif actual_class == 0 and int8_class == 1:
        fp += 1

    elif actual_class == 1 and int8_class == 1:
        tp += 1

    elif actual_class == 1 and int8_class == 0:
        fn += 1

    total += 1

    results.append(
        (
            file_path,
            actual_class,
            fp32_class,
            int8_class
        )
    )


# ============================================================
# Metrics
# ============================================================

mean_difference = (
    sum_difference / num_values
    if num_values > 0
    else 0
)

accuracy = (
    (tp + tn) / total
    if total > 0
    else 0
)

precision = (
    tp / (tp + fp)
    if (tp + fp) > 0
    else 0
)

recall = (
    tp / (tp + fn)
    if (tp + fn) > 0
    else 0
)

f1 = (
    2 * precision * recall /
    (precision + recall)
    if (precision + recall) > 0
    else 0
)


# ============================================================
# Print results
# ============================================================

print("\n" + "=" * 70)
print("INT8 VALIDATION RESULTS")
print("=" * 70)

print(f"\nTotal samples             : {total}")

print(
    f"FP32 / INT8 prediction "
    f"matches                  : {prediction_matches}"
)

print(
    f"FP32 / INT8 agreement     : "
    f"{prediction_matches / total * 100:.2f}%"
)

print(
    f"\nMaximum logit difference : "
    f"{max_difference:.8f}"
)

print(
    f"Mean logit difference    : "
    f"{mean_difference:.8f}"
)

print("\nClassification metrics:")
print(
    f"Accuracy                 : "
    f"{accuracy:.4f}"
)

print(
    f"Precision                : "
    f"{precision:.4f}"
)

print(
    f"Recall                   : "
    f"{recall:.4f}"
)

print(
    f"F1 Score                 : "
    f"{f1:.4f}"
)

print("\nConfusion Matrix:")
print(
    f"Actual LIVE  -> LIVE: {tn}, SPOOF: {fp}"
)

print(
    f"Actual SPOOF -> LIVE: {fn}, SPOOF: {tp}"
)