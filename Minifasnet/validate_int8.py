import os
import numpy as np
import tensorflow as tf


# ============================================================
# Paths
# ============================================================

BASE_DIR = r"D:\MinifasnetV1"

FP32_MODEL = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_float32.tflite"
)

INT8_MODEL = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_int8.tflite"
)


# ============================================================
# Create identical test input
# ============================================================

np.random.seed(42)

# Original PyTorch/ONNX format:
# [batch, channels, height, width]
input_nchw = (
    np.random.rand(1, 3, 80, 80).astype(np.float32)
    * 255.0
)

# TFLite format:
# [batch, height, width, channels]
input_nhwc = np.transpose(
    input_nchw,
    (0, 2, 3, 1)
)


# ============================================================
# FP32 TFLite
# ============================================================

print("\n==========================================")
print("FP32 TFLITE")
print("==========================================")

fp32_interpreter = tf.lite.Interpreter(
    model_path=FP32_MODEL
)

fp32_interpreter.allocate_tensors()

fp32_input = fp32_interpreter.get_input_details()[0]
fp32_output = fp32_interpreter.get_output_details()[0]

print("Input shape :", fp32_input["shape"])
print("Input dtype :", fp32_input["dtype"])

fp32_interpreter.set_tensor(
    fp32_input["index"],
    input_nhwc.astype(np.float32)
)

fp32_interpreter.invoke()

fp32_result = fp32_interpreter.get_tensor(
    fp32_output["index"]
)


# ============================================================
# INT8 TFLite
# ============================================================

print("\n==========================================")
print("INT8 TFLITE")
print("==========================================")

int8_interpreter = tf.lite.Interpreter(
    model_path=INT8_MODEL
)

int8_interpreter.allocate_tensors()

int8_input = int8_interpreter.get_input_details()[0]
int8_output = int8_interpreter.get_output_details()[0]

print("Input shape :", int8_input["shape"])
print("Input dtype :", int8_input["dtype"])

print("Input quantization:")
print("  Scale      :", int8_input["quantization"][0])
print("  Zero point :", int8_input["quantization"][1])

print("\nOutput quantization:")
print("  Scale      :", int8_output["quantization"][0])
print("  Zero point :", int8_output["quantization"][1])


# ============================================================
# Quantize input
# ============================================================

input_scale = int8_input["quantization"][0]
input_zero_point = int8_input["quantization"][1]

int8_input_data = np.round(
    input_nhwc / input_scale
    + input_zero_point
).astype(np.int8)


# ============================================================
# INT8 inference
# ============================================================

int8_interpreter.set_tensor(
    int8_input["index"],
    int8_input_data
)

int8_interpreter.invoke()

int8_raw_output = int8_interpreter.get_tensor(
    int8_output["index"]
)


# ============================================================
# Dequantize output
# ============================================================

output_scale = int8_output["quantization"][0]
output_zero_point = int8_output["quantization"][1]

int8_dequantized = (
    output_scale
    * (
        int8_raw_output.astype(np.float32)
        - output_zero_point
    )
)


# ============================================================
# Compare outputs
# ============================================================

print("\n==========================================")
print("OUTPUT COMPARISON")
print("==========================================")

print("\nFP32 output:")
print(fp32_result)

print("\nINT8 raw output:")
print(int8_raw_output)

print("\nINT8 dequantized output:")
print(int8_dequantized)


# Absolute difference
absolute_difference = np.abs(
    fp32_result - int8_dequantized
)

max_difference = np.max(
    absolute_difference
)

mean_difference = np.mean(
    absolute_difference
)


print("\nMaximum absolute difference:")
print(max_difference)

print("\nMean absolute difference:")
print(mean_difference)


# ============================================================
# Prediction comparison
# ============================================================

fp32_class = np.argmax(
    fp32_result,
    axis=1
)[0]

int8_class = np.argmax(
    int8_dequantized,
    axis=1
)[0]


print("\n==========================================")
print("PREDICTION")
print("==========================================")

print("FP32 predicted class :", fp32_class)
print("INT8 predicted class :", int8_class)

if fp32_class == int8_class:
    print("\nPrediction match: PASS")
else:
    print("\nPrediction match: FAIL")


# ============================================================
# Softmax
# ============================================================

def softmax(x):

    x = x - np.max(
        x,
        axis=1,
        keepdims=True
    )

    exp_x = np.exp(x)

    return exp_x / np.sum(
        exp_x,
        axis=1,
        keepdims=True
    )


fp32_probability = softmax(
    fp32_result
)

int8_probability = softmax(
    int8_dequantized
)


print("\n==========================================")
print("PROBABILITIES")
print("==========================================")

print("\nFP32:")
print(fp32_probability)

print("\nINT8:")
print(int8_probability)