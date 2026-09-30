import cv2
import numpy as np
import onnxruntime as ort
import tensorflow as tf
import os

# ============================================================
# Paths
# ============================================================

ONNX_MODEL = "models/MN3_antispoof.onnx"
TFLITE_MODEL = "models/mn3_tflite/MN3_antispoof_float32.tflite"

IMAGE_PATH = "test_images_organized/live/494756.png"

# ============================================================
# Preprocessing
# Same preprocessing used for PyTorch/ONNX validation
# ============================================================

MEAN = np.array([0.5931, 0.4690, 0.4229], dtype=np.float32)
STD = np.array([0.2471, 0.2214, 0.2157], dtype=np.float32)


def preprocess(image_path):
    image = cv2.imread(image_path)

    if image is None:
        raise ValueError(f"Could not read image: {image_path}")

    # BGR -> RGB
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    # Resize
    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_CUBIC
    )

    # Normalize to [0,1]
    image = image.astype(np.float32) / 255.0

    # Mean / std normalization
    image = (image - MEAN) / STD

    # HWC -> CHW for ONNX
    chw = np.transpose(image, (2, 0, 1))

    # ONNX input
    onnx_input = np.expand_dims(chw, axis=0).astype(np.float32)

    # TFLite input expects NHWC
    tflite_input = np.expand_dims(image, axis=0).astype(np.float32)

    return onnx_input, tflite_input


# ============================================================
# Load ONNX
# ============================================================

print("=" * 60)
print("Loading ONNX model")
print("=" * 60)

onnx_session = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CPUExecutionProvider"]
)

onnx_input_name = onnx_session.get_inputs()[0].name
onnx_output_name = onnx_session.get_outputs()[0].name

print("ONNX input :", onnx_session.get_inputs()[0].shape)
print("ONNX output:", onnx_session.get_outputs()[0].shape)


# ============================================================
# Load TFLite
# ============================================================

print("\n" + "=" * 60)
print("Loading TFLite model")
print("=" * 60)

interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite input :", input_details[0]["shape"])
print("TFLite dtype :", input_details[0]["dtype"])
print("TFLite output:", output_details[0]["shape"])
print("TFLite dtype :", output_details[0]["dtype"])


# ============================================================
# Preprocess
# ============================================================

onnx_input, tflite_input = preprocess(IMAGE_PATH)


# ============================================================
# ONNX inference
# ============================================================

onnx_output = onnx_session.run(
    [onnx_output_name],
    {onnx_input_name: onnx_input}
)[0]


# ============================================================
# TFLite inference
# ============================================================

interpreter.set_tensor(
    input_details[0]["index"],
    tflite_input
)

interpreter.invoke()

tflite_output = interpreter.get_tensor(
    output_details[0]["index"]
)


# ============================================================
# Compare
# ============================================================

print("\n" + "=" * 60)
print("RESULT")
print("=" * 60)

print("\nONNX logits:")
print(onnx_output)

print("\nTFLite logits:")
print(tflite_output)

max_diff = np.max(
    np.abs(onnx_output - tflite_output)
)

mean_diff = np.mean(
    np.abs(onnx_output - tflite_output)
)

onnx_class = np.argmax(onnx_output, axis=1)[0]
tflite_class = np.argmax(tflite_output, axis=1)[0]

print("\nMaximum absolute difference:", max_diff)
print("Mean absolute difference   :", mean_diff)

print("\nONNX class   :", onnx_class)
print("TFLite class :", tflite_class)

if onnx_class == tflite_class:
    print("\nPrediction MATCH")
else:
    print("\nPrediction MISMATCH")