import os
import numpy as np
import torch
import tensorflow as tf

from src.model_lib.MiniFASNet import MiniFASNetV2


# --------------------------------------------------
# Paths
# --------------------------------------------------
BASE_DIR = r"D:\MinifasnetV1"

PYTORCH_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "2.7_80x80_MiniFASNetV2.pth"
)

TFLITE_MODEL = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_float32.tflite"
)


# --------------------------------------------------
# 1. Load PyTorch model
# --------------------------------------------------
device = torch.device("cpu")

model = MiniFASNetV2(conv6_kernel=(5, 5)).to(device)

state_dict = torch.load(
    PYTORCH_MODEL,
    map_location=device
)

# Remove DataParallel prefix if present
if list(state_dict.keys())[0].startswith("module."):
    state_dict = {
        k.replace("module.", "", 1): v
        for k, v in state_dict.items()
    }

model.load_state_dict(state_dict)
model.eval()


# --------------------------------------------------
# 2. Create identical test input
# --------------------------------------------------
np.random.seed(42)

input_data = (
    np.random.rand(1, 3, 80, 80).astype(np.float32) * 255.0
)


# --------------------------------------------------
# 3. PyTorch inference
# --------------------------------------------------
torch_input = torch.from_numpy(input_data)

with torch.no_grad():
    pytorch_output = model(torch_input).numpy()


# --------------------------------------------------
# 4. Load TFLite model
# --------------------------------------------------
interpreter = tf.lite.Interpreter(
    model_path=TFLITE_MODEL
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()


print("\n========== TFLite Model Information ==========")

print("Input:")
print("  Shape :", input_details[0]["shape"])
print("  Dtype :", input_details[0]["dtype"])

print("\nOutput:")
print("  Shape :", output_details[0]["shape"])
print("  Dtype :", output_details[0]["dtype"])


# --------------------------------------------------
# 5. TFLite inference
# --------------------------------------------------

# PyTorch/ONNX: NCHW [1, 3, 80, 80]
# TFLite:      NHWC [1, 80, 80, 3]

tflite_input = np.transpose(
    input_data,
    (0, 2, 3, 1)
)

print("\nTFLite input after transpose:")
print("  Shape :", tflite_input.shape)

interpreter.set_tensor(
    input_details[0]["index"],
    tflite_input
)
tflite_input = np.transpose(
    input_data,
    (0, 2, 3, 1)
)

print("\nTFLite input after transpose:")
print("  Shape :", tflite_input.shape)

interpreter.set_tensor(
    input_details[0]["index"],
    tflite_input
)

interpreter.invoke()

tflite_output = interpreter.get_tensor(
    output_details[0]["index"]
)


# --------------------------------------------------
# 6. Compare outputs
# --------------------------------------------------
print("\n========== Output Comparison ==========")

print("PyTorch output:")
print(pytorch_output)

print("\nTFLite output:")
print(tflite_output)

max_diff = np.max(
    np.abs(pytorch_output - tflite_output)
)

mean_diff = np.mean(
    np.abs(pytorch_output - tflite_output)
)

print("\nMaximum absolute difference :", max_diff)
print("Mean absolute difference    :", mean_diff)


# --------------------------------------------------
# 7. Compare predicted classes
# --------------------------------------------------
pytorch_class = np.argmax(pytorch_output, axis=1)[0]
tflite_class = np.argmax(tflite_output, axis=1)[0]

print("\n========== Prediction ==========")

print("PyTorch predicted class :", pytorch_class)
print("TFLite predicted class  :", tflite_class)

if pytorch_class == tflite_class:
    print("Prediction match: PASS")
else:
    print("Prediction match: FAIL")


# --------------------------------------------------
# 8. Softmax probabilities
# --------------------------------------------------
def softmax(x):
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


pytorch_prob = softmax(pytorch_output)
tflite_prob = softmax(tflite_output)

print("\n========== Probabilities ==========")

print("PyTorch:")
print(pytorch_prob)

print("\nTFLite:")
print(tflite_prob)