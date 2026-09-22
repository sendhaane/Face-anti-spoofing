import os

import cv2
import numpy as np
import torch
import torch.nn.functional as F
import onnx
import onnxruntime as ort

from src.model_lib.MiniFASNet import MiniFASNetV2
from src.utility import get_kernel


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\MinifasnetV1"

PTH_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "2.7_80x80_MiniFASNetV2.pth"
)

ONNX_DIR = os.path.join(
    BASE_DIR,
    "onnx"
)

ONNX_MODEL = os.path.join(
    ONNX_DIR,
    "2.7_80x80_MiniFASNetV2.onnx"
)


# ============================================================
# SETTINGS
# ============================================================

DEVICE = torch.device("cpu")

INPUT_HEIGHT = 80
INPUT_WIDTH = 80

INPUT_SHAPE = (
    1,
    3,
    INPUT_HEIGHT,
    INPUT_WIDTH
)


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    ONNX_DIR,
    exist_ok=True
)


# ============================================================
# LOAD MINI FASNET
# ============================================================

print("=" * 60)
print("Loading MiniFASNetV2")
print("=" * 60)

kernel_size = get_kernel(
    INPUT_HEIGHT,
    INPUT_WIDTH
)

print()
print("Input size   :", f"{INPUT_WIDTH} x {INPUT_HEIGHT}")
print("Kernel size  :", kernel_size)
print("Device       :", DEVICE)


model = MiniFASNetV2(
    conv6_kernel=kernel_size
).to(DEVICE)


# ============================================================
# LOAD CHECKPOINT
# ============================================================

print()
print("Loading checkpoint:")
print(PTH_MODEL)

state_dict = torch.load(
    PTH_MODEL,
    map_location=DEVICE
)


# ------------------------------------------------------------
# Handle DataParallel checkpoints
# ------------------------------------------------------------

first_key = next(
    iter(state_dict)
)

if first_key.startswith("module."):

    print("Removing 'module.' prefix...")

    state_dict = {
        key[7:]: value
        for key, value in state_dict.items()
    }


model.load_state_dict(
    state_dict
)


# ============================================================
# EVALUATION MODE
# ============================================================

model.eval()

print("Model loaded successfully.")


# ============================================================
# CREATE EXAMPLE INPUT
# ============================================================

# IMPORTANT:
#
# This is the SAME numerical format used by our
# current working MiniFASNet inference:
#
# BGR
# float32
# 0-255
# NCHW
#
# We intentionally do NOT:
#
# BGR -> RGB
# /255

example_input = torch.rand(
    INPUT_SHAPE,
    dtype=torch.float32
) * 255.0

example_input = example_input.to(
    DEVICE
)


print()
print("Example input shape:")
print(example_input.shape)

print(
    "Example input dtype:",
    example_input.dtype
)

print(
    "Example input range:",
    float(example_input.min()),
    "to",
    float(example_input.max())
)


# ============================================================
# PYTORCH REFERENCE OUTPUT
# ============================================================

print()
print("Running PyTorch reference inference...")

with torch.no_grad():

    pytorch_logits = model(
        example_input
    )

    pytorch_probabilities = F.softmax(
        pytorch_logits,
        dim=1
    )


print()
print("PyTorch logits:")
print(
    pytorch_logits.numpy()
)

print()
print("PyTorch probabilities:")
print(
    pytorch_probabilities.numpy()
)


# ============================================================
# EXPORT TO ONNX
# ============================================================

print()
print("=" * 60)
print("Exporting PyTorch model to ONNX")
print("=" * 60)

onnx_program = torch.onnx.export(
    model,
    (example_input,),
    input_names=["input"],
    output_names=["output"],
    dynamo=True,
    optimize=True,
    verify=False
)

onnx_program.save(
    ONNX_MODEL
)


print()
print("ONNX model saved:")
print(ONNX_MODEL)


# ============================================================
# CHECK ONNX MODEL
# ============================================================

print()
print("=" * 60)
print("Checking ONNX model")
print("=" * 60)

onnx_model = onnx.load(
    ONNX_MODEL
)

onnx.checker.check_model(
    onnx_model
)

print("ONNX model check: PASSED")


# ============================================================
# PRINT ONNX INPUT / OUTPUT
# ============================================================

print()
print("ONNX inputs:")

for input_tensor in onnx_model.graph.input:

    print(
        "  Name :",
        input_tensor.name
    )

    print(
        "  Shape:",
        [
            dimension.dim_value
            for dimension
            in input_tensor.type.tensor_type.shape.dim
        ]
    )

print()
print("ONNX outputs:")

for output_tensor in onnx_model.graph.output:

    print(
        "  Name :",
        output_tensor.name
    )

    print(
        "  Shape:",
        [
            dimension.dim_value
            for dimension
            in output_tensor.type.tensor_type.shape.dim
        ]
    )


# ============================================================
# ONNX RUNTIME INFERENCE
# ============================================================

print()
print("=" * 60)
print("Running ONNX Runtime inference")
print("=" * 60)

session = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CPUExecutionProvider"]
)


# Convert PyTorch tensor to NumPy

onnx_input = (
    example_input
    .cpu()
    .numpy()
)


ort_outputs = session.run(
    ["output"],
    {
        "input": onnx_input
    }
)

onnx_logits = ort_outputs[0]


# ============================================================
# ONNX SOFTMAX
# ============================================================

onnx_probabilities = (
    np.exp(
        onnx_logits
        - np.max(
            onnx_logits,
            axis=1,
            keepdims=True
        )
    )
)

onnx_probabilities = (
    onnx_probabilities
    /
    np.sum(
        onnx_probabilities,
        axis=1,
        keepdims=True
    )
)


# ============================================================
# PRINT RESULTS
# ============================================================

print()
print("ONNX logits:")
print(
    onnx_logits
)

print()
print("ONNX probabilities:")
print(
    onnx_probabilities
)


# ============================================================
# COMPARE PYTORCH vs ONNX
# ============================================================

pytorch_logits_np = (
    pytorch_logits
    .cpu()
    .numpy()
)

pytorch_probabilities_np = (
    pytorch_probabilities
    .cpu()
    .numpy()
)


logit_difference = np.max(
    np.abs(
        pytorch_logits_np
        -
        onnx_logits
    )
)

probability_difference = np.max(
    np.abs(
        pytorch_probabilities_np
        -
        onnx_probabilities
    )
)


print()
print("=" * 60)
print("PyTorch vs ONNX comparison")
print("=" * 60)

print()
print(
    "Maximum logit difference       :",
    logit_difference
)

print(
    "Maximum probability difference :",
    probability_difference
)


# ============================================================
# FINAL STATUS
# ============================================================

print()

if probability_difference < 1e-4:

    print(
        "VALIDATION: PASSED"
    )

    print(
        "PyTorch and ONNX outputs match closely."
    )

else:

    print(
        "VALIDATION: CHECK REQUIRED"
    )

    print(
        "PyTorch and ONNX outputs differ."
    )

print()
print("=" * 60)
print("ONNX export complete")
print("=" * 60)