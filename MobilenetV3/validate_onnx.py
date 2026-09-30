import os
import cv2
import numpy as np
import torch
import onnxruntime as ort

from src.mobilenetv3 import mobilenetv3_large


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "models/MN3_antispoof.pth"
ONNX_PATH = "models/MN3_antispoof.onnx"

DATASET_DIR = "test_images_organized"

NUM_IMAGES_PER_CLASS = 20

DEVICE = "cpu"


# ============================================================
# PREPROCESSING
# Same preprocessing used by MobileNetV3
# ============================================================

MEAN = np.array(
    [0.5931, 0.4690, 0.4229],
    dtype=np.float32
)

STD = np.array(
    [0.2471, 0.2214, 0.2157],
    dtype=np.float32
)


def preprocess(image):

    # BGR -> RGB
    image = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    # Resize
    image = cv2.resize(
        image,
        (128, 128),
        interpolation=cv2.INTER_CUBIC
    )

    # Convert to [0, 1]
    image = image.astype(
        np.float32
    ) / 255.0

    # Normalize
    image = (
        image - MEAN
    ) / STD

    # HWC -> CHW
    image = np.transpose(
        image,
        (2, 0, 1)
    )

    # Add batch dimension
    image = np.expand_dims(
        image,
        axis=0
    )

    return image.astype(
        np.float32
    )


# ============================================================
# LOAD PYTORCH MODEL
# ============================================================

print("=" * 60)
print("LOADING PYTORCH MODEL")
print("=" * 60)

device = torch.device(DEVICE)

model = mobilenetv3_large(
    width_mult=1.0,
    prob_dropout=0.1,
    type_dropout="bernoulli",
    prob_dropout_linear=0.35,
    embeding_dim=1280,
    mu=0.5,
    sigma=0.3,
    theta=0,
    multi_heads=True
)

checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=True
)

model.load_state_dict(
    checkpoint["state_dict"],
    strict=True
)

model.to(device)
model.eval()

print("PyTorch model loaded.")


# ============================================================
# PYTORCH INFERENCE
# ============================================================

def pytorch_predict(input_numpy):

    input_tensor = torch.from_numpy(
        input_numpy
    ).float().to(device)

    with torch.no_grad():

        features = model(
            input_tensor
        )

        logits = model.make_logits(
            features,
            all=False
        )

        if isinstance(logits, tuple):
            logits = logits[0]

    return logits.cpu().numpy()


# ============================================================
# LOAD ONNX MODEL
# ============================================================

print()
print("=" * 60)
print("LOADING ONNX MODEL")
print("=" * 60)

session = ort.InferenceSession(
    ONNX_PATH,
    providers=[
        "CPUExecutionProvider"
    ]
)

input_name = session.get_inputs()[0].name
output_name = session.get_outputs()[0].name

print(
    f"ONNX input name  : {input_name}"
)

print(
    f"ONNX output name : {output_name}"
)

print(
    f"ONNX input shape : "
    f"{session.get_inputs()[0].shape}"
)

print(
    f"ONNX output shape: "
    f"{session.get_outputs()[0].shape}"
)


# ============================================================
# ONNX INFERENCE
# ============================================================

def onnx_predict(input_numpy):

    output = session.run(
        [output_name],
        {
            input_name: input_numpy
        }
    )

    return output[0]


# ============================================================
# SINGLE RANDOM/SELECTED IMAGE VALIDATION
# ============================================================

def validate_single_image(image_path):

    print()
    print("=" * 60)
    print("SINGLE IMAGE VALIDATION")
    print("=" * 60)

    print(
        f"Image: {image_path}"
    )

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        return

    input_data = preprocess(
        image
    )

    pytorch_output = pytorch_predict(
        input_data
    )

    onnx_output = onnx_predict(
        input_data
    )

    print()
    print("PyTorch logits:")
    print(pytorch_output)

    print()
    print("ONNX logits:")
    print(onnx_output)

    # --------------------------------------------------------
    # Difference
    # --------------------------------------------------------

    absolute_difference = np.abs(
        pytorch_output -
        onnx_output
    )

    max_difference = np.max(
        absolute_difference
    )

    mean_difference = np.mean(
        absolute_difference
    )

    print()
    print("Output difference:")
    print(
        f"Maximum absolute difference : "
        f"{max_difference:.10f}"
    )

    print(
        f"Mean absolute difference    : "
        f"{mean_difference:.10f}"
    )

    # --------------------------------------------------------
    # Predictions
    # --------------------------------------------------------

    pytorch_class = np.argmax(
        pytorch_output,
        axis=1
    )[0]

    onnx_class = np.argmax(
        onnx_output,
        axis=1
    )[0]

    print()
    print(
        f"PyTorch predicted class : "
        f"{pytorch_class}"
    )

    print(
        f"ONNX predicted class    : "
        f"{onnx_class}"
    )

    if pytorch_class == onnx_class:

        print(
            "\n✓ Prediction MATCH"
        )

    else:

        print(
            "\n✗ Prediction MISMATCH"
        )


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_dataset():

    print()
    print("=" * 60)
    print("DATASET VALIDATION")
    print("=" * 60)

    total = 0
    matching = 0
    mismatching = 0

    max_difference = 0.0
    mean_differences = []

    for class_name in [
        "live",
        "spoof"
    ]:

        class_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.exists(
            class_dir
        ):

            print(
                f"Directory not found: "
                f"{class_dir}"
            )

            continue

        image_files = sorted([
            f
            for f in os.listdir(
                class_dir
            )
            if f.lower().endswith(
                (
                    ".jpg",
                    ".jpeg",
                    ".png",
                    ".bmp",
                    ".webp"
                )
            )
        ])

        image_files = image_files[
            :NUM_IMAGES_PER_CLASS
        ]

        print()
        print(
            f"{class_name.upper()}: "
            f"{len(image_files)} images"
        )

        for image_name in image_files:

            image_path = os.path.join(
                class_dir,
                image_name
            )

            image = cv2.imread(
                image_path
            )

            if image is None:

                print(
                    f"Could not read: "
                    f"{image_name}"
                )

                continue

            # ------------------------------------------------
            # Preprocess
            # ------------------------------------------------

            input_data = preprocess(
                image
            )

            # ------------------------------------------------
            # PyTorch
            # ------------------------------------------------

            pytorch_output = (
                pytorch_predict(
                    input_data
                )
            )

            # ------------------------------------------------
            # ONNX
            # ------------------------------------------------

            onnx_output = (
                onnx_predict(
                    input_data
                )
            )

            # ------------------------------------------------
            # Difference
            # ------------------------------------------------

            difference = np.abs(
                pytorch_output -
                onnx_output
            )

            current_max = np.max(
                difference
            )

            current_mean = np.mean(
                difference
            )

            max_difference = max(
                max_difference,
                current_max
            )

            mean_differences.append(
                current_mean
            )

            # ------------------------------------------------
            # Predictions
            # ------------------------------------------------

            pytorch_class = np.argmax(
                pytorch_output,
                axis=1
            )[0]

            onnx_class = np.argmax(
                onnx_output,
                axis=1
            )[0]

            total += 1

            if pytorch_class == onnx_class:

                matching += 1

            else:

                mismatching += 1

                print()
                print(
                    f"MISMATCH: "
                    f"{class_name}/{image_name}"
                )

                print(
                    "PyTorch:",
                    pytorch_output
                )

                print(
                    "ONNX:",
                    onnx_output
                )

    # ========================================================
    # SUMMARY
    # ========================================================

    mean_difference = (
        np.mean(mean_differences)
        if mean_differences
        else 0
    )

    print()
    print("=" * 60)
    print("ONNX VALIDATION SUMMARY")
    print("=" * 60)

    print(
        f"\nImages tested       : {total}"
    )

    print(
        f"Prediction matches  : {matching}"
    )

    print(
        f"Prediction mismatch : {mismatching}"
    )

    if total > 0:

        agreement = (
            matching / total
        ) * 100

        print(
            f"Prediction agreement: "
            f"{agreement:.2f}%"
        )

    print()
    print(
        f"Maximum logit difference : "
        f"{max_difference:.10f}"
    )

    print(
        f"Mean logit difference    : "
        f"{mean_difference:.10f}"
    )

    print()

    if mismatching == 0:

        print(
            "✓ PyTorch and ONNX predictions "
            "are identical on the tested images."
        )

    else:

        print(
            "⚠ Some predictions differ. "
            "Investigate before quantization."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Validate one image first
    # --------------------------------------------------------

    single_image = os.path.join(
        DATASET_DIR,
        "live",
        sorted(
            os.listdir(
                os.path.join(
                    DATASET_DIR,
                    "live"
                )
            )
        )[0]
    )

    validate_single_image(
        single_image
    )

    # --------------------------------------------------------
    # Validate dataset
    # --------------------------------------------------------

    validate_dataset()