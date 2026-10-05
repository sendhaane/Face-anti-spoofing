import os
import cv2
import numpy as np

from ultralytics import YOLO
import onnxruntime as ort


# ============================================================
# Paths
# ============================================================

PT_MODEL = "best.pt"
ONNX_MODEL = "best.onnx"
DATASET_PATH = "test_images"


# ============================================================
# Settings
# ============================================================

CLASS_MAPPING = {
    "real": 0,
    "spoof": 1
}

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)


# ============================================================
# Load models
# ============================================================

print("Loading PyTorch model...")
pt_model = YOLO(PT_MODEL)

print("Loading ONNX model...")
session = ort.InferenceSession(
    ONNX_MODEL,
    providers=["CPUExecutionProvider"]
)

input_info = session.get_inputs()[0]

print("\nONNX input information")
print("-" * 60)
print("Name :", input_info.name)
print("Shape:", input_info.shape)
print("Type :", input_info.type)


# ============================================================
# Preprocessing
# ============================================================

def preprocess(image):
    """
    Match Ultralytics classification preprocessing:
        BGR -> RGB
        Resize to 224x224
        HWC -> CHW
        uint8 -> float32
        /255
    """

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

    image = cv2.resize(
        image,
        (224, 224),
        interpolation=cv2.INTER_LINEAR
    )

    image = image.astype(np.float32) / 255.0

    image = np.transpose(image, (2, 0, 1))

    image = np.expand_dims(image, axis=0)

    return image


# ============================================================
# Evaluation
# ============================================================

total = 0
valid = 0

prediction_mismatches = 0

max_probability_difference = 0.0
sum_probability_difference = 0.0

pt_correct = 0
onnx_correct = 0

mismatch_examples = []


for folder_name, true_class in CLASS_MAPPING.items():

    folder_path = os.path.join(
        DATASET_PATH,
        folder_name
    )

    print("\n" + "=" * 70)
    print(f"Evaluating {folder_name.upper()}")
    print("=" * 70)

    image_files = sorted([
        f for f in os.listdir(folder_path)
        if f.lower().endswith(IMAGE_EXTENSIONS)
    ])

    print("Images:", len(image_files))

    for image_name in image_files:

        image_path = os.path.join(
            folder_path,
            image_name
        )

        total += 1

        image = cv2.imread(image_path)

        if image is None:
            continue

        # ----------------------------------------------------
        # PyTorch prediction
        # ----------------------------------------------------

        pt_result = pt_model(
            image,
            verbose=False
        )[0]

        pt_probs = (
            pt_result.probs.data
            .cpu()
            .numpy()
            .astype(np.float32)
        )

        pt_pred = int(np.argmax(pt_probs))

        # ----------------------------------------------------
        # ONNX prediction
        # ----------------------------------------------------

        input_tensor = preprocess(image)

        onnx_output = session.run(
            None,
            {
                input_info.name: input_tensor
            }
        )[0]

        # ONNX classification output
        # Convert logits to probabilities if necessary

        onnx_output = onnx_output[0]

        exp_output = np.exp(
            onnx_output - np.max(onnx_output)
        )

        onnx_probs = (
            exp_output /
            np.sum(exp_output)
        )

        onnx_pred = int(np.argmax(onnx_probs))

        # ----------------------------------------------------
        # Compare
        # ----------------------------------------------------

        probability_difference = float(
            np.max(
                np.abs(
                    pt_probs - onnx_probs
                )
            )
        )

        max_probability_difference = max(
            max_probability_difference,
            probability_difference
        )

        sum_probability_difference += probability_difference

        if pt_pred != onnx_pred:

            prediction_mismatches += 1

            if len(mismatch_examples) < 10:

                mismatch_examples.append({
                    "image": image_name,
                    "true": true_class,
                    "pt_pred": pt_pred,
                    "onnx_pred": onnx_pred,
                    "pt_probs": pt_probs,
                    "onnx_probs": onnx_probs
                })

        if pt_pred == true_class:
            pt_correct += 1

        if onnx_pred == true_class:
            onnx_correct += 1

        valid += 1


# ============================================================
# Results
# ============================================================

print("\n")
print("=" * 70)
print("ONNX VALIDATION RESULTS")
print("=" * 70)

print(f"Total images                 : {total}")
print(f"Successfully evaluated      : {valid}")

print()
print("PyTorch accuracy             : "
      f"{pt_correct / valid:.4f} "
      f"({pt_correct / valid * 100:.2f}%)")

print("ONNX accuracy                : "
      f"{onnx_correct / valid:.4f} "
      f"({onnx_correct / valid * 100:.2f}%)")

print()
print("Prediction mismatches        : "
      f"{prediction_mismatches}")

print("Prediction agreement        : "
      f"{(valid - prediction_mismatches) / valid * 100:.4f}%")

print()
print("Maximum probability diff     : "
      f"{max_probability_difference:.8f}")

print("Mean probability diff        : "
      f"{sum_probability_difference / valid:.8f}")


# ============================================================
# Mismatch details
# ============================================================

if mismatch_examples:

    print("\n")
    print("=" * 70)
    print("PREDICTION MISMATCH EXAMPLES")
    print("=" * 70)

    for item in mismatch_examples:

        print(f"\nImage: {item['image']}")
        print(f"True class : {item['true']}")

        print(
            "PyTorch    : "
            f"{item['pt_pred']} "
            f"{item['pt_probs']}"
        )

        print(
            "ONNX       : "
            f"{item['onnx_pred']} "
            f"{item['onnx_probs']}"
        )

else:

    print("\nNo prediction mismatches found.")


print("\n" + "=" * 70)
print("VALIDATION COMPLETE")
print("=" * 70)