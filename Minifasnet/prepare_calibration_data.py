import os
import random
import cv2

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\MinifasnetV1"

RAW_DIR = os.path.join(
    BASE_DIR,
    "calibration_data",
    "raw"
)

FACES_DIR = os.path.join(
    BASE_DIR,
    "calibration_data",
    "faces"
)

FACE_PROTO = os.path.join(
    BASE_DIR,
    "models",
    "deploy.prototxt"
)

FACE_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "Widerface-RetinaFace.caffemodel"
)

ANTI_SPOOF_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "2.7_80x80_MiniFASNetV2.pth"
)

# ============================================================
# SETTINGS
# ============================================================

DEVICE_ID = 0

RANDOM_SEED = 42


# ============================================================
# CREATE OUTPUT DIRECTORY
# ============================================================

os.makedirs(
    FACES_DIR,
    exist_ok=True
)


# ============================================================
# GET MODEL INPUT INFORMATION
# ============================================================

model_name = os.path.basename(
    ANTI_SPOOF_MODEL
)

h_input, w_input, model_type, scale = parse_model_name(
    model_name
)

print("=" * 60)
print("Calibration Dataset Preparation")
print("=" * 60)

print()
print("Model       :", model_name)
print("Model type  :", model_type)
print("Input size  :", f"{w_input} x {h_input}")
print("Crop scale  :", scale)

print()
print("Raw dataset :", RAW_DIR)
print("Output      :", FACES_DIR)


# ============================================================
# LOAD FACE DETECTOR
# ============================================================

print()
print("Loading face detector...")

predictor = AntiSpoofPredict(
    DEVICE_ID
)

cropper = CropImage()

print("Face detector loaded.")


# ============================================================
# COLLECT IMAGE FILES
# ============================================================

valid_extensions = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".webp"
)

image_files = []

for filename in os.listdir(RAW_DIR):

    filepath = os.path.join(
        RAW_DIR,
        filename
    )

    if not os.path.isfile(filepath):
        continue

    if filename.lower().endswith(
        valid_extensions
    ):
        image_files.append(
            filepath
        )


print()
print(
    f"Images found: {len(image_files)}"
)


# ============================================================
# SHUFFLE DATASET
# ============================================================

random.seed(
    RANDOM_SEED
)

random.shuffle(
    image_files
)

print(
    f"Dataset shuffled using seed: {RANDOM_SEED}"
)


# ============================================================
# PROCESS IMAGES
# ============================================================

processed_count = 0
failed_load_count = 0
no_face_count = 0
multiple_face_count = 0


for index, image_path in enumerate(
    image_files,
    start=1
):

    filename = os.path.basename(
        image_path
    )

    print(
        f"[{index}/{len(image_files)}] "
        f"Processing: {filename}"
    )


    # --------------------------------------------------------
    # LOAD IMAGE
    # --------------------------------------------------------

    image = cv2.imread(
        image_path
    )

    if image is None:

        print(
            "  -> Could not read image"
        )

        failed_load_count += 1

        continue


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    bbox = predictor.get_bbox(
        image
    )

    if bbox is None:

        print(
            "  -> No face detected"
        )

        no_face_count += 1

        continue


    # --------------------------------------------------------
    # OFFICIAL CROP
    # --------------------------------------------------------

    face_crop = cropper.crop(
        org_img=image,
        bbox=bbox,
        scale=scale,
        out_w=w_input,
        out_h=h_input,
        crop=True
    )


    # --------------------------------------------------------
    # SAVE FACE CROP
    # --------------------------------------------------------

    output_filename = (
        f"face_{processed_count + 1:04d}.jpg"
    )

    output_path = os.path.join(
        FACES_DIR,
        output_filename
    )

    success = cv2.imwrite(
        output_path,
        face_crop
    )

    if not success:

        print(
            "  -> Failed to save crop"
        )

        continue


    processed_count += 1

    print(
        f"  -> Saved: {output_filename}"
    )


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 60)
print("Processing Complete")
print("=" * 60)

print(
    f"Total images       : {len(image_files)}"
)

print(
    f"Successfully saved : {processed_count}"
)

print(
    f"No face detected   : {no_face_count}"
)

print(
    f"Read failures      : {failed_load_count}"
)

print()
print(
    f"Output directory:"
)

print(
    FACES_DIR
)

print("=" * 60)