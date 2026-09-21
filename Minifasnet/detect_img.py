import os
import cv2
import numpy as np

from src.anti_spoof_predict import AntiSpoofPredict
from src.generate_patches import CropImage
from src.utility import parse_model_name


# ============================================================
# PATHS
# ============================================================

BASE_DIR = r"D:\Face_anti_spoofing"

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

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "test_images",
    "real.png"
)

OUTPUT_PATH = os.path.join(
    BASE_DIR,
    "output_result.jpg"
)

# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

RESULT_DIR = os.path.join(
    BASE_DIR,
    "result"
)

CROPPED_DIR = os.path.join(
    BASE_DIR,
    "cropped_faces"
)

# Create directories if they don't exist
os.makedirs(
    RESULT_DIR,
    exist_ok=True
)

os.makedirs(
    CROPPED_DIR,
    exist_ok=True
)


# ============================================================
# OUTPUT PATHS
# ============================================================

OUTPUT_PATH = os.path.join(
    RESULT_DIR,
    "real_output_result.jpg"
)

CROP_PATH = os.path.join(
    CROPPED_DIR,
    "real_cropped_face.jpg"
)

# ============================================================
# DEVICE
# ============================================================

DEVICE_ID = 0


# ============================================================
# CREATE PREDICTOR
# ============================================================

predictor = AntiSpoofPredict(DEVICE_ID)
cropper = CropImage()


# ============================================================
# LOAD IMAGE
# ============================================================

image = cv2.imread(
    IMAGE_PATH
)

if image is None:
    raise FileNotFoundError(
        f"Could not load image: {IMAGE_PATH}"
    )

# Keep a copy for drawing the result
output_image = image.copy()


# ============================================================
# FACE DETECTION
# ============================================================

bbox = predictor.get_bbox(
    image
)

if bbox is None:

    print("No face detected.")

    # Save image even if no face is detected
    cv2.imwrite(
        OUTPUT_PATH,
        output_image
    )

    print(
        f"Output saved to: {OUTPUT_PATH}"
    )

    raise SystemExit


print()
print("Face bounding box:")
print(bbox)


# ============================================================
# MODEL INFORMATION
# ============================================================

model_name = os.path.basename(
    ANTI_SPOOF_MODEL
)

h_input, w_input, model_type, scale = parse_model_name(
    model_name
)

print()
print("Anti-spoof model:")
print(model_name)

print("Model type :", model_type)
print("Input size :", f"{w_input} x {h_input}")
print("Scale      :", scale)


# ============================================================
# OFFICIAL CROP
# ============================================================

param = {
    "org_img": image,
    "bbox": bbox,
    "scale": scale,
    "out_w": w_input,
    "out_h": h_input,
    "crop": True
}

face_crop = cropper.crop(
    **param
)


# ============================================================
# SAVE CROPPED FACE
# ============================================================

cv2.imwrite(
    CROP_PATH,
    face_crop
)

print()
print(
    "Cropped face saved to:"
)

print(
    CROP_PATH
)


# ============================================================
# MINI FASNET PREDICTION
# ============================================================

prediction = predictor.predict(
    face_crop,
    ANTI_SPOOF_MODEL
)


# ============================================================
# PRINT RAW OUTPUT
# ============================================================

print()
print("Raw model output:")
print(prediction)


# ============================================================
# CLASS PROBABILITIES
# ============================================================

spoof_class_0 = prediction[0][0]
real_class_1 = prediction[0][1]
spoof_class_2 = prediction[0][2]


print()
print("Class probabilities:")

print(
    f"Class 0 : {spoof_class_0:.6f}"
)

print(
    f"Class 1 : {real_class_1:.6f}"
)

print(
    f"Class 2 : {spoof_class_2:.6f}"
)


# ============================================================
# FINAL CLASS
# ============================================================

label = np.argmax(
    prediction
)



# label == 1 -> Real
# otherwise  -> Fake/Spoof

if label == 1:

    result = "REAL"

    box_color = (
        0,
        255,
        0
    )

else:

    result = "SPOOF"

    box_color = (
        0,
        0,
        255
    )


confidence = prediction[0][label]


# ============================================================
# DRAW FACE BOUNDING BOX
# ============================================================

x, y, w, h = bbox

x1 = max(
    0,
    x
)

y1 = max(
    0,
    y
)

x2 = min(
    output_image.shape[1] - 1,
    x + w
)

y2 = min(
    output_image.shape[0] - 1,
    y + h
)


cv2.rectangle(
    output_image,
    (x1, y1),
    (x2, y2),
    box_color,
    3
)


# ============================================================
# DRAW RESULT LABEL
# ============================================================

result_text = (
    f"{result} "
    f"{confidence * 100:.1f}%"
)


cv2.putText(
    output_image,
    result_text,
    (x1, max(35, y1 - 10)),
    cv2.FONT_HERSHEY_SIMPLEX,
    1.0,
    box_color,
    3,
    cv2.LINE_AA
)


# ============================================================
# DRAW CLASS PROBABILITIES
# ============================================================

cv2.putText(
    output_image,
    f"Class 0 (Spoof): {spoof_class_0:.3f}",
    (20, 40),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    output_image,
    f"Class 1 (Real): {real_class_1:.3f}",
    (20, 70),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

cv2.putText(
    output_image,
    f"Class 2 (Spoof): {spoof_class_2:.3f}",
    (20, 100),
    cv2.FONT_HERSHEY_SIMPLEX,
    0.7,
    (255, 255, 255),
    2,
    cv2.LINE_AA
)

# ============================================================
# SAVE FINAL OUTPUT IMAGE
# ============================================================

success = cv2.imwrite(
    OUTPUT_PATH,
    output_image
)

if success:

    print()
    print("=" * 50)
    print(f"Prediction : {result}")
    print(f"Class      : {label}")
    print(f"Confidence : {confidence:.4f}")
    print("=" * 50)

    print()
    print("Final output saved to:")
    print(OUTPUT_PATH)

else:

    print(
        "ERROR: Could not save output image."
    )