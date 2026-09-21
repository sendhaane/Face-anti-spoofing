import os
import time

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


# ============================================================
# SETTING
# ============================================================

CAMERA_INDEX = 0

DEVICE_ID = 0

DETECTOR_CONFIDENCE = 0.6

# Run anti-spoofing every N frames.
# 1 = every frame
# 2 = every second frame
# 3 = every third frame
ANTI_SPOOF_INTERVAL = 1


# ============================================================
# LOAD MODEL
# ============================================================

print("Loading anti-spoofing model...")

predictor = AntiSpoofPredict(DEVICE_ID)

cropper = CropImage()

print("Model loaded.")


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
print("Model information")
print("-----------------")
print("Model       :", model_name)
print("Model type  :", model_type)
print("Input size  :", f"{w_input} x {h_input}")
print("Crop scale  :", scale)
print()


# ============================================================
# OPEN WEBCAM
# ============================================================

cap = cv2.VideoCapture(
    CAMERA_INDEX
)

if not cap.isOpened():

    raise RuntimeError(
        "Could not open webcam."
    )


# Optional camera resolution

cap.set(
    cv2.CAP_PROP_FRAME_WIDTH,
    640
)

cap.set(
    cv2.CAP_PROP_FRAME_HEIGHT,
    480
)


print("Webcam started.")
print("Press Q to quit.")
print()


# ============================================================
# VARIABLES
# ============================================================

frame_count = 0

last_label = "NO FACE"

last_confidence = 0.0

last_probabilities = np.array(
    [0.0, 0.0, 0.0]
)

last_bbox = None

fps = 0.0

previous_time = time.time()


# ============================================================
# MAIN LOOP
# ============================================================

while True:

    ret, frame = cap.read()

    if not ret:

        print("Failed to read frame.")

        break


    frame_count += 1


    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    current_time = time.time()

    elapsed = current_time - previous_time

    if elapsed > 0:

        current_fps = 1.0 / elapsed

        # Smooth FPS
        fps = (
            0.9 * fps
            + 0.1 * current_fps
        )

    previous_time = current_time


    # --------------------------------------------------------
    # FACE DETECTION
    # --------------------------------------------------------

    bbox = predictor.get_bbox(
        frame
    )


    if bbox is None:

        last_bbox = None

        last_label = "NO FACE"

        last_confidence = 0.0

        last_probabilities = np.array(
            [0.0, 0.0, 0.0]
        )

    else:

        last_bbox = bbox


        # ----------------------------------------------------
        # ANTI-SPOOFING
        # ----------------------------------------------------

        if frame_count % ANTI_SPOOF_INTERVAL == 0:

            # Official crop
            #
            # scale = 2.7
            # output = 80 x 80

            face_crop = cropper.crop(
                org_img=frame,
                bbox=bbox,
                scale=scale,
                out_w=w_input,
                out_h=h_input,
                crop=True
            )


            # ------------------------------------------------
            # MINI FASNET PREDICTION
            # ------------------------------------------------

            prediction = predictor.predict(
                face_crop,
                ANTI_SPOOF_MODEL
            )


            # prediction shape:
            #
            # [[class0, class1, class2]]

            last_probabilities = prediction[0]


            # ------------------------------------------------
            # FINAL CLASS
            # ------------------------------------------------

            label = np.argmax(
                last_probabilities
            )

            last_confidence = float(
                last_probabilities[label]
            )


            if label == 1:

                last_label = "REAL"

            else:

                last_label = "SPOOF"


    # ========================================================
    # DRAW FACE BOX
    # ========================================================

    if last_bbox is not None:

        x, y, w, h = last_bbox

        x1 = max(
            0,
            x
        )

        y1 = max(
            0,
            y
        )

        x2 = min(
            frame.shape[1] - 1,
            x + w
        )

        y2 = min(
            frame.shape[0] - 1,
            y + h
        )


        # ----------------------------------------------------
        # BOX STYLE
        # ----------------------------------------------------

        if last_label == "REAL":

            box_color = (
                0,
                255,
                0
            )

        elif last_label == "SPOOF":

            box_color = (
                0,
                0,
                255
            )

        else:

            box_color = (
                255,
                255,
                0
            )


        cv2.rectangle(
            frame,
            (x1, y1),
            (x2, y2),
            box_color,
            2
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label_text = (
            f"{last_label} "
            f"{last_confidence * 100:.1f}%"
        )


        cv2.putText(
            frame,
            label_text,
            (x1, max(30, y1 - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            box_color,
            2,
            cv2.LINE_AA
        )


        # ----------------------------------------------------
        # CLASS PROBABILITIES
        # ----------------------------------------------------

        class0 = last_probabilities[0]

        class1 = last_probabilities[1]

        class2 = last_probabilities[2]


        cv2.putText(
            frame,
            f"C0 Spoof: {class0:.3f}",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"C1 Real : {class1:.3f}",
            (20, 65),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )

        cv2.putText(
            frame,
            f"C2 Spoof: {class2:.3f}",
            (20, 90),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.6,
            (255, 255, 255),
            2
        )


    else:

        cv2.putText(
            frame,
            "NO FACE DETECTED",
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 255),
            2
        )


    # ========================================================
    # FPS DISPLAY
    # ========================================================

    cv2.putText(
        frame,
        f"FPS: {fps:.1f}",
        (20, frame.shape[0] - 20),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 255, 255),
        2
    )


    # ========================================================
    # SHOW
    # ========================================================

    cv2.imshow(
        "Face Anti-Spoofing - MiniFASNetV2",
        frame
    )


    # ========================================================
    # QUIT
    # ========================================================

    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):

        break


# ============================================================
# CLEANUP
# ============================================================

cap.release()

cv2.destroyAllWindows()

print()
print("Webcam stopped.")