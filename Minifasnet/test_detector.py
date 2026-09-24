import os
import cv2
import shutil
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = r"D:\Face_anti_spoofing\Minifasnet\models\scrfd_int8.tflite"

INPUT_FOLDER = r"D:\Face_anti_spoofing\Minifasnet\test_images"

OUTPUT_FOLDER = r"D:\Face_anti_spoofing\Minifasnet\scrfd_detection_results"

DETECTED_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "detected"
)

NO_FACE_FOLDER = os.path.join(
    OUTPUT_FOLDER,
    "no_face_detected"
)

CONF_THRESHOLD = 0.30
NMS_THRESHOLD = 0.40

STRIDES = [8, 16, 32]

# SCRFD normally uses 2 anchors per location
NUM_ANCHORS = 2


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

os.makedirs(DETECTED_FOLDER, exist_ok=True)
os.makedirs(NO_FACE_FOLDER, exist_ok=True)


# ============================================================
# LOAD TFLITE MODEL
# ============================================================

print("=" * 60)
print("LOADING SCRFD INT8 MODEL")
print("=" * 60)

interpreter = tf.lite.Interpreter(
    model_path=MODEL_PATH
)

interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

input_info = input_details[0]

input_h = input_info["shape"][1]
input_w = input_info["shape"][2]

print("Input shape :", input_info["shape"])
print("Input dtype :", input_info["dtype"])
print("Input quantization :", input_info["quantization"])

print("\nOutputs:")

for i, detail in enumerate(output_details):
    print(
        i,
        detail["name"],
        detail["shape"],
        detail["dtype"],
        "quantization:",
        detail["quantization"]
    )


# ============================================================
# INPUT PREPROCESSING
# ============================================================

def preprocess_bgr(
    image,
    input_h,
    input_w,
    input_detail
):
    """
    SCRFD preprocessing:

    BGR
      ↓
    Resize while preserving aspect ratio
      ↓
    Top-left padding
      ↓
    BGR -> RGB
      ↓
    (pixel - 127.5) / 128
      ↓
    INT8 quantization
    """

    im_h, im_w = image.shape[:2]

    im_ratio = im_h / im_w
    model_ratio = input_h / input_w

    if im_ratio > model_ratio:
        new_h = input_h
        new_w = int(new_h / im_ratio)
    else:
        new_w = input_w
        new_h = int(new_w * im_ratio)

    det_scale = new_h / im_h

    resized = cv2.resize(
        image,
        (new_w, new_h)
    )

    canvas = np.zeros(
        (input_h, input_w, 3),
        dtype=np.uint8
    )

    canvas[:new_h, :new_w] = resized

    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    ).astype(np.float32)

    normalized = (
        rgb - 127.5
    ) / 128.0

    batch = normalized[None, ...]

    # --------------------------------------------------------
    # Quantize for INT8 TFLite
    # --------------------------------------------------------

    scale, zero_point = input_detail["quantization"]

    if (
        scale != 0
        and input_detail["dtype"] in (np.int8, np.uint8)
    ):

        batch = np.round(
            batch / scale + zero_point
        )

        info = np.iinfo(
            input_detail["dtype"]
        )

        batch = np.clip(
            batch,
            info.min,
            info.max
        ).astype(
            input_detail["dtype"]
        )

    return batch, det_scale


# ============================================================
# DEQUANTIZE OUTPUT
# ============================================================

def dequantize_output(
    output,
    detail
):

    scale, zero_point = detail["quantization"]

    if (
        scale != 0
        and np.issubdtype(output.dtype, np.integer)
    ):
        output = (
            output.astype(np.float32) - zero_point
        ) * scale

    else:
        output = output.astype(np.float32)

    return output


# ============================================================
# FIND SCRFD OUTPUT GROUPS
# ============================================================

def organize_outputs(outputs):
    """
    SCRFD has 3 detection levels.

    Each level contains:

        bbox    -> (..., 4)
        score   -> (..., 1)
        landmark-> (..., 10)

    The output ordering may vary, so we identify them
    using their tensor sizes.
    """

    groups = []

    used = set()

    for i in range(len(outputs)):

        if i in used:
            continue

        shape = outputs[i].shape

        # Remove batch dimension if present
        arr = outputs[i]

        if arr.ndim == 3:
            arr = arr[0]

        if arr.ndim != 2:
            continue

        if arr.shape[1] != 4:
            continue

        n = arr.shape[0]

        # Find score tensor
        score_index = None
        landmark_index = None

        for j in range(len(outputs)):

            if j in used or j == i:
                continue

            other = outputs[j]

            if other.ndim == 3:
                other = other[0]

            if other.ndim != 2:
                continue

            if other.shape[0] != n:
                continue

            if other.shape[1] == 1:
                score_index = j

            elif other.shape[1] == 10:
                landmark_index = j

        if score_index is not None:

            groups.append(
                (
                    i,
                    score_index,
                    landmark_index,
                    n
                )
            )

            used.add(i)
            used.add(score_index)

            if landmark_index is not None:
                used.add(landmark_index)

    return groups


# ============================================================
# SCRFD BOX DECODER
# ============================================================

def decode_boxes(
    box_output,
    score_output,
    stride,
    det_scale
):

    boxes = box_output
    scores = score_output

    if boxes.ndim == 3:
        boxes = boxes[0]

    if scores.ndim == 3:
        scores = scores[0]

    scores = scores.reshape(-1)

    # --------------------------------------------------------
    # Number of locations
    # --------------------------------------------------------

    feature_h = input_h // stride
    feature_w = input_w // stride

    expected = (
        feature_h *
        feature_w *
        NUM_ANCHORS
    )

    if len(boxes) != expected:

        print(
            f"Warning: stride {stride}: "
            f"expected {expected}, "
            f"got {len(boxes)}"
        )

        return [], []

    # --------------------------------------------------------
    # Generate anchor centers
    # --------------------------------------------------------

    centers = []

    for y in range(feature_h):

        for x in range(feature_w):

            cx = (
                x + 0.5
            ) * stride

            cy = (
                y + 0.5
            ) * stride

            for _ in range(NUM_ANCHORS):

                centers.append(
                    (cx, cy)
                )

    centers = np.asarray(
        centers,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Decode distances
    #
    # SCRFD:
    #
    # left
    # top
    # right
    # bottom
    #
    # multiplied by stride
    # --------------------------------------------------------

    distances = boxes.astype(
        np.float32
    ) * stride

    x1 = (
        centers[:, 0]
        - distances[:, 0]
    )

    y1 = (
        centers[:, 1]
        - distances[:, 1]
    )

    x2 = (
        centers[:, 0]
        + distances[:, 2]
    )

    y2 = (
        centers[:, 1]
        + distances[:, 3]
    )

    decoded = np.stack(
        [x1, y1, x2, y2],
        axis=1
    )

    # --------------------------------------------------------
    # Convert from detector coordinates
    # back to original image coordinates
    # --------------------------------------------------------

    decoded /= det_scale

    return decoded, scores


# ============================================================
# NMS
# ============================================================

def nms(
    boxes,
    scores,
    threshold
):

    if len(boxes) == 0:
        return []

    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (
        x2 - x1
    ) * (
        y2 - y1
    )

    order = scores.argsort()[::-1]

    keep = []

    while len(order) > 0:

        i = order[0]

        keep.append(i)

        xx1 = np.maximum(
            x1[i],
            x1[order[1:]]
        )

        yy1 = np.maximum(
            y1[i],
            y1[order[1:]]
        )

        xx2 = np.minimum(
            x2[i],
            x2[order[1:]]
        )

        yy2 = np.minimum(
            y2[i],
            y2[order[1:]]
        )

        w = np.maximum(
            0,
            xx2 - xx1
        )

        h = np.maximum(
            0,
            yy2 - yy1
        )

        intersection = w * h

        union = (
            areas[i]
            + areas[order[1:]]
            - intersection
        )

        iou = intersection / np.maximum(
            union,
            1e-6
        )

        remaining = np.where(
            iou <= threshold
        )[0]

        order = order[
            remaining + 1
        ]

    return keep


# ============================================================
# DETECT FACES
# ============================================================

def detect_faces(
    image
):

    batch, det_scale = preprocess_bgr(
        image,
        input_h,
        input_w,
        input_info
    )

    interpreter.set_tensor(
        input_info["index"],
        batch
    )

    interpreter.invoke()

    # --------------------------------------------------------
    # Read outputs
    # --------------------------------------------------------

    outputs = []

    for detail in output_details:

        output = interpreter.get_tensor(
            detail["index"]
        )

        output = dequantize_output(
            output,
            detail
        )

        outputs.append(
            output
        )

    # --------------------------------------------------------
    # Organize detection heads
    # --------------------------------------------------------

    groups = organize_outputs(
        outputs
    )

    if len(groups) == 0:

        print(
            "ERROR: Could not identify SCRFD output heads."
        )

        return []

    all_boxes = []
    all_scores = []

    # --------------------------------------------------------
    # Process each stride
    # --------------------------------------------------------

    for (
        box_index,
        score_index,
        landmark_index,
        n
    ) in groups:

        # Determine stride from number of anchors
        possible_stride = None

        for stride in STRIDES:

            feature_h = input_h // stride
            feature_w = input_w // stride

            expected = (
                feature_h *
                feature_w *
                NUM_ANCHORS
            )

            if expected == n:

                possible_stride = stride
                break

        if possible_stride is None:
            continue

        boxes = outputs[
            box_index
        ]

        scores = outputs[
            score_index
        ]

        decoded_boxes, decoded_scores = decode_boxes(
            boxes,
            scores,
            possible_stride,
            det_scale
        )

        if len(decoded_boxes) == 0:
            continue

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        mask = (
            decoded_scores >= CONF_THRESHOLD
        )

        decoded_boxes = decoded_boxes[
            mask
        ]

        decoded_scores = decoded_scores[
            mask
        ]

        all_boxes.extend(
            decoded_boxes
        )

        all_scores.extend(
            decoded_scores
        )

    if len(all_boxes) == 0:

        return []

    all_boxes = np.asarray(
        all_boxes,
        dtype=np.float32
    )

    all_scores = np.asarray(
        all_scores,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # NMS
    # --------------------------------------------------------

    keep = nms(
        all_boxes,
        all_scores,
        NMS_THRESHOLD
    )

    detections = []

    h, w = image.shape[:2]

    for i in keep:

        x1, y1, x2, y2 = all_boxes[i]

        x1 = max(
            0,
            min(w - 1, int(x1))
        )

        y1 = max(
            0,
            min(h - 1, int(y1))
        )

        x2 = max(
            0,
            min(w - 1, int(x2))
        )

        y2 = max(
            0,
            min(h - 1, int(y2))
        )

        if x2 <= x1 or y2 <= y1:
            continue

        detections.append(
            (
                x1,
                y1,
                x2,
                y2,
                float(all_scores[i])
            )
        )

    return detections


# ============================================================
# PROCESS DATASET
# ============================================================

print("\n")
print("=" * 60)
print("STARTING SCRFD FACE DETECTION")
print("=" * 60)

total = 0
processed = 0
detected_count = 0
no_face_count = 0
error_count = 0


for root, dirs, files in os.walk(
    INPUT_FOLDER
):

    for filename in files:

        if not filename.lower().endswith(
            (
                ".jpg",
                ".jpeg",
                ".png",
                ".bmp",
                ".webp"
            )
        ):
            continue

        total += 1

        input_path = os.path.join(
            root,
            filename
        )

        print(
            f"\nProcessing: {filename}"
        )

        image = cv2.imread(
            input_path
        )

        if image is None:

            print(
                "ERROR: Could not load image"
            )

            error_count += 1
            continue

        try:

            detections = detect_faces(
                image
            )

            # =================================================
            # FACE DETECTED
            # =================================================

            if len(detections) > 0:

                detected_count += 1

                result = image.copy()

                for (
                    x1,
                    y1,
                    x2,
                    y2,
                    confidence
                ) in detections:

                    cv2.rectangle(
                        result,
                        (x1, y1),
                        (x2, y2),
                        (0, 255, 0),
                        2
                    )

                    label = (
                        f"Face "
                        f"{confidence:.2f}"
                    )

                    cv2.putText(
                        result,
                        label,
                        (x1, max(20, y1 - 8)),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (0, 255, 0),
                        2
                    )

                output_path = os.path.join(
                    DETECTED_FOLDER,
                    filename
                )

                cv2.imwrite(
                    output_path,
                    result
                )

                print(
                    f"FACE DETECTED: "
                    f"{len(detections)}"
                )

                for detection in detections:

                    print(
                        "  Box:",
                        detection[:4],
                        "Confidence:",
                        f"{detection[4]:.4f}"
                    )

            # =================================================
            # NO FACE DETECTED
            # =================================================

            else:

                no_face_count += 1

                output_path = os.path.join(
                    NO_FACE_FOLDER,
                    filename
                )

                shutil.copy2(
                    input_path,
                    output_path
                )

                print(
                    "NO FACE DETECTED"
                )

        except Exception as e:

            error_count += 1

            print(
                "ERROR:",
                str(e)
            )

        processed += 1


# ============================================================
# FINAL RESULTS
# ============================================================

print("\n")
print("=" * 60)
print("SCRFD DETECTION COMPLETE")
print("=" * 60)

print(
    f"Total images       : {total}"
)

print(
    f"Processed          : {processed}"
)

print(
    f"Faces detected     : {detected_count}"
)

print(
    f"No face detected   : {no_face_count}"
)

print(
    f"Processing errors  : {error_count}"
)

if total > 0:

    detection_rate = (
        detected_count / total
    ) * 100

    no_face_rate = (
        no_face_count / total
    ) * 100

    print(
        f"Detection rate     : {detection_rate:.2f}%"
    )

    print(
        f"No-face rate       : {no_face_rate:.2f}%"
    )

print("\nOutput folders:")

print(
    "Detected images    :",
    DETECTED_FOLDER
)

print(
    "No-face images     :",
    NO_FACE_FOLDER
)

print("=" * 60)