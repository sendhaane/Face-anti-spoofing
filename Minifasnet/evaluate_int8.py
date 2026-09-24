import os
import cv2
import csv
import numpy as np
import tensorflow as tf


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.abspath(__file__)
)

TEST_DIR = os.path.join(
    BASE_DIR,
    "test_images"
)

REAL_DIR = os.path.join(
    TEST_DIR,
    "real"
)

SPOOF_DIR = os.path.join(
    TEST_DIR,
    "spoof"
)


# ============================================================
# SCRFD INT8 MODEL
# ============================================================

SCRFD_MODEL = os.path.join(
    BASE_DIR,
    "models",
    "scrfd_int8.tflite"
)


# ============================================================
# MINIFASNET INT8 MODEL
# ============================================================

ANTISPOOF_MODEL = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_int8.tflite"
)


# ============================================================
# OUTPUT CSV
# ============================================================

CSV_PATH = os.path.join(
    BASE_DIR,
    "evaluation_results_scrfd_int8.csv"
)


# ============================================================
# SCRFD CONFIGURATION
# ============================================================

CONF_THRESHOLD = 0.30

NMS_THRESHOLD = 0.40

STRIDES = [8, 16, 32]

NUM_ANCHORS = 2


# ============================================================
# MINIFASNET CONFIGURATION
# ============================================================

CROP_SCALE = 2.7

FACE_SIZE = 80


# ============================================================
# LOAD SCRFD
# ============================================================

print()
print("=" * 60)
print("LOADING SCRFD INT8")
print("=" * 60)

scrfd_interpreter = tf.lite.Interpreter(
    model_path=SCRFD_MODEL
)

scrfd_interpreter.allocate_tensors()

scrfd_input_details = (
    scrfd_interpreter.get_input_details()
)

scrfd_output_details = (
    scrfd_interpreter.get_output_details()
)

scrfd_input_info = scrfd_input_details[0]

input_h = int(
    scrfd_input_info["shape"][1]
)

input_w = int(
    scrfd_input_info["shape"][2]
)

print(
    "Input shape:",
    scrfd_input_info["shape"]
)

print(
    "Input dtype:",
    scrfd_input_info["dtype"]
)

print(
    "Input quantization:",
    scrfd_input_info["quantization"]
)

print()
print("SCRFD Outputs:")

for i, detail in enumerate(
    scrfd_output_details
):

    print(
        i,
        detail["name"],
        detail["shape"],
        detail["dtype"],
        "quantization:",
        detail["quantization"]
    )


# ============================================================
# LOAD MINIFASNET INT8
# ============================================================

print()
print("=" * 60)
print("LOADING MINIFASNETV2 INT8")
print("=" * 60)

antispoof_interpreter = tf.lite.Interpreter(
    model_path=ANTISPOOF_MODEL
)

antispoof_interpreter.allocate_tensors()

antispoof_input_details = (
    antispoof_interpreter.get_input_details()
)

antispoof_output_details = (
    antispoof_interpreter.get_output_details()
)

antispoof_input_info = (
    antispoof_input_details[0]
)

antispoof_output_info = (
    antispoof_output_details[0]
)

print(
    "Input shape:",
    antispoof_input_info["shape"]
)

print(
    "Input dtype:",
    antispoof_input_info["dtype"]
)

print(
    "Input quantization:",
    antispoof_input_info["quantization"]
)

print(
    "Output shape:",
    antispoof_output_info["shape"]
)

print(
    "Output dtype:",
    antispoof_output_info["dtype"]
)

print(
    "Output quantization:",
    antispoof_output_info["quantization"]
)


# ============================================================
# SOFTMAX
# ============================================================

def softmax(x):

    x = x - np.max(x)

    exp_x = np.exp(x)

    return exp_x / np.sum(exp_x)


# ============================================================
# SCRFD PREPROCESSING
# ============================================================

def preprocess_bgr(
    image,
    input_h,
    input_w,
    input_detail
):

    im_h, im_w = image.shape[:2]

    im_ratio = im_h / im_w

    model_ratio = input_h / input_w

    if im_ratio > model_ratio:

        new_h = input_h

        new_w = int(
            new_h / im_ratio
        )

    else:

        new_w = input_w

        new_h = int(
            new_w * im_ratio
        )

    det_scale = new_h / im_h

    resized = cv2.resize(
        image,
        (new_w, new_h)
    )

    canvas = np.zeros(
        (
            input_h,
            input_w,
            3
        ),
        dtype=np.uint8
    )

    canvas[
        :new_h,
        :new_w
    ] = resized

    # --------------------------------------------------------
    # BGR -> RGB
    # --------------------------------------------------------

    rgb = cv2.cvtColor(
        canvas,
        cv2.COLOR_BGR2RGB
    ).astype(
        np.float32
    )

    # --------------------------------------------------------
    # SCRFD normalization
    # --------------------------------------------------------

    normalized = (
        rgb - 127.5
    ) / 128.0

    batch = normalized[
        None,
        ...
    ]

    # --------------------------------------------------------
    # INT8 quantization
    # --------------------------------------------------------

    scale, zero_point = (
        input_detail["quantization"]
    )

    if (
        scale != 0
        and input_detail["dtype"]
        in (np.int8, np.uint8)
    ):

        batch = np.round(
            batch / scale
            + zero_point
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

    else:

        batch = batch.astype(
            input_detail["dtype"]
        )

    return batch, det_scale


# ============================================================
# SCRFD OUTPUT DEQUANTIZATION
# ============================================================

def dequantize_output(
    output,
    detail
):

    scale, zero_point = (
        detail["quantization"]
    )

    if (
        scale != 0
        and np.issubdtype(
            output.dtype,
            np.integer
        )
    ):

        output = (
            output.astype(
                np.float32
            )
            - zero_point
        ) * scale

    else:

        output = output.astype(
            np.float32
        )

    return output


# ============================================================
# ORGANIZE SCRFD OUTPUTS
# ============================================================

def organize_outputs(outputs):

    groups = []

    used = set()

    for i in range(
        len(outputs)
    ):

        if i in used:
            continue

        arr = outputs[i]

        # Remove batch dimension
        if arr.ndim == 3:
            arr = arr[0]

        if arr.ndim != 2:
            continue

        # Bounding box output
        if arr.shape[1] != 4:
            continue

        n = arr.shape[0]

        score_index = None

        landmark_index = None

        for j in range(
            len(outputs)
        ):

            if (
                j in used
                or j == i
            ):
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

                used.add(
                    landmark_index
                )

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

    feature_h = (
        input_h // stride
    )

    feature_w = (
        input_w // stride
    )

    expected = (
        feature_h
        * feature_w
        * NUM_ANCHORS
    )

    if len(boxes) != expected:

        print(
            f"Warning: stride {stride}: "
            f"expected {expected}, "
            f"got {len(boxes)}"
        )

        return [], []

    # --------------------------------------------------------
    # Anchor centers
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

            for _ in range(
                NUM_ANCHORS
            ):

                centers.append(
                    (cx, cy)
                )

    centers = np.asarray(
        centers,
        dtype=np.float32
    )

    # --------------------------------------------------------
    # Decode distances
    # --------------------------------------------------------

    distances = (
        boxes.astype(
            np.float32
        )
        * stride
    )

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
        [
            x1,
            y1,
            x2,
            y2
        ],
        axis=1
    )

    # --------------------------------------------------------
    # Convert detector coordinates
    # to original image coordinates
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

    order = scores.argsort()[
        ::-1
    ]

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

        intersection = (
            w * h
        )

        union = (
            areas[i]
            + areas[order[1:]]
            - intersection
        )

        iou = (
            intersection
            / np.maximum(
                union,
                1e-6
            )
        )

        remaining = np.where(
            iou <= threshold
        )[0]

        order = order[
            remaining + 1
        ]

    return keep


# ============================================================
# SCRFD FACE DETECTION
# ============================================================

def detect_faces(image):

    original_h, original_w = (
        image.shape[:2]
    )

    # --------------------------------------------------------
    # PREPROCESS
    # --------------------------------------------------------

    batch, det_scale = preprocess_bgr(
        image,
        input_h,
        input_w,
        scrfd_input_info
    )

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    scrfd_interpreter.set_tensor(
        scrfd_input_info["index"],
        batch
    )

    scrfd_interpreter.invoke()

    # --------------------------------------------------------
    # READ + DEQUANTIZE OUTPUTS
    # --------------------------------------------------------

    outputs = []

    for detail in scrfd_output_details:

        output = (
            scrfd_interpreter.get_tensor(
                detail["index"]
            )
        )

        output = dequantize_output(
            output,
            detail
        )

        outputs.append(
            output
        )

    # --------------------------------------------------------
    # ORGANIZE OUTPUT HEADS
    # --------------------------------------------------------

    groups = organize_outputs(
        outputs
    )

    if len(groups) == 0:

        print(
            "ERROR: Could not identify "
            "SCRFD output heads."
        )

        return []

    all_boxes = []

    all_scores = []

    # --------------------------------------------------------
    # PROCESS THREE SCRFD HEADS
    # --------------------------------------------------------

    for (
        box_index,
        score_index,
        landmark_index,
        n
    ) in groups:

        possible_stride = None

        for stride in STRIDES:

            feature_h = (
                input_h // stride
            )

            feature_w = (
                input_w // stride
            )

            expected = (
                feature_h
                * feature_w
                * NUM_ANCHORS
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

        decoded_boxes, decoded_scores = (
            decode_boxes(
                boxes,
                scores,
                possible_stride,
                det_scale
            )
        )

        if len(decoded_boxes) == 0:

            continue

        # ----------------------------------------------------
        # Confidence filtering
        # ----------------------------------------------------

        mask = (
            decoded_scores
            >= CONF_THRESHOLD
        )

        decoded_boxes = (
            decoded_boxes[mask]
        )

        decoded_scores = (
            decoded_scores[mask]
        )

        all_boxes.extend(
            decoded_boxes
        )

        all_scores.extend(
            decoded_scores
        )

    # --------------------------------------------------------
    # No detection
    # --------------------------------------------------------

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

        x1, y1, x2, y2 = (
            all_boxes[i]
        )

        x1 = max(
            0,
            min(
                w - 1,
                int(x1)
            )
        )

        y1 = max(
            0,
            min(
                h - 1,
                int(y1)
            )
        )

        x2 = max(
            0,
            min(
                w - 1,
                int(x2)
            )
        )

        y2 = max(
            0,
            min(
                h - 1,
                int(y2)
            )
        )

        if (
            x2 <= x1
            or y2 <= y1
        ):

            continue

        detections.append(
            (
                x1,
                y1,
                x2,
                y2,
                float(
                    all_scores[i]
                )
            )
        )

    return detections


# ============================================================
# SELECT BEST FACE
# ============================================================

def detect_face(image):

    detections = detect_faces(
        image
    )

    if len(detections) == 0:

        return None, 0.0

    # Highest-confidence face
    best = max(
        detections,
        key=lambda x: x[4]
    )

    x1, y1, x2, y2, confidence = (
        best
    )

    return (
        (
            x1,
            y1,
            x2,
            y2
        ),
        confidence
    )


# ============================================================
# MINI FASNET CROP
# ============================================================

def crop_face(
    image,
    box
):

    x1, y1, x2, y2 = box

    h, w = image.shape[:2]

    x1 = max(
        0,
        min(
            x1,
            w - 1
        )
    )

    y1 = max(
        0,
        min(
            y1,
            h - 1
        )
    )

    x2 = max(
        0,
        min(
            x2,
            w
        )
    )

    y2 = max(
        0,
        min(
            y2,
            h
        )
    )

    if (
        x2 <= x1
        or y2 <= y1
    ):

        return None

    face_h = y2 - y1

    face_w = x2 - x1

    center_x = (
        x1 + x2
    ) / 2

    center_y = (
        y1 + y2
    ) / 2

    new_w = (
        face_w
        * CROP_SCALE
    )

    new_h = (
        face_h
        * CROP_SCALE
    )

    nx1 = int(
        center_x
        - new_w / 2
    )

    ny1 = int(
        center_y
        - new_h / 2
    )

    nx2 = int(
        center_x
        + new_w / 2
    )

    ny2 = int(
        center_y
        + new_h / 2
    )

    nx1 = max(
        0,
        nx1
    )

    ny1 = max(
        0,
        ny1
    )

    nx2 = min(
        w,
        nx2
    )

    ny2 = min(
        h,
        ny2
    )

    if (
        nx2 <= nx1
        or ny2 <= ny1
    ):

        return None

    face = image[
        ny1:ny2,
        nx1:nx2
    ]

    if face.size == 0:

        return None

    face = cv2.resize(
        face,
        (
            FACE_SIZE,
            FACE_SIZE
        )
    )

    return face


# ============================================================
# MINIFASNET INT8 PREPROCESSING
# ============================================================

def preprocess_antispoof(
    face
):

    input_detail = (
        antispoof_input_info
    )

    # --------------------------------------------------------
    # BGR image remains BGR
    # --------------------------------------------------------
    #
    # This follows the same convention
    # used by the previous MiniFASNet
    # evaluation pipeline.
    #
    # Face is:
    #
    # H x W x 3
    #
    # converted to:
    #
    # 1 x H x W x 3
    # --------------------------------------------------------

    image = face.astype(
        np.float32
    )

    image = image[
        None,
        ...
    ]

    scale, zero_point = (
        input_detail["quantization"]
    )

    if (
        scale != 0
        and input_detail["dtype"]
        in (np.int8, np.uint8)
    ):

        image = np.round(
            image / scale
            + zero_point
        )

        info = np.iinfo(
            input_detail["dtype"]
        )

        image = np.clip(
            image,
            info.min,
            info.max
        ).astype(
            input_detail["dtype"]
        )

    else:

        image = image.astype(
            input_detail["dtype"]
        )

    return image


# ============================================================
# MINIFASNET INT8 INFERENCE
# ============================================================

def predict(face):

    input_tensor = (
        preprocess_antispoof(
            face
        )
    )

    antispoof_interpreter.set_tensor(
        antispoof_input_info["index"],
        input_tensor
    )

    antispoof_interpreter.invoke()

    output = (
        antispoof_interpreter.get_tensor(
            antispoof_output_info["index"]
        )
    )

    # --------------------------------------------------------
    # Remove only dimensions of size 1
    # --------------------------------------------------------

    output = np.squeeze(
        output
    )

    # --------------------------------------------------------
    # Dequantize MiniFASNet output
    # --------------------------------------------------------

    scale, zero_point = (
        antispoof_output_info[
            "quantization"
        ]
    )

    if scale != 0:

        logits = (
            output.astype(
                np.float32
            )
            - zero_point
        ) * scale

    else:

        logits = output.astype(
            np.float32
        )

    probabilities = softmax(
        logits
    )

    predicted_class = int(
        np.argmax(
            probabilities
        )
    )

    confidence = float(
        probabilities[
            predicted_class
        ]
    )

    # --------------------------------------------------------
    # MiniFASNet classes
    #
    # Class 1 = REAL
    # Class 0/2 = SPOOF
    # --------------------------------------------------------

    if predicted_class == 1:

        prediction = "REAL"

    else:

        prediction = "SPOOF"

    return (
        prediction,
        predicted_class,
        confidence,
        probabilities
    )


# ============================================================
# PROCESS ONE IMAGE
# ============================================================

def process_image(
    image_path,
    actual_label
):

    filename = os.path.basename(
        image_path
    )

    image = cv2.imread(
        image_path
    )

    # --------------------------------------------------------
    # Image loading failure
    # --------------------------------------------------------

    if image is None:

        return {
            "filename": filename,
            "actual": actual_label,
            "prediction": "FAILED",
            "status": "IMAGE_LOAD_FAILED",
            "detection_confidence": "",
            "class": "",
            "confidence": "",
            "class0": "",
            "class1": "",
            "class2": ""
        }

    # --------------------------------------------------------
    # SCRFD detection
    # --------------------------------------------------------

    try:

        box, detection_confidence = (
            detect_face(image)
        )

    except Exception as e:

        print(
            "Detection error:",
            str(e)
        )

        return {
            "filename": filename,
            "actual": actual_label,
            "prediction": "FAILED",
            "status": "FACE_DETECTION_ERROR",
            "detection_confidence": "",
            "class": "",
            "confidence": "",
            "class0": "",
            "class1": "",
            "class2": ""
        }

    # --------------------------------------------------------
    # Face not detected
    # --------------------------------------------------------

    if box is None:

        return {
            "filename": filename,
            "actual": actual_label,
            "prediction": "FAILED",
            "status": "FACE_NOT_DETECTED",
            "detection_confidence": 0.0,
            "class": "",
            "confidence": "",
            "class0": "",
            "class1": "",
            "class2": ""
        }

    # --------------------------------------------------------
    # Crop face
    # --------------------------------------------------------

    try:

        face = crop_face(
            image,
            box
        )

    except Exception as e:

        print(
            "Crop error:",
            str(e)
        )

        face = None

    if face is None:

        return {
            "filename": filename,
            "actual": actual_label,
            "prediction": "FAILED",
            "status": "INVALID_CROP",
            "detection_confidence": detection_confidence,
            "class": "",
            "confidence": "",
            "class0": "",
            "class1": "",
            "class2": ""
        }

    # --------------------------------------------------------
    # MiniFASNet INT8 inference
    # --------------------------------------------------------

    try:

        (
            prediction,
            predicted_class,
            confidence,
            probabilities
        ) = predict(face)

    except Exception as e:

        print(
            "Inference error:",
            str(e)
        )

        return {
            "filename": filename,
            "actual": actual_label,
            "prediction": "FAILED",
            "status": "INFERENCE_ERROR",
            "detection_confidence": detection_confidence,
            "class": "",
            "confidence": "",
            "class0": "",
            "class1": "",
            "class2": ""
        }

    # --------------------------------------------------------
    # Successful result
    # --------------------------------------------------------

    return {
        "filename": filename,
        "actual": actual_label,
        "prediction": prediction,
        "status": "SUCCESS",
        "detection_confidence": detection_confidence,
        "class": predicted_class,
        "confidence": confidence,
        "class0": probabilities[0],
        "class1": probabilities[1],
        "class2": probabilities[2]
    }


# ============================================================
# GET IMAGES
# ============================================================

def get_images(folder):

    valid_extensions = (
        ".jpg",
        ".jpeg",
        ".png",
        ".bmp",
        ".webp"
    )

    images = []

    if not os.path.exists(folder):

        return images

    for filename in sorted(
        os.listdir(folder)
    ):

        if filename.lower().endswith(
            valid_extensions
        ):

            images.append(
                os.path.join(
                    folder,
                    filename
                )
            )

    return images


# ============================================================
# LOAD DATASET
# ============================================================

real_images = get_images(
    REAL_DIR
)

spoof_images = get_images(
    SPOOF_DIR
)

print()
print("=" * 60)
print("DATASET")
print("=" * 60)

print(
    "Real images :",
    len(real_images)
)

print(
    "Spoof images:",
    len(spoof_images)
)

print(
    "Total       :",
    len(real_images)
    + len(spoof_images)
)


# ============================================================
# EVALUATION
# ============================================================

results = []

print()
print("=" * 60)
print("STARTING EVALUATION")
print("=" * 60)


# ============================================================
# REAL
# ============================================================

print()
print("Processing REAL images...")

for i, image_path in enumerate(
    real_images,
    1
):

    result = process_image(
        image_path,
        "REAL"
    )

    results.append(
        result
    )

    print(
        f"[REAL {i}/{len(real_images)}] "
        f"{result['filename']} → "
        f"{result['prediction']} "
        f"({result['status']})"
    )


# ============================================================
# SPOOF
# ============================================================

print()
print("Processing SPOOF images...")

for i, image_path in enumerate(
    spoof_images,
    1
):

    result = process_image(
        image_path,
        "SPOOF"
    )

    results.append(
        result
    )

    print(
        f"[SPOOF {i}/{len(spoof_images)}] "
        f"{result['filename']} → "
        f"{result['prediction']} "
        f"({result['status']})"
    )


# ============================================================
# SUCCESSFUL RESULTS
# ============================================================

successful = [
    r for r in results
    if r["status"] == "SUCCESS"
]

failed = [
    r for r in results
    if r["status"] != "SUCCESS"
]


# ============================================================
# CONFUSION MATRIX
# ============================================================

true_real = sum(
    1
    for r in successful
    if (
        r["actual"] == "REAL"
        and
        r["prediction"] == "REAL"
    )
)

false_real = sum(
    1
    for r in successful
    if (
        r["actual"] == "REAL"
        and
        r["prediction"] == "SPOOF"
    )
)

true_spoof = sum(
    1
    for r in successful
    if (
        r["actual"] == "SPOOF"
        and
        r["prediction"] == "SPOOF"
    )
)

false_spoof = sum(
    1
    for r in successful
    if (
        r["actual"] == "SPOOF"
        and
        r["prediction"] == "REAL"
    )
)


# ============================================================
# APCER
# ============================================================

if len(spoof_images) > 0:

    APCER = (
        false_spoof
        /
        len(spoof_images)
    ) * 100

else:

    APCER = 0.0


# ============================================================
# BPCER
# ============================================================

if len(real_images) > 0:

    BPCER = (
        false_real
        /
        len(real_images)
    ) * 100

else:

    BPCER = 0.0


# ============================================================
# ACER
# ============================================================

ACER = (
    APCER
    +
    BPCER
) / 2


# ============================================================
# ACCURACY
# ============================================================

if len(successful) > 0:

    accuracy = (
        (
            true_real
            +
            true_spoof
        )
        /
        len(successful)
    ) * 100

else:

    accuracy = 0.0


# ============================================================
# FAILURE BREAKDOWN
# ============================================================

face_not_detected = sum(
    1
    for r in results
    if r["status"]
    == "FACE_NOT_DETECTED"
)

invalid_crop = sum(
    1
    for r in results
    if r["status"]
    == "INVALID_CROP"
)

image_load_failed = sum(
    1
    for r in results
    if r["status"]
    == "IMAGE_LOAD_FAILED"
)

detection_error = sum(
    1
    for r in results
    if r["status"]
    == "FACE_DETECTION_ERROR"
)

inference_error = sum(
    1
    for r in results
    if r["status"]
    == "INFERENCE_ERROR"
)


# ============================================================
# DETECTION RATE
# ============================================================

if len(results) > 0:

    detection_failure_rate = (
        face_not_detected
        /
        len(results)
    ) * 100

else:

    detection_failure_rate = 0.0


# ============================================================
# FINAL RESULTS
# ============================================================

print()
print("=" * 60)
print("SCRFD INT8 + MINIFASNETV2 INT8")
print("EVALUATION RESULTS")
print("=" * 60)

print(
    f"Total images           : "
    f"{len(results)}"
)

print(
    f"Successful             : "
    f"{len(successful)}"
)

print(
    f"Failed                 : "
    f"{len(failed)}"
)

print()
print("Failure Breakdown")
print("-" * 42)

print(
    f"Face not detected     : "
    f"{face_not_detected}"
)

print(
    f"Invalid crop          : "
    f"{invalid_crop}"
)

print(
    f"Image load failure    : "
    f"{image_load_failed}"
)

print(
    f"Face detection error  : "
    f"{detection_error}"
)

print(
    f"Inference error       : "
    f"{inference_error}"
)

print()
print("Confusion Matrix")
print("-" * 42)

print(
    f"REAL → REAL           : "
    f"{true_real}"
)

print(
    f"REAL → SPOOF          : "
    f"{false_real}"
)

print(
    f"SPOOF → SPOOF         : "
    f"{true_spoof}"
)

print(
    f"SPOOF → REAL          : "
    f"{false_spoof}"
)

print()

print("Confusion Matrix")

print("-" * 42)

print(
    f"True Positive (TP)  : {true_real}"
)

print(
    f"True Negative (TN)  : {true_spoof}"
)

print(
    f"False Positive (FP) : {false_spoof}"
)

print(
    f"False Negative (FN) : {false_real}"
)

print()
print("Metrics")
print("-" * 42)

print(
    f"Accuracy              : "
    f"{accuracy:.2f}%"
)

print(
    f"APCER                 : "
    f"{APCER:.2f}%"
)

print(
    f"BPCER                 : "
    f"{BPCER:.2f}%"
)

print(
    f"ACER                  : "
    f"{ACER:.2f}%"
)

print(
    f"Detection failure rate: "
    f"{detection_failure_rate:.2f}%"
)

print()
print("CSV saved to:")
print(CSV_PATH)

print()
print("=" * 60)
print("EVALUATION COMPLETE")
print("=" * 60)