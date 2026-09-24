import os
import sys
import cv2
import time
import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\Face_anti_spoofing\Minifasnet"

# ------------------------------------------------------------
# SCRFD INT8 model
# ------------------------------------------------------------

SCRFD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scrfd_int8.tflite"
)

# ------------------------------------------------------------
# MiniFASNetV2 INT8 model
# ------------------------------------------------------------

ANTISPOOF_MODEL_PATH = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_int8.tflite"
)


# ============================================================
# IMPORT OFFICIAL MINIFASNET CROP
# ============================================================

sys.path.insert(
    0,
    BASE_DIR
)

from src.generate_patches import CropImage


# ============================================================
# SCRFD CONFIGURATION
# ============================================================

CONF_THRESHOLD = 0.30

NMS_THRESHOLD = 0.40

STRIDES = [
    8,
    16,
    32
]

NUM_ANCHORS = 2


# ============================================================
# SCRFD INT8 DETECTOR
# ============================================================

class SCRFDINT8:

    def __init__(self, model_path):

        print()
        print("=" * 60)
        print("LOADING SCRFD INT8")
        print("=" * 60)

        self.interpreter = tf.lite.Interpreter(
            model_path=model_path
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        self.input_info = (
            self.input_details[0]
        )

        self.input_h = int(
            self.input_info["shape"][1]
        )

        self.input_w = int(
            self.input_info["shape"][2]
        )

        print(
            "Input shape       :",
            self.input_info["shape"]
        )

        print(
            "Input dtype       :",
            self.input_info["dtype"]
        )

        print(
            "Input quantization:",
            self.input_info["quantization"]
        )

        print()
        print("SCRFD outputs:")

        for i, detail in enumerate(
            self.output_details
        ):

            print(
                i,
                detail["name"],
                detail["shape"],
                detail["dtype"],
                "quantization:",
                detail["quantization"]
            )


    # ========================================================
    # PREPROCESS
    # ========================================================

    def preprocess(
        self,
        image
    ):

        im_h, im_w = image.shape[:2]

        im_ratio = im_h / im_w

        model_ratio = (
            self.input_h /
            self.input_w
        )

        if im_ratio > model_ratio:

            new_h = self.input_h

            new_w = int(
                new_h / im_ratio
            )

        else:

            new_w = self.input_w

            new_h = int(
                new_w * im_ratio
            )

        det_scale = (
            new_h /
            im_h
        )

        resized = cv2.resize(
            image,
            (new_w, new_h)
        )

        canvas = np.zeros(
            (
                self.input_h,
                self.input_w,
                3
            ),
            dtype=np.uint8
        )

        canvas[
            :new_h,
            :new_w
        ] = resized

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        rgb = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB
        ).astype(
            np.float32
        )

        # ----------------------------------------------------
        # SCRFD normalization
        # ----------------------------------------------------

        normalized = (
            rgb - 127.5
        ) / 128.0

        batch = normalized[
            None,
            ...
        ]

        # ----------------------------------------------------
        # INT8 quantization
        # ----------------------------------------------------

        scale, zero_point = (
            self.input_info["quantization"]
        )

        if (
            scale != 0
            and self.input_info["dtype"]
            in (np.int8, np.uint8)
        ):

            batch = np.round(
                batch / scale
                + zero_point
            )

            info = np.iinfo(
                self.input_info["dtype"]
            )

            batch = np.clip(
                batch,
                info.min,
                info.max
            ).astype(
                self.input_info["dtype"]
            )

        else:

            batch = batch.astype(
                self.input_info["dtype"]
            )

        return (
            batch,
            det_scale
        )


    # ========================================================
    # DEQUANTIZE OUTPUT
    # ========================================================

    def dequantize_output(
        self,
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


    # ========================================================
    # ORGANIZE SCRFD OUTPUTS
    # ========================================================

    def organize_outputs(
        self,
        outputs
    ):

        groups = []

        used = set()

        for i in range(
            len(outputs)
        ):

            if i in used:
                continue

            arr = outputs[i]

            if arr.ndim == 3:
                arr = arr[0]

            if arr.ndim != 2:
                continue

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

                used.add(
                    score_index
                )

                if landmark_index is not None:

                    used.add(
                        landmark_index
                    )

        return groups


    # ========================================================
    # DECODE BOXES
    # ========================================================

    def decode_boxes(
        self,
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
            self.input_h //
            stride
        )

        feature_w = (
            self.input_w //
            stride
        )

        expected = (
            feature_h
            * feature_w
            * NUM_ANCHORS
        )

        if len(boxes) != expected:

            return (
                np.array([]),
                np.array([])
            )

        # ----------------------------------------------------
        # Anchor centers
        # ----------------------------------------------------

        centers = []

        for y in range(
            feature_h
        ):

            for x in range(
                feature_w
            ):

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

        # ----------------------------------------------------
        # SCRFD box distances
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Convert to original image coordinates
        # ----------------------------------------------------

        decoded /= det_scale

        return (
            decoded,
            scores
        )


    # ========================================================
    # NMS
    # ========================================================

    def nms(
        self,
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

        order = (
            scores.argsort()[::-1]
        )

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


    # ========================================================
    # DETECT FACES
    # ========================================================

    def detect_faces(
        self,
        image
    ):

        batch, det_scale = (
            self.preprocess(image)
        )

        self.interpreter.set_tensor(
            self.input_info["index"],
            batch
        )

        self.interpreter.invoke()

        # ----------------------------------------------------
        # Read outputs
        # ----------------------------------------------------

        outputs = []

        for detail in (
            self.output_details
        ):

            output = (
                self.interpreter.get_tensor(
                    detail["index"]
                )
            )

            output = (
                self.dequantize_output(
                    output,
                    detail
                )
            )

            outputs.append(
                output
            )

        # ----------------------------------------------------
        # Organize SCRFD heads
        # ----------------------------------------------------

        groups = (
            self.organize_outputs(
                outputs
            )
        )

        if len(groups) == 0:

            return []

        all_boxes = []

        all_scores = []

        # ----------------------------------------------------
        # Decode each detection level
        # ----------------------------------------------------

        for (
            box_index,
            score_index,
            landmark_index,
            n
        ) in groups:

            possible_stride = None

            for stride in STRIDES:

                feature_h = (
                    self.input_h //
                    stride
                )

                feature_w = (
                    self.input_w //
                    stride
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
                self.decode_boxes(
                    boxes,
                    scores,
                    possible_stride,
                    det_scale
                )
            )

            if len(decoded_boxes) == 0:

                continue

            # ------------------------------------------------
            # Confidence filtering
            # ------------------------------------------------

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

        # ----------------------------------------------------
        # NMS
        # ----------------------------------------------------

        keep = self.nms(
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
# MINIFASNETV2 INT8
# ============================================================

class MiniFASNetINT8:

    def __init__(
        self,
        model_path
    ):

        print()
        print("=" * 60)
        print("LOADING MINIFASNETV2 INT8")
        print("=" * 60)

        self.interpreter = (
            tf.lite.Interpreter(
                model_path=model_path
            )
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        self.input_index = (
            self.input_details[0]["index"]
        )

        self.output_index = (
            self.output_details[0]["index"]
        )

        self.input_scale = (
            self.input_details[0][
                "quantization"
            ][0]
        )

        self.input_zero_point = (
            self.input_details[0][
                "quantization"
            ][1]
        )

        self.output_scale = (
            self.output_details[0][
                "quantization"
            ][0]
        )

        self.output_zero_point = (
            self.output_details[0][
                "quantization"
            ][1]
        )

        print(
            "Input shape       :",
            self.input_details[0]["shape"]
        )

        print(
            "Input dtype       :",
            self.input_details[0]["dtype"]
        )

        print(
            "Input scale       :",
            self.input_scale
        )

        print(
            "Input zero point  :",
            self.input_zero_point
        )

        print(
            "Output shape      :",
            self.output_details[0]["shape"]
        )

        print(
            "Output dtype      :",
            self.output_details[0]["dtype"]
        )

        print(
            "Output scale      :",
            self.output_scale
        )

        print(
            "Output zero point :",
            self.output_zero_point
        )


    # ========================================================
    # PREDICT
    # ========================================================

    def predict(
        self,
        face
    ):

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        face = cv2.resize(
            face,
            (80, 80),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # ----------------------------------------------------
        # Float32 0-255
        # ----------------------------------------------------

        face = face.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Add batch dimension
        # ----------------------------------------------------

        face = np.expand_dims(
            face,
            axis=0
        )

        # ----------------------------------------------------
        # Quantize
        # ----------------------------------------------------

        face_int8 = np.round(
            face / self.input_scale
            + self.input_zero_point
        )

        face_int8 = np.clip(
            face_int8,
            -128,
            127
        ).astype(
            np.int8
        )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        self.interpreter.set_tensor(
            self.input_index,
            face_int8
        )

        self.interpreter.invoke()

        # ----------------------------------------------------
        # Raw output
        # ----------------------------------------------------

        raw_output = (
            self.interpreter.get_tensor(
                self.output_index
            )
        )

        # ----------------------------------------------------
        # Dequantize
        # ----------------------------------------------------

        logits = (
            self.output_scale
            * (
                raw_output.astype(
                    np.float32
                )
                - self.output_zero_point
            )
        )

        # ----------------------------------------------------
        # Softmax
        # ----------------------------------------------------

        logits_shifted = (
            logits
            - np.max(
                logits,
                axis=1,
                keepdims=True
            )
        )

        exp_logits = np.exp(
            logits_shifted
        )

        probabilities = (
            exp_logits
            / np.sum(
                exp_logits,
                axis=1,
                keepdims=True
            )
        )

        return (
            raw_output,
            logits,
            probabilities
        )


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("=" * 60)
    print("SCRFD INT8 + MINIFASNETV2 INT8 WEBCAM")
    print("=" * 60)

    # --------------------------------------------------------
    # Load SCRFD
    # --------------------------------------------------------

    detector = SCRFDINT8(
        SCRFD_MODEL_PATH
    )

    # --------------------------------------------------------
    # Load MiniFASNet
    # --------------------------------------------------------

    antispoof = MiniFASNetINT8(
        ANTISPOOF_MODEL_PATH
    )

    # --------------------------------------------------------
    # Official MiniFASNet cropper
    # --------------------------------------------------------

    cropper = CropImage()

    CROP_SCALE = 2.7

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print(
            "\nERROR: Could not open webcam."
        )

        return

    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    print()
    print("Webcam started.")
    print("Press 'q' to quit.")

    # --------------------------------------------------------
    # FPS
    # --------------------------------------------------------

    prev_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to read frame."
            )

            break

        # ----------------------------------------------------
        # SCRFD FACE DETECTION
        # ----------------------------------------------------

        try:

            detections = (
                detector.detect_faces(
                    frame
                )
            )

        except Exception as e:

            print(
                "SCRFD detection error:",
                e
            )

            detections = []

        # ----------------------------------------------------
        # Select highest-confidence face
        # ----------------------------------------------------

        if len(detections) > 0:

            best_detection = max(
                detections,
                key=lambda x: x[4]
            )

            (
                x1,
                y1,
                x2,
                y2,
                detection_confidence
            ) = best_detection

            # ------------------------------------------------
            # Convert SCRFD xyxy -> official
            # MiniFASNet xywh format
            # ------------------------------------------------

            bbox = [
                x1,
                y1,
                x2 - x1,
                y2 - y1
            ]

            # ------------------------------------------------
            # Official MiniFASNet crop
            # ------------------------------------------------

            face = cropper.crop(
                frame,
                bbox,
                CROP_SCALE,
                80,
                80
            )

            # ------------------------------------------------
            # MiniFASNet INT8 inference
            # ------------------------------------------------

            try:

                (
                    raw_output,
                    logits,
                    probabilities
                ) = antispoof.predict(
                    face
                )

                # ------------------------------------------------
                # Classes
                #
                # 0 = SPOOF
                # 1 = REAL
                # 2 = SPOOF
                # ------------------------------------------------

                class_0 = float(
                    probabilities[0][0]
                )

                class_1 = float(
                    probabilities[0][1]
                )

                class_2 = float(
                    probabilities[0][2]
                )

                predicted_class = int(
                    np.argmax(
                        probabilities[0]
                    )
                )

                real_probability = (
                    class_1
                )

                spoof_probability = (
                    class_0
                    + class_2
                )

                if predicted_class == 1:

                    result = "REAL"

                else:

                    result = "SPOOF"

                # ------------------------------------------------
                # Draw face bounding box
                # ------------------------------------------------

                cv2.rectangle(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # Main result
                # ------------------------------------------------

                if result == "REAL":

                    label_text = (
                        f"REAL "
                        f"{real_probability * 100:.1f}%"
                    )

                else:

                    label_text = (
                        f"SPOOF "
                        f"{spoof_probability * 100:.1f}%"
                    )

                cv2.putText(
                    frame,
                    label_text,
                    (
                        x1,
                        max(
                            y1 - 10,
                            30
                        )
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (0, 255, 0),
                    2
                )

                # ------------------------------------------------
                # Detection confidence
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    (
                        f"Face: "
                        f"{detection_confidence:.2f}"
                    ),
                    (
                        10,
                        30
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                # ------------------------------------------------
                # Class probabilities
                # ------------------------------------------------

                cv2.putText(
                    frame,
                    (
                        f"C0 Spoof: "
                        f"{class_0 * 100:.1f}%"
                    ),
                    (
                        10,
                        55
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"C1 Real: "
                        f"{class_1 * 100:.1f}%"
                    ),
                    (
                        10,
                        80
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

                cv2.putText(
                    frame,
                    (
                        f"C2 Spoof: "
                        f"{class_2 * 100:.1f}%"
                    ),
                    (
                        10,
                        105
                    ),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (255, 255, 255),
                    2
                )

            except Exception as e:

                print(
                    "MiniFASNet inference error:",
                    e
                )

                cv2.putText(
                    frame,
                    "Anti-spoof inference error",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (0, 0, 255),
                    2
                )

        else:

            # ------------------------------------------------
            # No face
            # ------------------------------------------------

            cv2.putText(
                frame,
                "No face detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ====================================================
        # FPS
        # ====================================================

        current_time = time.time()

        elapsed = (
            current_time -
            prev_time
        )

        if elapsed > 0:

            fps = 1.0 / elapsed

        else:

            fps = 0.0

        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (
                10,
                135
            ),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            "SCRFD INT8 + MiniFASNetV2 INT8",
            frame
        )

        # ====================================================
        # QUIT
        # ====================================================

        key = (
            cv2.waitKey(1)
            & 0xFF
        )

        if key == ord("q"):

            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    print()
    print("Webcam stopped.")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()