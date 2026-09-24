import os
import sys
import cv2
import numpy as np
import tensorflow as tf


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = r"D:\Face_anti_spoofing\Minifasnet"


# ============================================================
# INPUT IMAGE
# ============================================================

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "test_images",
    "3.jpg"
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
# MINIFASNETV2 INT8 MODEL
# ============================================================

MINIFAS_MODEL = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_int8.tflite"
)


# ============================================================
# OUTPUT DIRECTORIES
# ============================================================

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "result"
)

CROP_DIR = os.path.join(
    BASE_DIR,
    "cropped_faces"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    CROP_DIR,
    exist_ok=True
)


# ============================================================
# SCRFD CONFIGURATION
# ============================================================

FACE_THRESHOLD = 0.30

NMS_THRESHOLD = 0.40

STRIDES = [8, 16, 32]

NUM_ANCHORS = 2


# ============================================================
# MINIFASNET CROP
# ============================================================

sys.path.insert(
    0,
    BASE_DIR
)

from src.generate_patches import CropImage


# ============================================================
# SCRFD INT8 DETECTOR
# ============================================================

class SCRFDINT8:

    def __init__(self, model_path):

        print()
        print("==========================================")
        print("LOADING SCRFD INT8")
        print("==========================================")

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

        self.input_detail = (
            self.input_details[0]
        )

        self.input_h = int(
            self.input_detail["shape"][1]
        )

        self.input_w = int(
            self.input_detail["shape"][2]
        )

        print(
            "Input shape:",
            self.input_detail["shape"]
        )

        print(
            "Input dtype:",
            self.input_detail["dtype"]
        )

        print(
            "Input quantization:",
            self.input_detail["quantization"]
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

    def preprocess(self, image):

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
            new_h / im_h
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
            self.input_detail["quantization"]
        )

        if (
            scale != 0
            and self.input_detail["dtype"]
            in (np.int8, np.uint8)
        ):

            batch = np.round(
                batch / scale
                + zero_point
            )

            info = np.iinfo(
                self.input_detail["dtype"]
            )

            batch = np.clip(
                batch,
                info.min,
                info.max
            ).astype(
                self.input_detail["dtype"]
            )

        else:

            batch = batch.astype(
                self.input_detail["dtype"]
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
            self.input_h // stride
        )

        feature_w = (
            self.input_w // stride
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

        # ----------------------------------------------------
        # Generate anchor centers
        #
        # IMPORTANT:
        # These are the same centers used
        # by your working test_detector.
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
        # Decode distances
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

        order = scores.argsort(
            descending=False
        ) if False else scores.argsort()[::-1]

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
            self.preprocess(
                image
            )
        )

        self.interpreter.set_tensor(
            self.input_detail["index"],
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
        # Organize heads
        # ----------------------------------------------------

        groups = (
            self.organize_outputs(
                outputs
            )
        )

        if len(groups) == 0:

            print(
                "ERROR: Could not identify "
                "SCRFD output heads."
            )

            return []

        all_boxes = []

        all_scores = []

        # ----------------------------------------------------
        # Process all detection levels
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
                    self.input_h
                    // stride
                )

                feature_w = (
                    self.input_w
                    // stride
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

            (
                decoded_boxes,
                decoded_scores
            ) = self.decode_boxes(
                boxes,
                scores,
                possible_stride,
                det_scale
            )

            if len(decoded_boxes) == 0:

                continue

            # ------------------------------------------------
            # Confidence filtering
            # ------------------------------------------------

            mask = (
                decoded_scores
                >= FACE_THRESHOLD
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
        print("==========================================")
        print("LOADING MINIFASNETV2 INT8")
        print("==========================================")

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
            "Input shape:",
            self.input_details[0]["shape"]
        )

        print(
            "Input dtype:",
            self.input_details[0]["dtype"]
        )

        print(
            "Input quantization:",
            self.input_details[0][
                "quantization"
            ]
        )

        print(
            "Output shape:",
            self.output_details[0]["shape"]
        )

        print(
            "Output dtype:",
            self.output_details[0]["dtype"]
        )

        print(
            "Output quantization:",
            self.output_details[0][
                "quantization"
            ]
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
        # Float 0-255
        # ----------------------------------------------------

        face = face.astype(
            np.float32
        )

        # ----------------------------------------------------
        # Batch
        # ----------------------------------------------------

        face = np.expand_dims(
            face,
            axis=0
        )

        # ----------------------------------------------------
        # Quantize
        # ----------------------------------------------------

        if (
            self.input_details[0][
                "dtype"
            ] in (np.int8, np.uint8)
            and self.input_scale != 0
        ):

            quantized_face = np.round(
                face
                / self.input_scale
                + self.input_zero_point
            )

            dtype = (
                self.input_details[0][
                    "dtype"
                ]
            )

            info = np.iinfo(dtype)

            quantized_face = np.clip(
                quantized_face,
                info.min,
                info.max
            ).astype(dtype)

        else:

            quantized_face = (
                face.astype(
                    self.input_details[0][
                        "dtype"
                    ]
                )
            )

        # ----------------------------------------------------
        # Inference
        # ----------------------------------------------------

        self.interpreter.set_tensor(
            self.input_index,
            quantized_face
        )

        self.interpreter.invoke()

        # ----------------------------------------------------
        # Output
        # ----------------------------------------------------

        raw_output = (
            self.interpreter.get_tensor(
                self.output_index
            )
        )

        # ----------------------------------------------------
        # Dequantize
        # ----------------------------------------------------

        if (
            self.output_scale != 0
        ):

            logits = (
                raw_output.astype(
                    np.float32
                )
                - self.output_zero_point
            ) * self.output_scale

        else:

            logits = raw_output.astype(
                np.float32
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

        predicted_class = int(
            np.argmax(
                probabilities[0]
            )
        )

        # ----------------------------------------------------
        # MiniFASNet classes
        #
        # 0 = SPOOF
        # 1 = REAL
        # 2 = SPOOF
        # ----------------------------------------------------

        if predicted_class == 1:

            prediction = "REAL"

        else:

            prediction = "SPOOF"

        real_probability = float(
            probabilities[0][1]
        )

        spoof_probability = float(
            probabilities[0][0]
            + probabilities[0][2]
        )

        return {
            "raw_output": raw_output,
            "logits": logits,
            "probabilities": probabilities,
            "class": predicted_class,
            "prediction": prediction,
            "real_probability": real_probability,
            "spoof_probability": spoof_probability
        }


# ============================================================
# MAIN
# ============================================================

def main():

    print()
    print("==========================================")
    print("SCRFD INT8 + MINIFASNETV2 INT8")
    print("==========================================")

    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(
        IMAGE_PATH
    ):

        print(
            "ERROR: Image not found:"
        )

        print(
            IMAGE_PATH
        )

        return

    if not os.path.exists(
        SCRFD_MODEL
    ):

        print(
            "ERROR: SCRFD model not found:"
        )

        print(
            SCRFD_MODEL
        )

        return

    if not os.path.exists(
        MINIFAS_MODEL
    ):

        print(
            "ERROR: MiniFASNet model not found:"
        )

        print(
            MINIFAS_MODEL
        )

        return

    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        IMAGE_PATH
    )

    if image is None:

        print(
            "ERROR: Could not read image."
        )

        return

    original_image = image.copy()

    # --------------------------------------------------------
    # Load models
    # --------------------------------------------------------

    detector = SCRFDINT8(
        SCRFD_MODEL
    )

    antispoof = MiniFASNetINT8(
        MINIFAS_MODEL
    )

    # --------------------------------------------------------
    # Face detection
    # --------------------------------------------------------

    print()
    print("==========================================")
    print("FACE DETECTION")
    print("==========================================")

    detections = detector.detect_faces(
        image
    )

    if len(detections) == 0:

        print()
        print(
            "NO FACE DETECTED"
        )

        # Save original image with message

        cv2.putText(
            image,
            "NO FACE DETECTED",
            (30, 50),
            cv2.FONT_HERSHEY_SIMPLEX,
            1.0,
            (0, 0, 255),
            2
        )

        output_path = os.path.join(
            OUTPUT_DIR,
            "3_no_face.jpg"
        )

        cv2.imwrite(
            output_path,
            image
        )

        print(
            "Output saved:",
            output_path
        )

        return

    print(
        "Faces detected:",
        len(detections)
    )

    # --------------------------------------------------------
    # Select highest confidence face
    # --------------------------------------------------------

    best_detection = max(
        detections,
        key=lambda x: x[4]
    )

    x1, y1, x2, y2, detection_confidence = (
        best_detection
    )

    print(
        "Best face box:",
        (x1, y1, x2, y2)
    )

    print(
        "Detection confidence:",
        f"{detection_confidence:.4f}"
    )

    # --------------------------------------------------------
    # Convert to official CropImage format
    #
    # CropImage expects:
    #
    # [x, y, width, height]
    # --------------------------------------------------------

    bbox = [
        x1,
        y1,
        x2 - x1 + 1,
        y2 - y1 + 1
    ]

    # --------------------------------------------------------
    # Official MiniFASNet crop
    # --------------------------------------------------------

    cropper = CropImage()

    face = cropper.crop(
        image,
        bbox,
        2.7,
        80,
        80
    )

    if face is None:

        print(
            "ERROR: Face crop failed."
        )

        return

    # --------------------------------------------------------
    # Save crop
    # --------------------------------------------------------

    crop_path = os.path.join(
        CROP_DIR,
        "scrfd_int8_minifasnet_crop.jpg"
    )

    cv2.imwrite(
        crop_path,
        face
    )

    print(
        "Face crop saved:",
        crop_path
    )

    # --------------------------------------------------------
    # MiniFASNet prediction
    # --------------------------------------------------------

    print()
    print("==========================================")
    print("MINIFASNETV2 INT8 PREDICTION")
    print("==========================================")

    result = antispoof.predict(
        face
    )

    raw_output = result[
        "raw_output"
    ]

    logits = result[
        "logits"
    ]

    probabilities = result[
        "probabilities"
    ]

    predicted_class = result[
        "class"
    ]

    prediction = result[
        "prediction"
    ]

    real_probability = result[
        "real_probability"
    ]

    spoof_probability = result[
        "spoof_probability"
    ]

    # --------------------------------------------------------
    # Print model results
    # --------------------------------------------------------

    print()
    print("Raw INT8 output:")
    print(raw_output)

    print()
    print("Dequantized logits:")
    print(logits)

    print()
    print("Probabilities:")
    print(probabilities)

    print()
    print(
        f"Class 0 SPOOF : "
        f"{probabilities[0][0] * 100:.2f}%"
    )

    print(
        f"Class 1 REAL  : "
        f"{probabilities[0][1] * 100:.2f}%"
    )

    print(
        f"Class 2 SPOOF : "
        f"{probabilities[0][2] * 100:.2f}%"
    )

    print()
    print(
        f"REAL probability  : "
        f"{real_probability * 100:.2f}%"
    )

    print(
        f"SPOOF probability : "
        f"{spoof_probability * 100:.2f}%"
    )

    print()
    print(
        "Predicted class:",
        predicted_class
    )

    print(
        "Final result:",
        prediction
    )

    # --------------------------------------------------------
    # Draw bounding box
    # --------------------------------------------------------

    annotated = original_image.copy()

    if prediction == "REAL":

        display_text = (
            f"REAL "
            f"{real_probability * 100:.1f}%"
        )

        text_color = (
            0,
            255,
            0
        )

    else:

        display_text = (
            f"SPOOF "
            f"{spoof_probability * 100:.1f}%"
        )

        text_color = (
            0,
            0,
            255
        )

    # --------------------------------------------------------
    # Bounding box
    # --------------------------------------------------------

    cv2.rectangle(
        annotated,
        (x1, y1),
        (x2, y2),
        text_color,
        3
    )

    # --------------------------------------------------------
    # Detection confidence
    # --------------------------------------------------------

    detection_text = (
        f"Face: "
        f"{detection_confidence:.2f}"
    )

    cv2.putText(
        annotated,
        detection_text,
        (
            x1,
            max(
                y1 - 40,
                25
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        text_color,
        2
    )

    # --------------------------------------------------------
    # REAL / SPOOF result
    # --------------------------------------------------------

    cv2.putText(
        annotated,
        display_text,
        (
            x1,
            max(
                y1 - 10,
                25
            )
        ),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        text_color,
        2
    )

    # --------------------------------------------------------
    # Save output
    # --------------------------------------------------------

    image_name = os.path.splitext(
        os.path.basename(
            IMAGE_PATH
        )
    )[0]

    image_ext = os.path.splitext(
        IMAGE_PATH
    )[1]

    output_path = os.path.join(
        OUTPUT_DIR,
        f"{image_name}_scrfd_int8_minifasnet_int8{image_ext}"
    )

    cv2.imwrite(
        output_path,
        annotated
    )

    print()
    print("==========================================")
    print("RESULT")
    print("==========================================")

    print(
        "Detection confidence:",
        f"{detection_confidence:.4f}"
    )

    print(
        "MiniFASNet result:",
        prediction
    )

    print(
        "REAL probability:",
        f"{real_probability * 100:.2f}%"
    )

    print(
        "SPOOF probability:",
        f"{spoof_probability * 100:.2f}%"
    )

    print()
    print(
        "Annotated image saved:"
    )

    print(
        output_path
    )

    print()
    print("==========================================")
    print("COMPLETE")
    print("==========================================")


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()