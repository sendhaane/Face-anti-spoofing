
import cv2
import numpy as np
import tensorflow as tf


class FaceDetector:

    def __init__(
        self,
        model_path,
        conf_threshold=0.30,
        nms_threshold=0.40
    ):
        """
        SCRFD INT8 TFLite face detector.

        Parameters
        ----------
        model_path : str
            Path to scrfd_int8.tflite

        conf_threshold : float
            Detection confidence threshold.

        nms_threshold : float
            NMS IoU threshold.
        """

        self.model_path = model_path

        self.conf_threshold = conf_threshold
        self.nms_threshold = nms_threshold

        self.strides = [8, 16, 32]

        # SCRFD uses 2 anchors per location
        self.num_anchors = 2

        # -----------------------------------------------------
        # Load TFLite model
        # -----------------------------------------------------

        print("=" * 60)
        print("LOADING SCRFD INT8 FACE DETECTOR")
        print("=" * 60)

        self.interpreter = tf.lite.Interpreter(
            model_path=self.model_path
        )

        self.interpreter.allocate_tensors()

        self.input_details = (
            self.interpreter.get_input_details()
        )

        self.output_details = (
            self.interpreter.get_output_details()
        )

        self.input_info = self.input_details[0]

        self.input_h = int(
            self.input_info["shape"][1]
        )

        self.input_w = int(
            self.input_info["shape"][2]
        )

        print(
            "Input shape:",
            self.input_info["shape"]
        )

        print(
            "Input dtype:",
            self.input_info["dtype"]
        )

        print(
            "Input quantization:",
            self.input_info["quantization"]
        )

        print("\nOutputs:")

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

        print("=" * 60)

    # =========================================================
    # PREPROCESS
    # =========================================================

    def preprocess(self, image):
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

        # BGR -> RGB
        rgb = cv2.cvtColor(
            canvas,
            cv2.COLOR_BGR2RGB
        ).astype(np.float32)

        # Normalize
        normalized = (
            rgb - 127.5
        ) / 128.0

        # Add batch dimension
        batch = normalized[None, ...]

        # -----------------------------------------------------
        # INT8 quantization
        # -----------------------------------------------------

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

        return batch, det_scale

    # =========================================================
    # DEQUANTIZE OUTPUT
    # =========================================================

    @staticmethod
    def dequantize_output(
        output,
        detail
    ):
        """
        Convert INT8/UINT8 output to float32.
        """

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
                output.astype(np.float32)
                - zero_point
            ) * scale

        else:

            output = output.astype(
                np.float32
            )

        return output

    # =========================================================
    # ORGANIZE OUTPUTS
    # =========================================================

    @staticmethod
    def organize_outputs(outputs):
        """
        Identify SCRFD detection groups.

        Each detection level contains:

            bbox     -> (..., 4)
            score    -> (..., 1)
            landmark -> (..., 10)

        The output ordering may vary.
        """

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

    # =========================================================
    # DECODE BOXES
    # =========================================================

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

        # -----------------------------------------------------
        # Number of locations
        # -----------------------------------------------------

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
            * self.num_anchors
        )

        if len(boxes) != expected:

            print(
                f"Warning: stride {stride}: "
                f"expected {expected}, "
                f"got {len(boxes)}"
            )

            return [], []

        # -----------------------------------------------------
        # Generate anchor centers
        # -----------------------------------------------------

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
                    self.num_anchors
                ):

                    centers.append(
                        (cx, cy)
                    )

        centers = np.asarray(
            centers,
            dtype=np.float32
        )

        # -----------------------------------------------------
        # Decode distances
        # -----------------------------------------------------

        distances = (
            boxes.astype(np.float32)
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

        # Convert back to original
        # image coordinates
        decoded /= det_scale

        return decoded, scores

    # =========================================================
    # NMS
    # =========================================================

    @staticmethod
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

    # =========================================================
    # DETECT FACES
    # =========================================================

    def detect_faces(self, image):
        """
        Detect faces in a BGR image.

        Returns
        -------
        list of tuples

        [
            (
                x1,
                y1,
                x2,
                y2,
                confidence
            ),
            ...
        ]
        """

        batch, det_scale = (
            self.preprocess(image)
        )

        # -----------------------------------------------------
        # Run TFLite inference
        # -----------------------------------------------------

        self.interpreter.set_tensor(
            self.input_info["index"],
            batch
        )

        self.interpreter.invoke()

        # -----------------------------------------------------
        # Read and dequantize outputs
        # -----------------------------------------------------

        outputs = []

        for detail in (
            self.output_details
        ):

            output = (
                self.interpreter
                .get_tensor(
                    detail["index"]
                )
            )

            output = (
                self.dequantize_output(
                    output,
                    detail
                )
            )

            outputs.append(output)

        # -----------------------------------------------------
        # Organize detection heads
        # -----------------------------------------------------

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

        # -----------------------------------------------------
        # Process each SCRFD level
        # -----------------------------------------------------

        for (
            box_index,
            score_index,
            landmark_index,
            n
        ) in groups:

            possible_stride = None

            for stride in self.strides:

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
                    * self.num_anchors
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

            # -------------------------------------------------
            # Confidence filtering
            # -------------------------------------------------

            mask = (
                decoded_scores
                >= self.conf_threshold
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

        # -----------------------------------------------------
        # NMS
        # -----------------------------------------------------

        keep = self.nms(
            all_boxes,
            all_scores,
            self.nms_threshold
        )

        detections = []

        h, w = image.shape[:2]

        for i in keep:

            x1, y1, x2, y2 = (
                all_boxes[i]
            )

            # Clamp coordinates
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
                    float(all_scores[i])
                )
            )

        return detections
