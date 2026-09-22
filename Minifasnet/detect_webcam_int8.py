
import os
import sys
import cv2
import time
import numpy as np
import tensorflow as tf


# ============================================================
# Project paths
# ============================================================

BASE_DIR = r"D:\MinifasnetV1"

MODEL_PATH = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_int8.tflite"
)

# ============================================================
# Import official crop implementation
# ============================================================

sys.path.insert(0, BASE_DIR)

from src.generate_patches import CropImage


# ============================================================
# RetinaFace Detector
# ============================================================

class Detection:

    def __init__(self):

        caffemodel = os.path.join(
            BASE_DIR,
            "models",
            "Widerface-RetinaFace.caffemodel"
        )

        deploy = os.path.join(
            BASE_DIR,
            "models",
            "deploy.prototxt"
        )

        print("Loading RetinaFace...")

        self.detector = cv2.dnn.readNetFromCaffe(
            deploy,
            caffemodel
        )

        self.detector_confidence = 0.6


    def get_bbox(self, img):

        height, width = img.shape[:2]

        aspect_ratio = width / height

        # Same logic as official MiniFASNet implementation
        if height * width >= 192 * 192:

            img = cv2.resize(
                img,
                (
                    int(
                        192 * np.sqrt(aspect_ratio)
                    ),
                    int(
                        192 / np.sqrt(aspect_ratio)
                    )
                ),
                interpolation=cv2.INTER_LINEAR
            )

        blob = cv2.dnn.blobFromImage(
            img,
            1,
            mean=(104, 117, 123)
        )

        self.detector.setInput(
            blob,
            "data"
        )

        out = self.detector.forward(
            "detection_out"
        ).squeeze()

        if out.ndim == 1:
            out = np.expand_dims(
                out,
                axis=0
            )

        max_conf_index = np.argmax(
            out[:, 2]
        )

        confidence = out[
            max_conf_index,
            2
        ]

        if confidence < self.detector_confidence:
            return None

        left = (
            out[max_conf_index, 3]
            * width
        )

        top = (
            out[max_conf_index, 4]
            * height
        )

        right = (
            out[max_conf_index, 5]
            * width
        )

        bottom = (
            out[max_conf_index, 6]
            * height
        )

        bbox = [
            int(left),
            int(top),
            int(right - left + 1),
            int(bottom - top + 1)
        ]

        return bbox


# ============================================================
# INT8 MiniFASNet
# ============================================================

class MiniFASNetINT8:

    def __init__(self, model_path):

        print("Loading INT8 TFLite model...")

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

        self.input_index = (
            self.input_details[0]["index"]
        )

        self.output_index = (
            self.output_details[0]["index"]
        )

        # Input quantization
        self.input_scale = (
            self.input_details[0]["quantization"][0]
        )

        self.input_zero_point = (
            self.input_details[0]["quantization"][1]
        )

        # Output quantization
        self.output_scale = (
            self.output_details[0]["quantization"][0]
        )

        self.output_zero_point = (
            self.output_details[0]["quantization"][1]
        )

        print("\nINT8 Model")
        print("--------------------------------")

        print(
            "Input:",
            self.input_details[0]["shape"],
            self.input_details[0]["dtype"]
        )

        print(
            "Input scale:",
            self.input_scale
        )

        print(
            "Input zero point:",
            self.input_zero_point
        )

        print(
            "Output:",
            self.output_details[0]["shape"],
            self.output_details[0]["dtype"]
        )

        print(
            "Output scale:",
            self.output_scale
        )

        print(
            "Output zero point:",
            self.output_zero_point
        )


    def predict(self, face):

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
        # Add batch
        # [80,80,3] -> [1,80,80,3]
        # ----------------------------------------------------

        face = np.expand_dims(
            face,
            axis=0
        )

        # ----------------------------------------------------
        # Quantize input
        # q = real / scale + zero_point
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
        # TFLite inference
        # ----------------------------------------------------

        self.interpreter.set_tensor(
            self.input_index,
            face_int8
        )

        self.interpreter.invoke()

        # ----------------------------------------------------
        # Get INT8 output
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

        return probabilities


# ============================================================
# Main
# ============================================================

def main():

    print("\n==========================================")
    print("MiniFASNetV2 INT8 WEBCAM")
    print("==========================================")

    # --------------------------------------------------------
    # Load detector
    # --------------------------------------------------------

    detector = Detection()

    # --------------------------------------------------------
    # Load INT8 model
    # --------------------------------------------------------

    model = MiniFASNetINT8(
        MODEL_PATH
    )

    # --------------------------------------------------------
    # Official cropper
    # --------------------------------------------------------

    cropper = CropImage()

    # MiniFASNet model scale
    scale = 2.7

    # --------------------------------------------------------
    # Open webcam
    # --------------------------------------------------------

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():

        print("\nERROR: Could not open webcam.")

        return

    # Optional camera resolution
    cap.set(
        cv2.CAP_PROP_FRAME_WIDTH,
        640
    )

    cap.set(
        cv2.CAP_PROP_FRAME_HEIGHT,
        480
    )

    print("\nWebcam started.")
    print("Press 'q' to quit.")

    # FPS variables
    prev_time = time.time()

    while True:

        ret, frame = cap.read()

        if not ret:

            print(
                "ERROR: Failed to read frame."
            )

            break

        # ----------------------------------------------------
        # Face detection
        # ----------------------------------------------------

        bbox = detector.get_bbox(
            frame
        )

        if bbox is not None:

            x, y, w, h = bbox

            # ------------------------------------------------
            # Official MiniFASNet crop
            # ------------------------------------------------

            face = cropper.crop(
                frame,
                bbox,
                scale,
                80,
                80
            )

            # ------------------------------------------------
            # INT8 prediction
            # ------------------------------------------------

            probabilities = model.predict(
                face
            )

            # ------------------------------------------------
            # Classes
            #
            # 0 = SPOOF
            # 1 = REAL
            # 2 = SPOOF
            # ------------------------------------------------

            class_0 = probabilities[0][0]
            class_1 = probabilities[0][1]
            class_2 = probabilities[0][2]

            predicted_class = int(
                np.argmax(
                    probabilities[0]
                )
            )

            # REAL = class 1
            # SPOOF = class 0 or 2

            real_probability = class_1

            spoof_probability = (
                class_0 + class_2
            )

            if predicted_class == 1:

                result = "REAL"

            else:

                result = "SPOOF"

            # ------------------------------------------------
            # Draw bounding box
            # ------------------------------------------------

            cv2.rectangle(
                frame,
                (x, y),
                (x + w, y + h),
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Display result
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
                    x,
                    max(y - 10, 30)
                ),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 255, 0),
                2
            )

            # ------------------------------------------------
            # Display class probabilities
            # ------------------------------------------------

            cv2.putText(
                frame,
                f"C0 Spoof: {class_0 * 100:.1f}%",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"C1 Real: {class_1 * 100:.1f}%",
                (10, 55),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

            cv2.putText(
                frame,
                f"C2 Spoof: {class_2 * 100:.1f}%",
                (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                2
            )

        else:

            cv2.putText(
                frame,
                "No face detected",
                (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.8,
                (0, 0, 255),
                2
            )

        # ----------------------------------------------------
        # FPS
        # ----------------------------------------------------

        current_time = time.time()

        fps = 1.0 / (
            current_time - prev_time
        )

        prev_time = current_time

        cv2.putText(
            frame,
            f"FPS: {fps:.2f}",
            (10, 115),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2
        )

        # ----------------------------------------------------
        # Show frame
        # ----------------------------------------------------

        cv2.imshow(
            "MiniFASNetV2 INT8 Anti-Spoofing",
            frame
        )

        # ----------------------------------------------------
        # Quit with Q
        # ----------------------------------------------------

        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):

            break

    # --------------------------------------------------------
    # Cleanup
    # --------------------------------------------------------

    cap.release()

    cv2.destroyAllWindows()

    print("\nWebcam stopped.")


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()
