
import os
import sys
import cv2
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

IMAGE_PATH = os.path.join(
    BASE_DIR,
    "test_images",
    "real.png"       # Change this to your image
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "result"
)

CROP_DIR = os.path.join(
    BASE_DIR,
    "cropped_faces"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CROP_DIR, exist_ok=True)


# ============================================================
# Import official MiniFASNet crop implementation
# ============================================================

sys.path.insert(
    0,
    BASE_DIR
)

from src.generate_patches import CropImage


# ============================================================
# RetinaFace detector
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

        print("Loading RetinaFace detector...")

        self.detector = cv2.dnn.readNetFromCaffe(
            deploy,
            caffemodel
        )

        self.detector_confidence = 0.6


    def get_bbox(self, img):

        height, width = img.shape[:2]

        aspect_ratio = width / height

        # Same resizing logic as the official implementation
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

        # Handle case where only one detection exists
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

        left = out[
            max_conf_index,
            3
        ] * width

        top = out[
            max_conf_index,
            4
        ] * height

        right = out[
            max_conf_index,
            5
        ] * width

        bottom = out[
            max_conf_index,
            6
        ] * height

        bbox = [
            int(left),
            int(top),
            int(right - left + 1),
            int(bottom - top + 1)
        ]

        return bbox


# ============================================================
# INT8 MiniFASNet inference
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

        print("\nINT8 Model Information")
        print("--------------------------------")

        print(
            "Input shape:",
            self.input_details[0]["shape"]
        )

        print(
            "Input dtype:",
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
            "Output shape:",
            self.output_details[0]["shape"]
        )

        print(
            "Output dtype:",
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
        # Ensure 80 x 80
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
        # Convert to float32
        #
        # Original MiniFASNet preprocessing uses
        # pixel values in the 0-255 range.
        # ----------------------------------------------------

        face = face.astype(
            np.float32
        )


        # ----------------------------------------------------
        # Add batch dimension
        #
        # [80,80,3]
        #      ↓
        # [1,80,80,3]
        # ----------------------------------------------------

        face = np.expand_dims(
            face,
            axis=0
        )


        # ----------------------------------------------------
        # Quantize input
        #
        # q = real / scale + zero_point
        # ----------------------------------------------------

        quantized_face = np.round(
            face / self.input_scale
            + self.input_zero_point
        )

        quantized_face = np.clip(
            quantized_face,
            -128,
            127
        ).astype(
            np.int8
        )


        # ----------------------------------------------------
        # Run INT8 inference
        # ----------------------------------------------------

        self.interpreter.set_tensor(
            self.input_index,
            quantized_face
        )

        self.interpreter.invoke()


        # ----------------------------------------------------
        # Get raw INT8 output
        # ----------------------------------------------------

        raw_output = (
            self.interpreter.get_tensor(
                self.output_index
            )
        )


        # ----------------------------------------------------
        # Dequantize output
        #
        # real =
        # scale * (quantized - zero_point)
        # ----------------------------------------------------

        output = (
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

        output_shifted = (
            output
            - np.max(
                output,
                axis=1,
                keepdims=True
            )
        )

        exp_output = np.exp(
            output_shifted
        )

        probabilities = (
            exp_output
            / np.sum(
                exp_output,
                axis=1,
                keepdims=True
            )
        )


        return (
            raw_output,
            output,
            probabilities
        )


# ============================================================
# Main
# ============================================================

def main():

    print("\n==========================================")
    print("MiniFASNetV2 INT8 IMAGE DETECTION")
    print("==========================================")

    # --------------------------------------------------------
    # Check image
    # --------------------------------------------------------

    if not os.path.exists(IMAGE_PATH):

        print("\nERROR:")
        print("Image not found:")
        print(IMAGE_PATH)

        print(
            "\nChange IMAGE_PATH in the script."
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


    # --------------------------------------------------------
    # Create detector
    # --------------------------------------------------------

    detector = Detection()


    # --------------------------------------------------------
    # Create INT8 model
    # --------------------------------------------------------

    model = MiniFASNetINT8(
        MODEL_PATH
    )


    # --------------------------------------------------------
    # Detect face
    # --------------------------------------------------------

    bbox = detector.get_bbox(
        image
    )

    if bbox is None:

        print(
            "\nNo face detected."
        )

        return


    print("\nFace detected:")
    print("Bounding box:", bbox)


    # --------------------------------------------------------
    # Official MiniFASNet crop
    # --------------------------------------------------------

    cropper = CropImage()

    scale = 2.7

    face = cropper.crop(
        image,
        bbox,
        scale,
        80,
        80
    )


    # --------------------------------------------------------
    # Save face crop
    # --------------------------------------------------------

    crop_path = os.path.join(
        CROP_DIR,
        "int8_face_crop.jpg"
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
    # INT8 prediction
    # --------------------------------------------------------

    raw_output, logits, probabilities = (
        model.predict(face)
    )


    # --------------------------------------------------------
    # Get predicted class
    # --------------------------------------------------------

    label = int(
        np.argmax(
            probabilities[0]
        )
    )

    # MiniFASNet:
    #
    # class 1 = REAL
    # class 0/2 = SPOOF

    if label == 1:

        result = "REAL"

    else:

        result = "SPOOF"


    # --------------------------------------------------------
    # Probabilities
    # --------------------------------------------------------

    spoof_probability = (
        probabilities[0][0]
        + probabilities[0][2]
    )

    real_probability = (
        probabilities[0][1]
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print("\n==========================================")
    print("INT8 PREDICTION")
    print("==========================================")

    print("\nRaw INT8 output:")
    print(raw_output)

    print("\nDequantized logits:")
    print(logits)

    print("\nProbabilities:")
    print(probabilities)

    print("\nClass 0 SPOOF:")
    print(
        f"{probabilities[0][0] * 100:.2f}%"
    )

    print("\nClass 1 REAL:")
    print(
        f"{probabilities[0][1] * 100:.2f}%"
    )

    print("\nClass 2 SPOOF:")
    print(
        f"{probabilities[0][2] * 100:.2f}%"
    )

    print("\nREAL probability:")
    print(
        f"{real_probability * 100:.2f}%"
    )

    print("\nSPOOF probability:")
    print(
        f"{spoof_probability * 100:.2f}%"
    )

    print("\nPredicted class:")
    print(label)

    print("\nFinal result:")
    print(result)


    # --------------------------------------------------------
    # Draw result
    # --------------------------------------------------------

    x, y, w, h = bbox

    if result == "REAL":

        display_text = (
            f"REAL "
            f"{real_probability * 100:.1f}%"
        )

    else:

        display_text = (
            f"SPOOF "
            f"{spoof_probability * 100:.1f}%"
        )


    cv2.rectangle(
        image,
        (x, y),
        (x + w, y + h),
        (0, 255, 0),
        2
    )

    cv2.putText(
        image,
        display_text,
        (x, max(y - 10, 30)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 255, 0),
        2
    )


    # --------------------------------------------------------
    # Save result
    # --------------------------------------------------------

    image_name = os.path.splitext(
    os.path.basename(IMAGE_PATH)
)[0]

    image_ext = os.path.splitext(
    IMAGE_PATH
)[1]

    output_path = os.path.join(
    OUTPUT_DIR,
    f"{image_name}_int8{image_ext}"
)

    cv2.imwrite(
        output_path,
        image
    )

    print(
        "\nAnnotated result saved:"
    )

    print(
        output_path
    )


# ============================================================
# Run
# ============================================================

if __name__ == "__main__":
    main()

