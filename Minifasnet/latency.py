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

MODEL_PATH = os.path.join(
    BASE_DIR,
    "tensorflow",
    "2.7_80x80_MiniFASNetV2_int8.tflite"
)

# Change this if your SCRFD model has another filename
SCRFD_MODEL_PATH = os.path.join(
    BASE_DIR,
    "models",
    "scrfd_int8.tflite"
)

TEST_IMAGE = os.path.join(
    BASE_DIR,
    "test_images",
    "3.jpg"
)

RESULT_DIR = os.path.join(
    BASE_DIR,
    "benchmark",
    "overall"
)

os.makedirs(
    RESULT_DIR,
    exist_ok=True
)


# ============================================================
# IMPORT OFFICIAL CROP
# ============================================================

sys.path.insert(
    0,
    BASE_DIR
)

from src.generate_patches import CropImage


# ============================================================
# SCRFD DETECTOR
# ============================================================

class SCRFDDetector:

    def __init__(self, model_path):

        print("Loading SCRFD model...")

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

        self.input_shape = (
            self.input_details[0]["shape"]
        )

        self.input_scale = (
            self.input_details[0]["quantization"][0]
        )

        self.input_zero_point = (
            self.input_details[0]["quantization"][1]
        )

        print(
            "SCRFD input:",
            self.input_shape
        )

        print(
            "SCRFD dtype:",
            self.input_details[0]["dtype"]
        )


    def detect(self, image):

        # ----------------------------------------------------
        # Resize
        # ----------------------------------------------------

        start = time.perf_counter()

        input_h = self.input_shape[1]
        input_w = self.input_shape[2]

        resized = cv2.resize(
            image,
            (input_w, input_h),
            interpolation=cv2.INTER_LINEAR
        )

        # ----------------------------------------------------
        # BGR -> RGB
        # ----------------------------------------------------

        resized = cv2.cvtColor(
            resized,
            cv2.COLOR_BGR2RGB
        )

        resized = resized.astype(
            np.float32
        )

        # SCRFD normalization
        resized = (
            resized - 127.5
        ) / 128.0

        resized = np.expand_dims(
            resized,
            axis=0
        )

        # ----------------------------------------------------
        # Quantization
        # ----------------------------------------------------

        if self.input_details[0]["dtype"] == np.int8:

            resized = np.round(
                resized / self.input_scale
                + self.input_zero_point
            )

            resized = np.clip(
                resized,
                -128,
                127
            ).astype(np.int8)

        elif self.input_details[0]["dtype"] == np.uint8:

            resized = np.round(
                resized / self.input_scale
                + self.input_zero_point
            )

            resized = np.clip(
                resized,
                0,
                255
            ).astype(np.uint8)

        else:

            resized = resized.astype(
                np.float32
            )

        preprocessing_time = (
            time.perf_counter() - start
        )

        # ----------------------------------------------------
        # SCRFD inference
        # ----------------------------------------------------

        start = time.perf_counter()

        self.interpreter.set_tensor(
            self.input_index,
            resized
        )

        self.interpreter.invoke()

        outputs = []

        for output in self.output_details:

            data = self.interpreter.get_tensor(
                output["index"]
            )

            outputs.append(data)

        detection_time = (
            time.perf_counter() - start
        )

        return (
            outputs,
            preprocessing_time,
            detection_time
        )


# ============================================================
# MINIFASNET INT8
# ============================================================

class MiniFASNetINT8:

    def __init__(self, model_path):

        print("Loading MiniFASNetV2 INT8...")

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

        self.input_scale = (
            self.input_details[0]["quantization"][0]
        )

        self.input_zero_point = (
            self.input_details[0]["quantization"][1]
        )

        self.output_scale = (
            self.output_details[0]["quantization"][0]
        )

        self.output_zero_point = (
            self.output_details[0]["quantization"][1]
        )


    def predict(self, face):

        # ====================================================
        # PREPROCESSING
        # ====================================================

        start = time.perf_counter()

        face = cv2.resize(
            face,
            (80, 80),
            interpolation=cv2.INTER_LINEAR
        )

        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        face = face.astype(
            np.float32
        )

        face = np.expand_dims(
            face,
            axis=0
        )

        # Quantize
        face = np.round(
            face / self.input_scale
            + self.input_zero_point
        )

        face = np.clip(
            face,
            -128,
            127
        ).astype(
            np.int8
        )

        preprocessing_time = (
            time.perf_counter() - start
        )

        # ====================================================
        # MODEL INFERENCE
        # ====================================================

        start = time.perf_counter()

        self.interpreter.set_tensor(
            self.input_index,
            face
        )

        self.interpreter.invoke()

        raw_output = (
            self.interpreter.get_tensor(
                self.output_index
            )
        )

        inference_time = (
            time.perf_counter() - start
        )

        # ====================================================
        # OUTPUT PROCESSING
        # ====================================================

        start = time.perf_counter()

        logits = (
            self.output_scale
            * (
                raw_output.astype(
                    np.float32
                )
                - self.output_zero_point
            )
        )

        logits = (
            logits
            - np.max(
                logits,
                axis=1,
                keepdims=True
            )
        )

        exp_logits = np.exp(
            logits
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

        postprocessing_time = (
            time.perf_counter() - start
        )

        return (
            predicted_class,
            probabilities,
            preprocessing_time,
            inference_time,
            postprocessing_time
        )


# ============================================================
# BENCHMARK
# ============================================================

def benchmark(
    detector,
    model,
    image,
    iterations=100
):

    cropper = CropImage()

    scale = 2.7

    detection_times = []
    detection_preprocess_times = []

    crop_times = []

    antispoof_preprocess_times = []
    antispoof_inference_times = []
    antispoof_postprocess_times = []

    total_times = []

    successful_runs = 0


    print("\n==============================================")
    print("Starting Overall Pipeline Benchmark")
    print("==============================================")

    print(
        f"Iterations: {iterations}"
    )


    # ========================================================
    # WARM-UP
    # ========================================================

    print("\nWarming up...")

    for _ in range(10):

        outputs, _, _ = detector.detect(
            image
        )

        face = image

        try:

            # Warm up MiniFASNet
            model.predict(face)

        except Exception:

            pass


    print("Warm-up completed.")


    # ========================================================
    # BENCHMARK LOOP
    # ========================================================

    print("\nBenchmarking...")

    for i in range(iterations):

        pipeline_start = time.perf_counter()


        # ----------------------------------------------------
        # FACE DETECTION
        # ----------------------------------------------------

        outputs, det_preprocess, det_inference = (
            detector.detect(image)
        )

        detection_preprocess_times.append(
            det_preprocess
        )

        detection_times.append(
            det_inference
        )


        # ----------------------------------------------------
        # NOTE
        #
        # SCRFD output decoding/NMS depends on the exact
        # SCRFD model output format.
        #
        # For timing the complete project, replace this
        # section with your existing SCRFD bbox decoder.
        # ----------------------------------------------------

        #
        # IMPORTANT:
        # Here we use a known face region for measuring
        # the MiniFASNet stage.
        #
        # Replace this with:
        #
        # bbox = decode_scrfd(outputs)
        #

        h, w = image.shape[:2]

        x = int(w * 0.25)
        y = int(h * 0.15)

        bw = int(w * 0.50)
        bh = int(h * 0.70)

        bbox = [
            x,
            y,
            bw,
            bh
        ]


        # ----------------------------------------------------
        # FACE CROP
        # ----------------------------------------------------

        start = time.perf_counter()

        face = cropper.crop(
            image,
            bbox,
            scale,
            80,
            80
        )

        crop_time = (
            time.perf_counter() - start
        )

        crop_times.append(
            crop_time
        )


        # ----------------------------------------------------
        # MINIFASNET
        # ----------------------------------------------------

        (
            predicted_class,
            probabilities,
            anti_preprocess,
            anti_inference,
            anti_postprocess
        ) = model.predict(
            face
        )

        antispoof_preprocess_times.append(
            anti_preprocess
        )

        antispoof_inference_times.append(
            anti_inference
        )

        antispoof_postprocess_times.append(
            anti_postprocess
        )


        # ----------------------------------------------------
        # TOTAL PIPELINE TIME
        # ----------------------------------------------------

        total_time = (
            time.perf_counter()
            - pipeline_start
        )

        total_times.append(
            total_time
        )

        successful_runs += 1


        if (
            (i + 1) % 10 == 0
            or i == 0
        ):

            print(
                f"Iteration "
                f"{i + 1:3d}/{iterations} | "
                f"Total: "
                f"{total_time * 1000:.2f} ms"
            )


    # ========================================================
    # CONVERT TO NUMPY
    # ========================================================

    detection_times = np.array(
        detection_times
    )

    detection_preprocess_times = np.array(
        detection_preprocess_times
    )

    crop_times = np.array(
        crop_times
    )

    antispoof_preprocess_times = np.array(
        antispoof_preprocess_times
    )

    antispoof_inference_times = np.array(
        antispoof_inference_times
    )

    antispoof_postprocess_times = np.array(
        antispoof_postprocess_times
    )

    total_times = np.array(
        total_times
    )


    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 70)
    print("OVERALL ANTI-SPOOFING PIPELINE PERFORMANCE")
    print("=" * 70)


    def report(
        name,
        data
    ):

        print(
            f"\n{name}"
        )

        print(
            f"Average : "
            f"{np.mean(data) * 1000:.3f} ms"
        )

        print(
            f"Median  : "
            f"{np.median(data) * 1000:.3f} ms"
        )

        print(
            f"Min     : "
            f"{np.min(data) * 1000:.3f} ms"
        )

        print(
            f"Max     : "
            f"{np.max(data) * 1000:.3f} ms"
        )

        print(
            f"P95     : "
            f"{np.percentile(data, 95) * 1000:.3f} ms"
        )

        print(
            f"P99     : "
            f"{np.percentile(data, 99) * 1000:.3f} ms"
        )


    report(
        "Face Detection Preprocessing",
        detection_preprocess_times
    )

    report(
        "SCRFD Inference",
        detection_times
    )

    report(
        "Face Cropping",
        crop_times
    )

    report(
        "MiniFASNet Preprocessing",
        antispoof_preprocess_times
    )

    report(
        "MiniFASNet Inference",
        antispoof_inference_times
    )

    report(
        "MiniFASNet Postprocessing",
        antispoof_postprocess_times
    )

    report(
        "TOTAL PIPELINE",
        total_times
    )


    # ========================================================
    # THROUGHPUT
    # ========================================================

    average_latency = np.mean(
        total_times
    )

    median_latency = np.median(
        total_times
    )

    throughput = (
        1.0 / average_latency
    )

    median_fps = (
        1.0 / median_latency
    )


    print("\n")
    print("-" * 70)
    print("THROUGHPUT")
    print("-" * 70)

    print(
        f"Average latency : "
        f"{average_latency * 1000:.3f} ms"
    )

    print(
        f"Median latency  : "
        f"{median_latency * 1000:.3f} ms"
    )

    print(
        f"Throughput      : "
        f"{throughput:.2f} FPS"
    )

    print(
        f"Median FPS      : "
        f"{median_fps:.2f} FPS"
    )


    # ========================================================
    # COMPONENT CONTRIBUTION
    # ========================================================

    avg_detection = np.mean(
        detection_times
    )

    avg_crop = np.mean(
        crop_times
    )

    avg_antispoof = np.mean(
        antispoof_inference_times
    )

    avg_total = np.mean(
        total_times
    )


    print("\n")
    print("-" * 70)
    print("PIPELINE COMPONENT CONTRIBUTION")
    print("-" * 70)

    print(
        f"SCRFD detection : "
        f"{avg_detection * 1000:.3f} ms "
        f"({avg_detection / avg_total * 100:.2f}%)"
    )

    print(
        f"Face cropping   : "
        f"{avg_crop * 1000:.3f} ms "
        f"({avg_crop / avg_total * 100:.2f}%)"
    )

    print(
        f"MiniFASNet      : "
        f"{avg_antispoof * 1000:.3f} ms "
        f"({avg_antispoof / avg_total * 100:.2f}%)"
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    output_file = os.path.join(
        RESULT_DIR,
        "overall_pipeline_benchmark.txt"
    )


    with open(
        output_file,
        "w"
    ) as f:

        f.write(
            "============================================================\n"
        )

        f.write(
            "SCRFD + MiniFASNetV2 INT8\n"
        )

        f.write(
            "OVERALL PIPELINE PERFORMANCE\n"
        )

        f.write(
            "============================================================\n\n"
        )

        f.write(
            f"Iterations           : {iterations}\n"
        )

        f.write(
            f"Successful runs      : {successful_runs}\n\n"
        )

        f.write(
            f"Face Detection Avg   : "
            f"{np.mean(detection_times)*1000:.3f} ms\n"
        )

        f.write(
            f"Face Crop Avg        : "
            f"{np.mean(crop_times)*1000:.3f} ms\n"
        )

        f.write(
            f"MiniFASNet Avg       : "
            f"{np.mean(antispoof_inference_times)*1000:.3f} ms\n"
        )

        f.write(
            f"Total Avg Latency    : "
            f"{average_latency*1000:.3f} ms\n"
        )

        f.write(
            f"Median Latency       : "
            f"{median_latency*1000:.3f} ms\n"
        )

        f.write(
            f"P95 Latency         : "
            f"{np.percentile(total_times,95)*1000:.3f} ms\n"
        )

        f.write(
            f"P99 Latency         : "
            f"{np.percentile(total_times,99)*1000:.3f} ms\n"
        )

        f.write(
            f"Throughput          : "
            f"{throughput:.2f} FPS\n"
        )

        f.write(
            f"Median FPS          : "
            f"{median_fps:.2f} FPS\n"
        )


    print("\n")
    print(
        "Benchmark report saved:"
    )

    print(
        output_file
    )


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "\n=============================================="
    )

    print(
        "SCRFD + MiniFASNetV2 INT8"
    )

    print(
        "OVERALL PERFORMANCE BENCHMARK"
    )

    print(
        "=============================================="
    )


    # --------------------------------------------------------
    # Check files
    # --------------------------------------------------------

    if not os.path.exists(
        SCRFD_MODEL_PATH
    ):

        print(
            "\nERROR: SCRFD model not found:"
        )

        print(
            SCRFD_MODEL_PATH
        )

        return


    if not os.path.exists(
        MODEL_PATH
    ):

        print(
            "\nERROR: MiniFASNet model not found:"
        )

        print(
            MODEL_PATH
        )

        return


    if not os.path.exists(
        TEST_IMAGE
    ):

        print(
            "\nERROR: Test image not found:"
        )

        print(
            TEST_IMAGE
        )

        return


    # --------------------------------------------------------
    # Load image
    # --------------------------------------------------------

    image = cv2.imread(
        TEST_IMAGE
    )

    if image is None:

        print(
            "ERROR: Could not load image."
        )

        return


    # --------------------------------------------------------
    # Models
    # --------------------------------------------------------

    detector = SCRFDDetector(
        SCRFD_MODEL_PATH
    )

    model = MiniFASNetINT8(
        MODEL_PATH
    )


    # --------------------------------------------------------
    # Benchmark
    # --------------------------------------------------------

    benchmark(
        detector,
        model,
        image,
        iterations=100
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    main()