import os
import csv
import cv2
import numpy as np

from src.face_detector import FaceDetector
from src.anti_spoofing import AntiSpoofing


# ============================================================
# CONFIGURATION
# ============================================================

DATASET_DIR = "test_images_organized"

SCRFD_MODEL = "models/scrfd_int8.tflite"
ANTISPOOF_MODEL = "models/MN3_antispoof.pth"

OUTPUT_DIR = "scrfd_evaluation_results"

IMAGE_EXTENSIONS = (
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp"
)


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

def create_output_dirs():

    folders = [
        "correct_live",
        "wrong_live",
        "correct_spoof",
        "wrong_spoof",
        "no_face"
    ]

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    for folder in folders:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                folder
            ),
            exist_ok=True
        )


# ============================================================
# DRAW RESULT
# ============================================================

def save_result_image(
    image,
    image_name,
    box,
    true_label,
    predicted_label,
    face_confidence,
    real_probability,
    spoof_probability,
    output_folder
):

    output = image.copy()

    x1, y1, x2, y2 = box

    # --------------------------------------------------------
    # Box color
    # --------------------------------------------------------

    if predicted_label == "LIVE":
        box_color = (0, 255, 0)
    else:
        box_color = (0, 0, 255)

    # --------------------------------------------------------
    # Draw face box
    # --------------------------------------------------------

    cv2.rectangle(
        output,
        (x1, y1),
        (x2, y2),
        box_color,
        2
    )

    # --------------------------------------------------------
    # Draw information
    # --------------------------------------------------------

    text1 = (
        f"True: {true_label} | "
        f"Pred: {predicted_label}"
    )

    text2 = (
        f"Real: {real_probability:.3f} "
        f"Spoof: {spoof_probability:.3f}"
    )

    text3 = (
        f"Face conf: {face_confidence:.3f}"
    )

    cv2.putText(
        output,
        text1,
        (10, 25),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        box_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        text2,
        (10, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2,
        cv2.LINE_AA
    )

    cv2.putText(
        output,
        text3,
        (10, 75),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        box_color,
        2,
        cv2.LINE_AA
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_path = os.path.join(
        OUTPUT_DIR,
        output_folder,
        image_name
    )

    cv2.imwrite(
        save_path,
        output
    )


# ============================================================
# SAVE NO-FACE IMAGE
# ============================================================

def save_no_face_image(
    image,
    image_name,
    true_label
):

    output = image.copy()

    cv2.putText(
        output,
        f"True: {true_label} | NO FACE",
        (10, 30),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (0, 0, 255),
        2,
        cv2.LINE_AA
    )

    save_path = os.path.join(
        OUTPUT_DIR,
        "no_face",
        image_name
    )

    cv2.imwrite(
        save_path,
        output
    )


# ============================================================
# MAIN
# ============================================================

def evaluate():

    create_output_dirs()

    print("=" * 60)
    print("SCRFD + MobileNetV3 ANTI-SPOOFING EVALUATION")
    print("=" * 60)

    # --------------------------------------------------------
    # Load SCRFD
    # --------------------------------------------------------

    print("\nLoading SCRFD...")

    detector = FaceDetector(
        SCRFD_MODEL
    )

    print("SCRFD loaded.")

    # --------------------------------------------------------
    # Load MobileNetV3 anti-spoof model
    # --------------------------------------------------------

    print("\nLoading MobileNetV3 anti-spoof model...")

    antispoof = AntiSpoofing(
        ANTISPOOF_MODEL,
        device="cpu"
    )

    print("Anti-spoofing model loaded.")

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    total_images = 0
    evaluated_images = 0
    no_face_count = 0

    correct_live = 0
    wrong_live = 0

    correct_spoof = 0
    wrong_spoof = 0

    y_true = []
    y_pred = []
    y_scores = []

    # --------------------------------------------------------
    # CSV
    # --------------------------------------------------------

    csv_path = os.path.join(
        OUTPUT_DIR,
        "results.csv"
    )

    csv_file = open(
        csv_path,
        "w",
        newline="",
        encoding="utf-8"
    )

    writer = csv.writer(csv_file)

    writer.writerow([
        "image",
        "true_label",
        "predicted_label",
        "class_id",
        "real_probability",
        "spoof_probability",
        "face_confidence",
        "x1",
        "y1",
        "x2",
        "y2",
        "result"
    ])

    # ========================================================
    # PROCESS LIVE AND SPOOF
    # ========================================================

    for class_name, true_class_id in [
        ("live", 0),
        ("spoof", 1)
    ]:

        class_dir = os.path.join(
            DATASET_DIR,
            class_name
        )

        if not os.path.exists(class_dir):

            print(
                f"\nDirectory not found: {class_dir}"
            )

            continue

        image_files = sorted([
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith(
                IMAGE_EXTENSIONS
            )
        ])

        print("\n" + "=" * 60)
        print(
            f"PROCESSING {class_name.upper()}"
        )
        print(
            f"Images: {len(image_files)}"
        )
        print("=" * 60)

        # ----------------------------------------------------
        # Process images
        # ----------------------------------------------------

        for index, image_name in enumerate(
            image_files,
            start=1
        ):

            total_images += 1

            print(
                f"[{index}/{len(image_files)}] "
                f"{image_name}"
            )

            image_path = os.path.join(
                class_dir,
                image_name
            )

            image = cv2.imread(
                image_path
            )

            if image is None:

                print(
                    "  Could not read image."
                )

                continue

            # =================================================
            # SCRFD FACE DETECTION
            # =================================================

            detections = detector.detect_faces(
                image
            )

            if detections is None or len(detections) == 0:

                print(
                    "  No face detected."
                )

                no_face_count += 1

                save_no_face_image(
                    image,
                    image_name,
                    class_name.upper()
                )

                writer.writerow([
                    image_name,
                    class_name.upper(),
                    "NO_FACE",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "",
                    "no_face"
                ])

                continue

            # =================================================
            # SELECT HIGHEST-CONFIDENCE FACE
            # Detection format:
            # (x1, y1, x2, y2, confidence)
            # =================================================

            best_detection = max(
                detections,
                key=lambda d: d[4]
            )

            x1 = int(best_detection[0])
            y1 = int(best_detection[1])
            x2 = int(best_detection[2])
            y2 = int(best_detection[3])

            face_confidence = float(
                best_detection[4]
            )

            # =================================================
            # VALIDATE BOX
            # =================================================

            h, w = image.shape[:2]

            x1 = max(
                0,
                min(x1, w - 1)
            )

            y1 = max(
                0,
                min(y1, h - 1)
            )

            x2 = max(
                0,
                min(x2, w)
            )

            y2 = max(
                0,
                min(y2, h)
            )

            if x2 <= x1 or y2 <= y1:

                print(
                    "  Invalid detection box."
                )

                continue

            # =================================================
            # CROP FACE
            # =================================================

            face = image[
                y1:y2,
                x1:x2
            ]

            if face.size == 0:

                print(
                    "  Empty face crop."
                )

                continue

            # =================================================
            # MOBILE NET V3 ANTI-SPOOF
            # =================================================

            result = antispoof.predict(
                face
            )

            predicted_class = result[
                "class_id"
            ]

            real_probability = result[
                "real_probability"
            ]

            spoof_probability = result[
                "spoof_probability"
            ]

            predicted_label = (
                "LIVE"
                if predicted_class == 0
                else "SPOOF"
            )

            true_label = class_name.upper()

            # =================================================
            # STORE METRICS
            # =================================================

            y_true.append(
                true_class_id
            )

            y_pred.append(
                predicted_class
            )

            y_scores.append(
                spoof_probability
            )

            evaluated_images += 1

            # =================================================
            # CORRECT / WRONG
            # =================================================

            is_correct = (
                predicted_class ==
                true_class_id
            )

            if true_class_id == 0:

                if is_correct:

                    correct_live += 1

                    output_folder = (
                        "correct_live"
                    )

                    result_type = "correct"

                else:

                    wrong_live += 1

                    output_folder = (
                        "wrong_live"
                    )

                    result_type = "wrong"

            else:

                if is_correct:

                    correct_spoof += 1

                    output_folder = (
                        "correct_spoof"
                    )

                    result_type = "correct"

                else:

                    wrong_spoof += 1

                    output_folder = (
                        "wrong_spoof"
                    )

                    result_type = "wrong"

            # =================================================
            # SAVE RESULT IMAGE
            # =================================================

            save_result_image(
                image=image,
                image_name=image_name,
                box=(
                    x1,
                    y1,
                    x2,
                    y2
                ),
                true_label=true_label,
                predicted_label=predicted_label,
                face_confidence=face_confidence,
                real_probability=real_probability,
                spoof_probability=spoof_probability,
                output_folder=output_folder
            )

            # =================================================
            # CSV
            # =================================================

            writer.writerow([
                image_name,
                true_label,
                predicted_label,
                predicted_class,
                f"{real_probability:.6f}",
                f"{spoof_probability:.6f}",
                f"{face_confidence:.6f}",
                x1,
                y1,
                x2,
                y2,
                result_type
            ])

    csv_file.close()

    # ========================================================
    # METRICS
    # ========================================================

    if len(y_true) == 0:

        print(
            "\nNo images were successfully evaluated."
        )

        return

    y_true_np = np.array(
        y_true
    )

    y_pred_np = np.array(
        y_pred
    )

    y_scores_np = np.array(
        y_scores
    )

    # --------------------------------------------------------
    # Confusion matrix
    # --------------------------------------------------------

    tn = np.sum(
        (y_true_np == 0) &
        (y_pred_np == 0)
    )

    fp = np.sum(
        (y_true_np == 0) &
        (y_pred_np == 1)
    )

    fn = np.sum(
        (y_true_np == 1) &
        (y_pred_np == 0)
    )

    tp = np.sum(
        (y_true_np == 1) &
        (y_pred_np == 1)
    )

    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = (
        (tp + tn) /
        (tp + tn + fp + fn)
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    if precision + recall > 0:

        f1 = (
            2 * precision * recall /
            (precision + recall)
        )

    else:

        f1 = 0

    # --------------------------------------------------------
    # ROC AUC
    # --------------------------------------------------------

    try:

        from sklearn.metrics import roc_auc_score

        roc_auc = roc_auc_score(
            y_true_np,
            y_scores_np
        )

    except Exception:

        roc_auc = float("nan")

    # --------------------------------------------------------
    # APCER / BPCER / ACER
    # --------------------------------------------------------

    apcer = (
        fp / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    bpcer = (
        fn / (fn + tp)
        if (fn + tp) > 0
        else 0
    )

    acer = (
        apcer + bpcer
    ) / 2

    live_accuracy = (
        tn / (tn + fp)
        if (tn + fp) > 0
        else 0
    )

    spoof_accuracy = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0
    )

    # ========================================================
    # PRINT RESULTS
    # ========================================================

    print("\n")
    print("=" * 60)
    print("SCRFD + MOBILE NET V3 RESULTS")
    print("=" * 60)

    print(
        f"\nTotal images              : "
        f"{total_images}"
    )

    print(
        f"Images successfully eval  : "
        f"{evaluated_images}"
    )

    print(
        f"No face detected          : "
        f"{no_face_count}"
    )

    print("\nOverall Metrics")
    print("-" * 40)

    print(
        f"Accuracy                : "
        f"{accuracy:.4f}"
    )

    print(
        f"Precision               : "
        f"{precision:.4f}"
    )

    print(
        f"Recall                  : "
        f"{recall:.4f}"
    )

    print(
        f"F1 Score                : "
        f"{f1:.4f}"
    )

    print(
        f"ROC-AUC                 : "
        f"{roc_auc:.4f}"
    )

    print("\nClass-wise Accuracy")
    print("-" * 40)

    print(
        f"LIVE accuracy           : "
        f"{live_accuracy:.4f}"
    )

    print(
        f"SPOOF accuracy          : "
        f"{spoof_accuracy:.4f}"
    )

    print("\nAnti-Spoofing Metrics")
    print("-" * 40)

    print(
        f"APCER                   : "
        f"{apcer:.4f}"
    )

    print(
        f"BPCER                   : "
        f"{bpcer:.4f}"
    )

    print(
        f"ACER                    : "
        f"{acer:.4f}"
    )

    print("\nConfusion Matrix")
    print("-" * 40)

    print(
        "                 Predicted"
    )

    print(
        "                 LIVE   SPOOF"
    )

    print(
        f"Actual LIVE     "
        f"{tn:5d}  {fp:5d}"
    )

    print(
        f"Actual SPOOF    "
        f"{fn:5d}  {tp:5d}"
    )

    print("\nSaved Results")
    print("-" * 40)

    print(
        f"Output directory: "
        f"{OUTPUT_DIR}"
    )

    print(
        f"CSV: {csv_path}"
    )

    print("\n" + "=" * 60)


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":
    evaluate()