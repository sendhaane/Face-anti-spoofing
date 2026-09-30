
import cv2
import torch
import torch.nn.functional as F
import numpy as np
import os

from src.mobilenetv3 import mobilenetv3_large


class AntiSpoofing:

    def __init__(self, model_path, device="cpu"):

        self.device = torch.device(device)

        # ---------------------------------------------------------
        # Build MobileNetV3
        # ---------------------------------------------------------
        self.model = mobilenetv3_large(
            width_mult=1.0,
            prob_dropout=0.1,
            type_dropout="bernoulli",
            prob_dropout_linear=0.35,
            embeding_dim=1280,
            mu=0.5,
            sigma=0.3,
            theta=0,
            multi_heads=True
        )

        # ---------------------------------------------------------
        # Load checkpoint
        # ---------------------------------------------------------
        checkpoint = torch.load(
            model_path,
            map_location=self.device,
            weights_only=True
        )

        state_dict = checkpoint["state_dict"]

        self.model.load_state_dict(
            state_dict,
            strict=True
        )

        self.model.to(self.device)
        self.model.eval()

        # ---------------------------------------------------------
        # Original preprocessing
        # ---------------------------------------------------------
        self.mean = np.array(
            [0.5931, 0.4690, 0.4229],
            dtype=np.float32
        )

        self.std = np.array(
            [0.2471, 0.2214, 0.2157],
            dtype=np.float32
        )

        self.input_size = (128, 128)

        print("Anti-spoofing model loaded successfully.")
        print(f"Device: {self.device}")

    def preprocess(self, face):

        # BGR -> RGB
        face = cv2.cvtColor(
            face,
            cv2.COLOR_BGR2RGB
        )

        # Resize
        face = cv2.resize(
            face,
            self.input_size,
            interpolation=cv2.INTER_CUBIC
        )

        # Convert to [0, 1]
        face = face.astype(np.float32) / 255.0

        # Normalize
        face = (face - self.mean) / self.std

        # HWC -> CHW
        face = np.transpose(
            face,
            (2, 0, 1)
        )

        # Add batch dimension
        face = np.expand_dims(
            face,
            axis=0
        )

        tensor = torch.from_numpy(face).float()

        return tensor.to(self.device)

    def predict(self, face):

        input_tensor = self.preprocess(face)

        with torch.no_grad():

            features = self.model(input_tensor)

            output = self.model.make_logits(
                features,
                all=False
            )

            if isinstance(output, tuple):
                output = output[0]

            probabilities = F.softmax(
                output,
                dim=1
            )

            predicted_class = torch.argmax(
                probabilities,
                dim=1
            ).item()

            real_probability = probabilities[0, 0].item()
            spoof_probability = probabilities[0, 1].item()

        if predicted_class == 0:
            label = "REAL"
        else:
            label = "SPOOF"

        return {
            "label": label,
            "class_id": predicted_class,
            "real_probability": real_probability,
            "spoof_probability": spoof_probability
        }


def main():

    MODEL_PATH = "models/MN3_antispoof.pth"

    antispoof = AntiSpoofing(
        model_path=MODEL_PATH,
        device="cpu"
    )

    # ---------------------------------------------------------
    # Open webcam
    # ---------------------------------------------------------
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam.")

    print("\nWebcam started.")
    print("Press Q to quit.")

    while True:

        ret, frame = cap.read()

        if not ret:
            print("Failed to read frame.")
            break


        result = antispoof.predict(frame)

        label = result["label"]
        real_prob = result["real_probability"]
        spoof_prob = result["spoof_probability"]

        # -----------------------------------------------------
        # Display result
        # -----------------------------------------------------

        text = (
            f"{label} | "
            f"Real: {real_prob:.2f} | "
            f"Spoof: {spoof_prob:.2f}"
        )

        cv2.putText(
            frame,
            text,
            (20, 40),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0) if label == "REAL" else (0, 0, 255),
            2
        )

        cv2.imshow(
            "MobileNetV3 Anti-Spoofing",
            frame
        )

        # -----------------------------------------------------
        # Quit
        # -----------------------------------------------------
        key = cv2.waitKey(1) & 0xFF

        if key == ord("q"):
            break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()

