from pathlib import Path

import torch

from app.ai.voice_model_v3 import (
    VoiceDistressModelV3,
)

from app.ai.emotion_utils import (
    extract_mfcc,
)


# ==========================================
# Model Path
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "trained_models"
    / "distress_last_v3.pth"
)


# ==========================================
# Voice Distress Detector
# ==========================================

class VoiceDistressDetector:

    def __init__(self):

        if not MODEL_PATH.exists():

            raise FileNotFoundError(
                f"Model not found:\n{MODEL_PATH}"
            )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"Loading Voice Distress Model on {self.device}"
        )

        self.model = VoiceDistressModelV3().to(
            self.device
        )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        print("\nCheckpoint Type :", type(checkpoint))

        if isinstance(checkpoint, dict):
            print("Checkpoint Keys :", checkpoint.keys())

        if isinstance(checkpoint, dict):

            if "model_state_dict" in checkpoint:

                self.model.load_state_dict(
                    checkpoint["model_state_dict"]
                )

            else:

                self.model.load_state_dict(
                    checkpoint
                )

        else:

            self.model.load_state_dict(
                checkpoint
            )

        self.model.eval()

        self.index_to_label = {
            0: "normal",
            1: "distress",
        }

    # ======================================
    # Prediction
    # ======================================

    def predict(
        self,
        audio_path,
    ):

        audio_path = Path(audio_path)

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found:\n{audio_path}"
            )

        features = extract_mfcc(
            str(audio_path)
        )

        features = (
            torch.from_numpy(features)
            .float()
            .unsqueeze(0)
            .to(self.device)
        )

        with torch.inference_mode():

            outputs = self.model(
                features
            )

            print("\n==============================")
            print("RAW LOGITS")
            print("==============================")
            print(outputs.cpu())

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            normal_probability = probabilities[0][0].item()
            distress_probability = probabilities[0][1].item()

            print("\n==============================")
            print("PROBABILITIES")
            print("==============================")

            print(
                f"Normal    : {normal_probability:.6f}"
            )

            print(
                f"Distress  : {distress_probability:.6f}"
            )

        # ======================================
        # Decision Threshold
        # ======================================

        THRESHOLD = 0.40

        if distress_probability >= THRESHOLD:

            prediction = 1
            confidence = distress_probability

        else:

            prediction = 0
            confidence = normal_probability

        label = self.index_to_label[
            prediction
        ]

        print("\n==============================")
        print("DECISION")
        print("==============================")

        print(
            f"Threshold : {THRESHOLD:.2f}"
        )

        print(
            f"Predicted : {label}"
        )

        return {

            "distress_detected":
                label == "distress",

            "label":
                label,

            "confidence":
                round(
                    confidence * 100,
                    2,
                ),

        }


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print("\n==============================")

    print("Voice Distress Detector V3")

    print("==============================")

    detector = VoiceDistressDetector()

    print("\nModel Loaded Successfully!")

    test_audio = input(
        "\nEnter WAV file path: "
    ).strip()

    result = detector.predict(
        test_audio
    )

    print("\nPrediction")

    print("------------------------------")

    for key, value in result.items():

        print(
            f"{key} : {value}"
        )