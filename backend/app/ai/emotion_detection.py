from pathlib import Path

import librosa
import torch

from transformers import (
    AutoModelForAudioClassification,
    AutoProcessor,
)


# ==========================================
# Configuration
# ==========================================

MODEL_NAME = (
    "Dpngtm/wav2vec2-emotion-recognition"
)

SAMPLE_RATE = 16000

BASE_DIR = (
    Path(__file__).resolve().parents[2]
)


# ==========================================
# Emotion Detector
# ==========================================

class EmotionDetector:

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        print(
            f"\nLoading pretrained Emotion Model on "
            f"{self.device}"
        )

        # ----------------------------------
        # Processor
        # ----------------------------------

        self.processor = (
            AutoProcessor.from_pretrained(
                MODEL_NAME
            )
        )

        # ----------------------------------
        # Pretrained Model
        # ----------------------------------

        self.model = (
            AutoModelForAudioClassification
            .from_pretrained(
                MODEL_NAME
            )
            .to(self.device)
        )

        self.model.eval()

        # ----------------------------------
        # Read labels directly from model
        # ----------------------------------

        self.id_to_label = (
            self.model.config.id2label
        )

        print(
            "\nEmotion Label Mapping:"
        )

        for index, label in (
            self.id_to_label.items()
        ):
            print(
                f"  {index} -> {label}"
            )

        print(
            "\nPretrained Emotion Model "
            "Loaded Successfully!"
        )


    # ======================================
    # Prediction
    # ======================================

    def predict(
        self,
        audio_path,
    ):

        audio_path = Path(
            audio_path
        )

        # ----------------------------------
        # Check audio
        # ----------------------------------

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found:\n"
                f"{audio_path}"
            )

        # ----------------------------------
        # Load audio
        # ----------------------------------

        audio, sample_rate = (
            librosa.load(
                str(audio_path),
                sr=SAMPLE_RATE,
                mono=True,
            )
        )

        if len(audio) == 0:

            raise ValueError(
                "Audio file is empty."
            )

        # ----------------------------------
        # Processor
        # ----------------------------------

        inputs = self.processor(
            audio,
            sampling_rate=SAMPLE_RATE,
            return_tensors="pt",
            padding=True,
        )

        # ----------------------------------
        # Move tensors to device
        # ----------------------------------

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }

        # ----------------------------------
        # Prediction
        # ----------------------------------

        with torch.inference_mode():

            outputs = self.model(
                **inputs
            )

            probabilities = (
                torch.softmax(
                    outputs.logits,
                    dim=-1,
                )
            )

            prediction = (
                torch.argmax(
                    probabilities,
                    dim=-1,
                ).item()
            )

            confidence = (
                probabilities[
                    0,
                    prediction,
                ].item()
            )

        # ----------------------------------
        # Label
        # ----------------------------------

        emotion = self.id_to_label[
            prediction
        ]

        return {
            "emotion": emotion,
            "confidence": round(
                confidence * 100,
                2,
            ),
        }


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print(
        "\n================================"
    )

    print(
        "GuardAI Pretrained Emotion Detector"
    )

    print(
        "================================"
    )

    detector = EmotionDetector()

    print(
        "\nMODEL OK"
    )

    test_audio = input(
        "\nEnter WAV file path: "
    ).strip()

    result = detector.predict(
        test_audio
    )

    print(
        "\nPrediction"
    )

    print(
        "------------------------------"
    )

    print(
        "Emotion    :",
        result["emotion"],
    )

    print(
        "Confidence :",
        f'{result["confidence"]}%',
    )
