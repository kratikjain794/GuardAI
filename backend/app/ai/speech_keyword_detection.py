from pathlib import Path

import torch
from transformers import pipeline


# ==========================================
# Model Configuration
# ==========================================

MODEL_NAME = "openai/whisper-tiny"

BASE_DIR = Path(__file__).resolve().parents[2]


# ==========================================
# Emergency Keywords
# ==========================================

EMERGENCY_KEYWORDS = [
    "help",
    "help me",
    "save me",
    "emergency",
    "bachao",
    "bachaao",
    "mujhe bachao",
    "madad",
    "madad karo",
    "police",
]


# ==========================================
# Speech Keyword Detector
# ==========================================

class SpeechKeywordDetector:

    _pipeline = None

    def __init__(self):

        # ----------------------------------
        # Load Whisper only once
        # ----------------------------------

        if SpeechKeywordDetector._pipeline is None:

            device = (
                0
                if torch.cuda.is_available()
                else -1
            )

            print(
                "\nLoading Whisper Tiny Speech "
                f"Model on "
                f"{'CUDA' if device == 0 else 'CPU'}..."
            )

            SpeechKeywordDetector._pipeline = pipeline(
                "automatic-speech-recognition",
                model=MODEL_NAME,
                device=device,
            )

            print(
                "Whisper Tiny Loaded Successfully!"
            )

        self.transcriber = (
            SpeechKeywordDetector._pipeline
        )

    # ======================================
    # Keyword Matching
    # ======================================

    @staticmethod
    def find_keyword(text):

        normalized = (
            str(text or "")
            .lower()
            .replace(".", " ")
            .replace(",", " ")
            .replace("!", " ")
            .replace("?", " ")
            .replace(";", " ")
            .replace(":", " ")
        )

        normalized = " ".join(
            normalized.split()
        )

        # Longer phrases first
        # so "help me" is preferred
        # over "help".

        keywords = sorted(
            EMERGENCY_KEYWORDS,
            key=len,
            reverse=True,
        )

        for keyword in keywords:

            if keyword in normalized:

                return keyword

        return None

    # ======================================
    # Prediction
    # ======================================

    def predict(self, audio_path):

        audio_path = Path(
            audio_path
        )

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found:\n"
                f"{audio_path}"
            )

        # ----------------------------------
        # Whisper transcription
        # ----------------------------------

        result = self.transcriber(
            str(audio_path),
            generate_kwargs={
                "language": "en",
                "task": "transcribe",
            },
        )

        transcript = str(
            result.get("text", "")
        ).strip()

        # ----------------------------------
        # Emergency keyword
        # ----------------------------------

        keyword = self.find_keyword(
            transcript
        )

        emergency_detected = (
            keyword is not None
        )

        return {

            "emergency_detected":
                emergency_detected,

            "keyword":
                keyword,

            "transcript":
                transcript,

        }


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print(
        "\n================================"
    )

    print(
        "GuardAI Speech Keyword Detector"
    )

    print(
        "================================"
    )

    detector = (
        SpeechKeywordDetector()
    )

    print(
        "\nModel Loaded Successfully!"
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

    for key, value in result.items():

        print(
            f"{key} : {value}"
        )