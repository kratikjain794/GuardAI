from pathlib import Path

import numpy as np
import soundfile as sf
import torch

from transformers import (
    WhisperProcessor,
    WhisperForConditionalGeneration,
)


# ==========================================
# Model Configuration
# ==========================================

MODEL_NAME = "openai/whisper-tiny"

TARGET_SAMPLE_RATE = 16000


# ==========================================
# Emergency Keywords
# ==========================================

EMERGENCY_KEYWORDS = [
    "help me",
    "save me",
    "emergency",
    "mujhe bachao",
    "bachaao",
    "bachao",
    "madad karo",
    "madad",
    "police",
    "help",
]


# ==========================================
# Speech Keyword Detector
# ==========================================

class SpeechKeywordDetector:

    _processor = None
    _model = None
    _device = None

    def __init__(self):

        # --------------------------------------
        # Load Whisper only once
        # --------------------------------------

        if (
            SpeechKeywordDetector._processor
            is None
        ):

            self.device = torch.device(
                "cuda"
                if torch.cuda.is_available()
                else "cpu"
            )

            print(
                "\nLoading Whisper Tiny Speech "
                f"Model on {self.device}..."
            )

            # ----------------------------------
            # Processor
            # ----------------------------------

            SpeechKeywordDetector._processor = (
                WhisperProcessor.from_pretrained(
                    MODEL_NAME
                )
            )

            # ----------------------------------
            # Model
            # ----------------------------------

            SpeechKeywordDetector._model = (
                WhisperForConditionalGeneration
                .from_pretrained(
                    MODEL_NAME
                )
                .to(self.device)
            )

            SpeechKeywordDetector._model.eval()

            SpeechKeywordDetector._device = (
                self.device
            )

            print(
                "Whisper Tiny Loaded Successfully!"
            )

        self.processor = (
            SpeechKeywordDetector._processor
        )

        self.model = (
            SpeechKeywordDetector._model
        )

        self.device = (
            SpeechKeywordDetector._device
        )


    # ==========================================
    # Audio Resampling
    # ==========================================

    @staticmethod
    def resample_audio(
        audio,
        original_sample_rate,
        target_sample_rate,
    ):

        if (
            original_sample_rate
            == target_sample_rate
        ):
            return audio

        if len(audio) == 0:
            return audio

        duration = (
            len(audio)
            / original_sample_rate
        )

        new_length = max(
            1,
            int(
                duration
                * target_sample_rate
            ),
        )

        old_indices = np.linspace(
            0,
            len(audio) - 1,
            num=len(audio),
        )

        new_indices = np.linspace(
            0,
            len(audio) - 1,
            num=new_length,
        )

        resampled_audio = np.interp(
            new_indices,
            old_indices,
            audio,
        )

        return resampled_audio.astype(
            np.float32
        )


    # ==========================================
    # Load Audio
    # ==========================================

    @staticmethod
    def load_audio(audio_path):

        try:

            audio, sample_rate = sf.read(
                str(audio_path),
                dtype="float32",
            )

        except Exception as e:

            raise RuntimeError(
                f"Could not read audio file: {e}"
            ) from e

        if audio is None:

            raise ValueError(
                "Audio data is empty."
            )

        if len(audio) == 0:

            raise ValueError(
                "Audio file contains no samples."
            )

        # --------------------------------------
        # Stereo -> Mono
        # --------------------------------------

        if audio.ndim > 1:

            audio = np.mean(
                audio,
                axis=1,
            )

        # --------------------------------------
        # Resample -> 16 kHz
        # --------------------------------------

        audio = (
            SpeechKeywordDetector
            .resample_audio(
                audio=audio,
                original_sample_rate=sample_rate,
                target_sample_rate=TARGET_SAMPLE_RATE,
            )
        )

        return np.asarray(
            audio,
            dtype=np.float32,
        )


    # ==========================================
    # Keyword Matching
    # ==========================================

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

        keywords = sorted(
            EMERGENCY_KEYWORDS,
            key=len,
            reverse=True,
        )

        for keyword in keywords:

            if keyword in normalized:

                return keyword

        return None


    # ==========================================
    # Prediction
    # ==========================================

    def predict(self, audio_path):

        audio_path = Path(
            audio_path
        )

        # --------------------------------------
        # Check audio file
        # --------------------------------------

        if not audio_path.exists():

            raise FileNotFoundError(
                f"Audio file not found:\n"
                f"{audio_path}"
            )

        # --------------------------------------
        # Load audio
        # --------------------------------------

        audio = self.load_audio(
            audio_path
        )

        print(
            "\nAudio loaded successfully."
        )

        print(
            f"Audio samples : {len(audio)}"
        )

        print(
            f"Sample rate   : "
            f"{TARGET_SAMPLE_RATE} Hz"
        )

        # ======================================
        # Whisper Processor
        # ======================================

        inputs = self.processor(
            audio,
            sampling_rate=TARGET_SAMPLE_RATE,
            return_tensors="pt",
        )

        input_features = (
            inputs.input_features
            .to(self.device)
        )

        # ======================================
        # Whisper Generation
        # ======================================

        with torch.inference_mode():

            generated_ids = (
                self.model.generate(
                    input_features,
                    language="english",
                    task="transcribe",
                )
            )

        # ======================================
        # Decode
        # ======================================

        transcript = (
            self.processor.batch_decode(
                generated_ids,
                skip_special_tokens=True,
            )[0]
            .strip()
        )

        # --------------------------------------
        # Emergency keyword
        # --------------------------------------

        keyword = self.find_keyword(
            transcript
        )

        emergency_detected = (
            keyword is not None
        )

        # ======================================
        # Logging
        # ======================================

        print(
            "\n=============================="
        )

        print(
            "SPEECH RESULT"
        )

        print(
            "=============================="
        )

        print(
            f"Transcript : {transcript}"
        )

        print(
            f"Keyword    : {keyword}"
        )

        print(
            f"Emergency  : "
            f"{emergency_detected}"
        )

        # ======================================
        # Result
        # ======================================

        return {

            "status": "success",

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

    try:

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

    except Exception as e:

        print(
            "\nSpeech prediction failed:"
        )

        print(
            str(e)
        )