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

MODEL_NAME = "Dpngtm/wav2vec2-emotion-recognition"
SAMPLE_RATE = 16000

BASE_DIR = Path(__file__).resolve().parents[2]


# ==========================================
# Shared Pretrained Model
# ==========================================

_SHARED_PROCESSOR = None
_SHARED_MODEL = None
_SHARED_DEVICE = None
_SHARED_ID_TO_LABEL = None


def load_shared_emotion_model():
    """
    Load the pretrained Wav2Vec2 emotion model
    only ONCE per Python worker.

    Voice and Emotion detectors both use this
    same model instance.
    """

    global _SHARED_PROCESSOR
    global _SHARED_MODEL
    global _SHARED_DEVICE
    global _SHARED_ID_TO_LABEL

    if (
        _SHARED_MODEL is not None
        and _SHARED_PROCESSOR is not None
    ):
        return (
            _SHARED_PROCESSOR,
            _SHARED_MODEL,
            _SHARED_DEVICE,
            _SHARED_ID_TO_LABEL,
        )

    _SHARED_DEVICE = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        "\n=========================================="
    )
    print(
        "Loading SHARED pretrained Wav2Vec2 model"
    )
    print(
        f"Model : {MODEL_NAME}"
    )
    print(
        f"Device: {_SHARED_DEVICE}"
    )
    print(
        "=========================================="
    )

    # --------------------------------------
    # Processor
    # --------------------------------------

    _SHARED_PROCESSOR = (
        AutoProcessor.from_pretrained(
            MODEL_NAME
        )
    )

    # --------------------------------------
    # Model
    # --------------------------------------

    _SHARED_MODEL = (
        AutoModelForAudioClassification
        .from_pretrained(
            MODEL_NAME
        )
        .to(_SHARED_DEVICE)
    )

    _SHARED_MODEL.eval()

    # --------------------------------------
    # Labels
    # --------------------------------------

    _SHARED_ID_TO_LABEL = (
        _SHARED_MODEL.config.id2label
    )

    print(
        "\nEmotion Label Mapping:"
    )

    for index, label in (
        _SHARED_ID_TO_LABEL.items()
    ):
        print(
            f"  {index} -> {label}"
        )

    print(
        "\nShared pretrained Wav2Vec2 "
        "loaded successfully!"
    )

    return (
        _SHARED_PROCESSOR,
        _SHARED_MODEL,
        _SHARED_DEVICE,
        _SHARED_ID_TO_LABEL,
    )


# ==========================================
# Emotion Detector
# ==========================================

class EmotionDetector:

    def __init__(self):

        (
            self.processor,
            self.model,
            self.device,
            self.id_to_label,
        ) = load_shared_emotion_model()

        print(
            "EmotionDetector using shared "
            "Wav2Vec2 model."
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
        # Move tensors
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

            probabilities = torch.softmax(
                outputs.logits,
                dim=-1,
            )

            prediction = int(
                torch.argmax(
                    probabilities,
                    dim=-1,
                ).item()
            )

            confidence = float(
                probabilities[
                    0,
                    prediction,
                ].item()
            )

        # ----------------------------------
        # Label
        # ----------------------------------

        emotion = self.id_to_label.get(
            prediction,
            str(prediction),
        )

        emotion = str(
            emotion
        ).lower()

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