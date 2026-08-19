import numpy as np
import soundfile as sf

from app.ai.emotion_detection import (
    load_shared_emotion_model,
)


# ==========================================
# Configuration
# ==========================================

TARGET_SAMPLE_RATE = 16000


# ==========================================
# Voice Emotion / Distress Detector
# ==========================================

class VoiceDistressDetector:

    def __init__(self):

        (
            self.feature_extractor,
            self.model,
            self.device,
            self.id2label,
        ) = load_shared_emotion_model()

        print(
            "\nVoiceDistressDetector using "
            "shared pretrained Wav2Vec2 model."
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
    # Prediction
    # ==========================================

    def predict(
        self,
        audio_path,
    ):

        audio_path = str(
            audio_path
        )

        # --------------------------------------
        # Load audio
        # --------------------------------------

        try:

            audio, sample_rate = (
                sf.read(
                    audio_path,
                    dtype="float32",
                )
            )

        except Exception as e:

            raise RuntimeError(
                f"Could not read audio file: {e}"
            ) from e


        # --------------------------------------
        # Validate
        # --------------------------------------

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

        audio = self.resample_audio(
            audio=audio,
            original_sample_rate=sample_rate,
            target_sample_rate=TARGET_SAMPLE_RATE,
        )


        # --------------------------------------
        # Float32
        # --------------------------------------

        audio = np.asarray(
            audio,
            dtype=np.float32,
        )


        # --------------------------------------
        # Feature extraction
        # --------------------------------------

        inputs = (
            self.feature_extractor(
                audio,
                sampling_rate=TARGET_SAMPLE_RATE,
                return_tensors="pt",
                padding=True,
            )
        )


        # --------------------------------------
        # Move tensors
        # --------------------------------------

        inputs = {
            key: value.to(self.device)
            for key, value in inputs.items()
        }


        # --------------------------------------
        # Prediction
        # --------------------------------------

        import torch

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


        # --------------------------------------
        # Best prediction
        # --------------------------------------

        prediction_index = int(
            torch.argmax(
                probabilities,
                dim=-1,
            ).item()
        )


        confidence = float(
            probabilities[
                0,
                prediction_index,
            ].item()
        )


        # --------------------------------------
        # Label
        # --------------------------------------

        emotion = self.id2label.get(
            prediction_index,
            str(prediction_index),
        )

        emotion = str(
            emotion
        ).lower().strip()


        # ======================================
        # Distress Mapping
        # ======================================

        distress_emotions = {
            "angry",
            "fear",
            "fearful",
            "scared",
            "afraid",
        }


        distress_detected = (
            emotion
            in distress_emotions
        )


        # ======================================
        # Logging
        # ======================================

        print(
            "\n=============================="
        )

        print(
            "VOICE EMOTION RESULT"
        )

        print(
            "=============================="
        )

        print(
            f"Emotion     : {emotion}"
        )

        print(
            f"Confidence  : "
            f"{confidence * 100:.2f}%"
        )

        print(
            f"Distress    : "
            f"{distress_detected}"
        )


        # ======================================
        # Result
        # ======================================

        return {

            "status": "success",

            "emotion":
                emotion,

            "confidence":
                round(
                    confidence * 100,
                    2,
                ),

            "distress_detected":
                distress_detected,

            "label": (
                "distress"
                if distress_detected
                else "normal"
            ),

        }


# ==========================================
# Local Test
# ==========================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "Pretrained Voice Emotion Detector"
    )

    print(
        "=============================="
    )


    detector = (
        VoiceDistressDetector()
    )


    print(
        "\nModel Loaded Successfully!"
    )


    audio_path = input(
        "\nEnter WAV file path: "
    ).strip()


    try:

        result = detector.predict(
            audio_path
        )

        print(
            "\nPrediction:"
        )

        print(
            "------------------------------"
        )

        for key, value in result.items():

            print(
                f"{key}: {value}"
            )


    except Exception as e:

        print(
            "\nVoice prediction failed:"
        )

        print(
            str(e)
        )