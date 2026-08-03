from pathlib import Path

import librosa
import numpy as np

from app.utils.constants import (
    AUDIO_DURATION_SECONDS,
    AUDIO_SAMPLE_RATE,
    N_MFCC,
)


# RAVDESS filename emotion codes
EMOTION_MAP = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised",
}


# Fixed MFCC width
MAX_FRAMES = 94


def get_emotion_from_filename(
    file_path: str | Path,
) -> str:
    """
    Extract emotion label from a RAVDESS filename.

    Example:
    03-01-06-01-02-01-12.wav
             ^^
             06 = fearful
    """

    filename = Path(file_path).stem
    parts = filename.split("-")

    if len(parts) != 7:
        raise ValueError(
            f"Invalid RAVDESS filename: {filename}"
        )

    emotion_code = parts[2]

    if emotion_code not in EMOTION_MAP:
        raise ValueError(
            f"Unknown emotion code: {emotion_code}"
        )

    return EMOTION_MAP[emotion_code]


def load_audio(
    file_path: str | Path,
) -> np.ndarray:
    """Load audio and normalize it to a fixed duration."""

    file_path = Path(file_path)

    if not file_path.exists():
        raise FileNotFoundError(
            f"Audio file not found: {file_path}"
        )

    audio, _ = librosa.load(
        file_path,
        sr=AUDIO_SAMPLE_RATE,
        mono=True,
        duration=AUDIO_DURATION_SECONDS,
    )

    target_length = int(
        AUDIO_SAMPLE_RATE
        * AUDIO_DURATION_SECONDS
    )

    if len(audio) < target_length:
        audio = np.pad(
            audio,
            (0, target_length - len(audio)),
        )

    else:
        audio = audio[:target_length]

    return audio.astype(np.float32)


def extract_mfcc(
    file_path: str | Path,
) -> np.ndarray:
    """
    Convert audio into fixed-size MFCC features.

    Output shape:
    (N_MFCC, MAX_FRAMES)
    """

    audio = load_audio(file_path)

    mfcc = librosa.feature.mfcc(
        y=audio,
        sr=AUDIO_SAMPLE_RATE,
        n_mfcc=N_MFCC,
    )

    if mfcc.shape[1] < MAX_FRAMES:

        difference = (
            MAX_FRAMES - mfcc.shape[1]
        )

        mfcc = np.pad(
            mfcc,
            (
                (0, 0),
                (0, difference),
            ),
        )

    else:
        mfcc = mfcc[:, :MAX_FRAMES]

    return mfcc.astype(np.float32)