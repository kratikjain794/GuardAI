from pathlib import Path
from dataclasses import dataclass
from typing import List

from app.ai.emotion_utils import (
    EMOTION_MAP,
    get_emotion_from_filename,
)


# ==========================================
# Base Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

RAVDESS_DIR = (
    BASE_DIR
    / "datasets"
    / "emotion"
)

CREMA_DIR = (
    BASE_DIR
    / "datasets"
    / "CREMA-D"
)

TESS_DIR = (
    BASE_DIR
    / "datasets"
    / "TESS"
)


# ==========================================
# Dataset Item
# ==========================================

@dataclass
class AudioSample:

    path: Path

    emotion: str

    speaker: str

    dataset: str


# ==========================================
# CREMA Mapping
# ==========================================

CREMA_MAP = {

    "ANG": "angry",

    "DIS": "disgust",

    "FEA": "fearful",

    "HAP": "happy",

    "NEU": "neutral",

    "SAD": "sad",
}


# ==========================================
# TESS Mapping
# ==========================================

TESS_MAP = {

    "angry": "angry",

    "disgust": "disgust",

    "fear": "fearful",

    "happy": "happy",

    "neutral": "neutral",

    "sad": "sad",

    "pleasant_surprise": "surprised",

    "pleasant_surprised": "surprised",
}

def load_ravdess():

    samples = []

    for actor in sorted(
        RAVDESS_DIR.glob("Actor_*")
    ):

        speaker = actor.name

        for wav in actor.glob("*.wav"):

            emotion = (
                get_emotion_from_filename(
                    wav
                )
            )

            samples.append(

                AudioSample(

                    path=wav,

                    emotion=emotion,

                    speaker=speaker,

                    dataset="RAVDESS",

                )

            )

    return samples


# ==========================================
# CREMA-D Loader
# ==========================================

def load_crema():

    samples = []

    for wav in sorted(
        CREMA_DIR.glob("*.wav")
    ):

        parts = wav.stem.split("_")

        if len(parts) != 4:
            continue

        speaker = parts[0]

        emotion_code = parts[2]

        if emotion_code not in CREMA_MAP:
            continue

        emotion = CREMA_MAP[
            emotion_code
        ]

        samples.append(

            AudioSample(

                path=wav,

                emotion=emotion,

                speaker=f"CREMA_{speaker}",

                dataset="CREMA-D",

            )

        )

    return samples


# ==========================================
# TESS Loader
# ==========================================

def load_tess():

    samples = []

    for folder in sorted(
        TESS_DIR.iterdir()
    ):

        if not folder.is_dir():
            continue

        folder_name = (
            folder.name
            .lower()
            .replace("-", "_")
            .replace(" ", "_")
        )

        # Skip duplicate nested dataset
        if folder_name.startswith(
            "tess_toronto"
        ):
            continue

        parts = folder_name.split("_")

        if len(parts) < 2:
            continue

        speaker = parts[0]

        emotion_key = "_".join(
            parts[1:]
        )

        if emotion_key not in TESS_MAP:
            continue

        emotion = TESS_MAP[
            emotion_key
        ]

        for wav in sorted(
            folder.glob("*.wav")
        ):

            samples.append(

                AudioSample(

                    path=wav,

                    emotion=emotion,

                    speaker=f"TESS_{speaker}",

                    dataset="TESS",

                )

            )

    return samples


# ==========================================
# Test Loader
# ==========================================

def load_all_datasets():

    ravdess = load_ravdess()

    crema = load_crema()

    tess = load_tess()

    all_samples = (
        ravdess
        + crema
        + tess
    )

    print("\n==============================")

    print("Combined Dataset Summary")

    print("==============================")

    print(
        f"RAVDESS : {len(ravdess)}"
    )

    print(
        f"CREMA-D : {len(crema)}"
    )

    print(
        f"TESS     : {len(tess)}"
    )

    print("------------------------------")

    print(
        f"TOTAL    : {len(all_samples)}"
    )

    return all_samples


if __name__ == "__main__":

    load_all_datasets()