from collections import Counter, defaultdict
from pathlib import Path


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "CREMA-D"
)


# ==========================================
# CREMA-D Emotion Mapping
# ==========================================

EMOTION_MAP = {
    "ANG": "angry",
    "DIS": "disgust",
    "FEA": "fearful",
    "HAP": "happy",
    "NEU": "neutral",
    "SAD": "sad",
}


# ==========================================
# Parse Filename
# Example:
# 1001_DFA_ANG_XX.wav
# ==========================================

def parse_crema_filename(
    file_path: Path,
):

    parts = file_path.stem.split("_")

    if len(parts) != 4:
        raise ValueError(
            f"Invalid filename format: "
            f"{file_path.name}"
        )

    speaker_id = parts[0]
    sentence_code = parts[1]
    emotion_code = parts[2]
    intensity_code = parts[3]

    if emotion_code not in EMOTION_MAP:
        raise ValueError(
            f"Unknown emotion code "
            f"'{emotion_code}' in "
            f"{file_path.name}"
        )

    emotion = EMOTION_MAP[
        emotion_code
    ]

    return {
        "speaker_id": speaker_id,
        "sentence": sentence_code,
        "emotion": emotion,
        "emotion_code": emotion_code,
        "intensity": intensity_code,
    }


# ==========================================
# Validation
# ==========================================

def main():

    print(
        "\n================================"
    )
    print(
        "CREMA-D Dataset Validation"
    )
    print(
        "================================"
    )

    print(
        f"\nDataset path:\n"
        f"{DATASET_DIR}"
    )

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"CREMA-D dataset not found: "
            f"{DATASET_DIR}"
        )

    wav_files = sorted(
        DATASET_DIR.glob("*.wav")
    )

    if not wav_files:

        raise RuntimeError(
            "No WAV files found in "
            f"{DATASET_DIR}"
        )

    emotion_counts = Counter()

    speakers = set()

    speaker_emotions = defaultdict(
        Counter
    )

    invalid_files = []

    # ======================================
    # Process Files
    # ======================================

    for file_path in wav_files:

        try:

            metadata = (
                parse_crema_filename(
                    file_path
                )
            )

            speaker_id = metadata[
                "speaker_id"
            ]

            emotion = metadata[
                "emotion"
            ]

            speakers.add(
                speaker_id
            )

            emotion_counts[
                emotion
            ] += 1

            speaker_emotions[
                speaker_id
            ][emotion] += 1

        except ValueError as error:

            invalid_files.append(
                {
                    "file": file_path.name,
                    "error": str(error),
                }
            )

    # ======================================
    # Results
    # ======================================

    print(
        f"\nTotal WAV files : "
        f"{len(wav_files)}"
    )

    print(
        f"Valid files     : "
        f"{len(wav_files) - len(invalid_files)}"
    )

    print(
        f"Invalid files   : "
        f"{len(invalid_files)}"
    )

    print(
        f"Unique speakers : "
        f"{len(speakers)}"
    )

    print(
        "\nEmotion distribution:"
    )

    print(
        "--------------------------------"
    )

    for emotion in EMOTION_MAP.values():

        count = emotion_counts[
            emotion
        ]

        print(
            f"{emotion:<10}: "
            f"{count}"
        )

    # ======================================
    # Invalid Files
    # ======================================

    if invalid_files:

        print(
            "\nInvalid files:"
        )

        print(
            "--------------------------------"
        )

        for item in invalid_files[:20]:

            print(
                f"{item['file']} -> "
                f"{item['error']}"
            )

        if len(invalid_files) > 20:

            print(
                f"... and "
                f"{len(invalid_files) - 20} "
                f"more."
            )

        raise RuntimeError(
            "CREMA-D validation failed."
        )

    # ======================================
    # Basic Dataset Checks
    # ======================================

    missing_emotions = [
        emotion
        for emotion in EMOTION_MAP.values()
        if emotion_counts[emotion] == 0
    ]

    if missing_emotions:

        raise RuntimeError(
            "Missing emotions: "
            + ", ".join(
                missing_emotions
            )
        )

    if len(speakers) < 2:

        raise RuntimeError(
            "Not enough unique speakers "
            "for speaker-independent split."
        )

    # ======================================
    # Success
    # ======================================

    print(
        "\n================================"
    )

    print(
        "CREMA-D validation successful!"
    )

    print(
        "================================"
    )


if __name__ == "__main__":
    main()