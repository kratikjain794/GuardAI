from collections import Counter
from pathlib import Path


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "TESS"
)


# ==========================================
# Expected TESS Folders
# ==========================================

FOLDER_EMOTION_MAP = {
    "oaf_angry": "angry",
    "oaf_disgust": "disgust",
    "oaf_fear": "fearful",
    "oaf_happy": "happy",
    "oaf_neutral": "neutral",
    "oaf_pleasant_surprise": "surprised",
    "oaf_sad": "sad",

    "yaf_angry": "angry",
    "yaf_disgust": "disgust",
    "yaf_fear": "fearful",
    "yaf_happy": "happy",
    "yaf_neutral": "neutral",
    "yaf_pleasant_surprised": "surprised",
    "yaf_sad": "sad",
}


EMOTIONS = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


# ==========================================
# Normalize Folder Name
# ==========================================

def normalize_folder_name(
    name: str,
) -> str:

    return (
        name
        .strip()
        .lower()
        .replace("-", "_")
        .replace(" ", "_")
    )


# ==========================================
# Main Validation
# ==========================================

def main():

    print("\n================================")
    print("TESS Dataset Validation")
    print("================================")

    print(
        f"\nDataset path:\n"
        f"{DATASET_DIR}"
    )

    if not DATASET_DIR.exists():

        raise FileNotFoundError(
            f"TESS dataset not found: "
            f"{DATASET_DIR}"
        )

    # ======================================
    # Find Main Emotion Folders
    # ======================================

    directories = [
        path
        for path in DATASET_DIR.iterdir()
        if path.is_dir()
    ]

    emotion_counts = Counter()

    speaker_counts = Counter()

    valid_files = []

    invalid_folders = []

    missing_folders = []

    detected_folders = set()

    # ======================================
    # Process Direct Child Folders
    # ======================================

    for folder in directories:

        normalized_name = (
            normalize_folder_name(
                folder.name
            )
        )

        if (
            normalized_name
            not in FOLDER_EMOTION_MAP
        ):

            invalid_folders.append(
                folder
            )

            continue

        detected_folders.add(
            normalized_name
        )

        emotion = (
            FOLDER_EMOTION_MAP[
                normalized_name
            ]
        )

        if normalized_name.startswith(
            "oaf_"
        ):
            speaker = "OAF"

        elif normalized_name.startswith(
            "yaf_"
        ):
            speaker = "YAF"

        else:
            speaker = "unknown"

        wav_files = sorted(
            folder.glob("*.wav")
        )

        for wav_file in wav_files:

            valid_files.append(
                wav_file
            )

            emotion_counts[
                emotion
            ] += 1

            speaker_counts[
                speaker
            ] += 1

    # ======================================
    # Missing Folder Check
    # ======================================

    for expected_folder in (
        FOLDER_EMOTION_MAP
    ):

        if (
            expected_folder
            not in detected_folders
        ):

            missing_folders.append(
                expected_folder
            )

    # ======================================
    # Results
    # ======================================

    print(
        f"\nValid WAV files : "
        f"{len(valid_files)}"
    )

    print(
        f"Emotion folders : "
        f"{len(detected_folders)}"
    )

    print(
        "\nSpeaker distribution:"
    )

    print(
        "--------------------------------"
    )

    for speaker in [
        "OAF",
        "YAF",
    ]:

        print(
            f"{speaker:<5}: "
            f"{speaker_counts[speaker]}"
        )

    print(
        "\nEmotion distribution:"
    )

    print(
        "--------------------------------"
    )

    for emotion in EMOTIONS:

        print(
            f"{emotion:<10}: "
            f"{emotion_counts[emotion]}"
        )

    # ======================================
    # Unknown / Extra Folders
    # ======================================

    print(
        "\nExtra folders:"
    )

    print(
        "--------------------------------"
    )

    if not invalid_folders:

        print("None")

    else:

        for folder in invalid_folders:

            recursive_wavs = list(
                folder.rglob("*.wav")
            )

            print(
                f"{folder.name}"
                f" -> "
                f"{len(recursive_wavs)} "
                f"WAV files"
            )

            if recursive_wavs:

                print(
                    "  WARNING: This folder "
                    "contains WAV files."
                )

                print(
                    "  It may contain a duplicate "
                    "copy of TESS."
                )

    # ======================================
    # Missing Folders
    # ======================================

    if missing_folders:

        print(
            "\nMissing expected folders:"
        )

        for folder in missing_folders:

            print(
                f"- {folder}"
            )

        raise RuntimeError(
            "TESS validation failed because "
            "expected folders are missing."
        )

    # ======================================
    # Missing Emotion Check
    # ======================================

    missing_emotions = [
        emotion
        for emotion in EMOTIONS
        if emotion_counts[emotion] == 0
    ]

    if missing_emotions:

        raise RuntimeError(
            "Missing TESS emotions: "
            + ", ".join(
                missing_emotions
            )
        )

    # ======================================
    # Speaker Check
    # ======================================

    if (
        speaker_counts["OAF"] == 0
        or speaker_counts["YAF"] == 0
    ):

        raise RuntimeError(
            "Both OAF and YAF speakers "
            "were not detected."
        )

    # ======================================
    # Duplicate Path Check
    # ======================================

    unique_paths = set(
        valid_files
    )

    if (
        len(unique_paths)
        != len(valid_files)
    ):

        raise RuntimeError(
            "Duplicate file paths detected."
        )

    # ======================================
    # Success
    # ======================================

    print("\n================================")
    print("TESS validation successful!")
    print("================================")


if __name__ == "__main__":
    main()