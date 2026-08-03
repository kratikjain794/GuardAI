from collections import Counter
from pathlib import Path

from app.ai.emotion_utils import get_emotion_from_filename


BASE_DIR = Path(__file__).resolve().parents[3]
DATASET_DIR = BASE_DIR / "datasets" / "emotion"


def main():

    print("\n================================")
    print("GuardIA RAVDESS Dataset Check")
    print("================================")

    print(f"\nDataset: {DATASET_DIR}")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset folder not found: {DATASET_DIR}"
        )

    actor_folders = sorted(
        DATASET_DIR.glob("Actor_*")
    )

    print(
        f"Actor folders found: "
        f"{len(actor_folders)}"
    )

    total_files = 0
    valid_files = 0
    invalid_files = []

    emotion_counts = Counter()

    for actor_folder in actor_folders:

        wav_files = sorted(
            actor_folder.glob("*.wav")
        )

        print(
            f"{actor_folder.name}: "
            f"{len(wav_files)} files"
        )

        total_files += len(wav_files)

        for file_path in wav_files:

            try:

                emotion = (
                    get_emotion_from_filename(
                        file_path
                    )
                )

                emotion_counts[emotion] += 1
                valid_files += 1

            except ValueError as error:

                invalid_files.append(
                    {
                        "file": file_path.name,
                        "error": str(error),
                    }
                )

    print("\n================================")
    print("Dataset Summary")
    print("================================")

    print(f"Actors        : {len(actor_folders)}")
    print(f"Total WAV     : {total_files}")
    print(f"Valid WAV     : {valid_files}")
    print(f"Invalid WAV   : {len(invalid_files)}")

    print("\nEmotion Distribution")

    for emotion, count in sorted(
        emotion_counts.items()
    ):
        print(
            f"{emotion:12s}: {count}"
        )

    if invalid_files:

        print("\nInvalid files:")

        for item in invalid_files[:10]:

            print(
                f"{item['file']} -> "
                f"{item['error']}"
            )

    print("\n================================")

    if (
        len(actor_folders) == 24
        and valid_files > 0
        and not invalid_files
    ):

        print("Dataset validation successful!")

    else:

        print(
            "Dataset validation completed "
            "with warnings."
        )

    print("================================\n")


if __name__ == "__main__":
    main()