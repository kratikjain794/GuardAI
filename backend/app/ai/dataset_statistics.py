from collections import Counter

from app.ai.combined_dataset import (
    load_all_datasets,
)


# ==========================================
# Dataset Statistics
# ==========================================

def main():

    print("\n================================")
    print("GuardIA Dataset Statistics")
    print("================================")

    samples = load_all_datasets()

    dataset_counter = Counter()

    emotion_counter = Counter()

    speaker_counter = Counter()

    duplicate_checker = set()

    duplicate_files = 0

    # --------------------------------------

    for sample in samples:

        dataset_counter[
            sample.dataset
        ] += 1

        emotion_counter[
            sample.emotion
        ] += 1

        speaker_counter[
            sample.speaker
        ] += 1

        if sample.path in duplicate_checker:

            duplicate_files += 1

        duplicate_checker.add(
            sample.path
        )

    # --------------------------------------

    print("\nDataset Distribution")

    print("------------------------------")

    for dataset, count in sorted(
        dataset_counter.items()
    ):

        print(
            f"{dataset:<10}: {count}"
        )

    # --------------------------------------

    print("\nEmotion Distribution")

    print("------------------------------")

    for emotion, count in sorted(
        emotion_counter.items()
    ):

        print(
            f"{emotion:<12}: {count}"
        )

    # --------------------------------------

    print("\nSummary")

    print("------------------------------")

    print(
        f"Total Samples   : "
        f"{len(samples)}"
    )

    print(
        f"Unique Speakers : "
        f"{len(speaker_counter)}"
    )

    print(
        f"Duplicate Files : "
        f"{duplicate_files}"
    )

    # --------------------------------------

    print("\nLargest Dataset")

    print("------------------------------")

    dataset = max(
        dataset_counter,
        key=dataset_counter.get,
    )

    print(
        dataset,
        dataset_counter[dataset],
    )

    # --------------------------------------

    print("\nMost Common Emotion")

    print("------------------------------")

    emotion = max(
        emotion_counter,
        key=emotion_counter.get,
    )

    print(
        emotion,
        emotion_counter[emotion],
    )

    print("\n================================")


if __name__ == "__main__":
    main()