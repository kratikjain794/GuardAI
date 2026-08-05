from pathlib import Path

import torch
from torch.utils.data import Dataset

from app.ai.combined_dataset import (
    load_all_datasets,
)
from app.ai.emotion_utils import (
    extract_mfcc,
)


# ==========================================
# Emotion Labels
# ==========================================

EMOTIONS = [
    "angry",
    "disgust",
    "fearful",
    "happy",
    "neutral",
    "sad",
    "surprised",
]


LABEL_TO_INDEX = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}


INDEX_TO_LABEL = {
    index: emotion
    for emotion, index in LABEL_TO_INDEX.items()
}

# ==========================================
# Emotion Dataset V3
# ==========================================

class EmotionDatasetV3(Dataset):

    def __init__(
        self,
        samples,
    ):

        self.samples = samples

    def __len__(self):

        return len(
            self.samples
        )

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[index]

        features = extract_mfcc(
            sample.path
        )

        label = LABEL_TO_INDEX[
            sample.emotion
        ]

        return (
            torch.from_numpy(
                features
            ).float(),
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


    
# ==========================================
# Loader
# ==========================================

def create_dataset():

    samples = load_all_datasets()

    dataset = EmotionDatasetV3(
        samples
    )

    return dataset


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    dataset = create_dataset()

    print("\n==============================")

    print(
        "Emotion Dataset V3"
    )

    print("==============================")

    print(
        f"Samples : "
        f"{len(dataset)}"
    )

    features, label = dataset[0]

    print(
        f"MFCC Shape : "
        f"{features.shape}"
    )

    print(
        f"Label : "
        f"{label}"
    )