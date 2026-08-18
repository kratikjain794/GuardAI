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
# Emotion Mapping
# ==========================================

EMOTION_MAPPING = {
    "calm": "neutral",
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

        return len(self.samples)

    def __getitem__(
        self,
        index,
    ):

        sample = self.samples[index]

        # -----------------------------
        # MFCC Features
        # -----------------------------

        features = extract_mfcc(
            sample.path
        )

        # -----------------------------
        # Normalize Emotion Label
        # -----------------------------

        emotion = sample.emotion.lower().strip()

        if emotion in EMOTION_MAPPING:
            emotion = EMOTION_MAPPING[emotion]

        if emotion not in LABEL_TO_INDEX:
            raise ValueError(
                f"Unknown emotion found: {emotion}"
            )

        label = LABEL_TO_INDEX[
            emotion
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

    print(
        f"\nLoaded {len(samples)} samples."
    )

    return EmotionDatasetV3(
        samples
    )


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
        "Total Samples :",
        len(dataset),
    )

    features, label = dataset[0]

    print(
        "MFCC Shape :",
        features.shape,
    )

    print(
        "Label Index :",
        label.item(),
    )

    print(
        "Label Name :",
        INDEX_TO_LABEL[
            label.item()
        ],
    )