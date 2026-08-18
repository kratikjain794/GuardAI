from pathlib import Path

import torch
from torch.utils.data import Dataset

from app.ai.emotion_utils import (
    extract_mfcc,
    get_emotion_from_filename,
)


# ==========================================
# Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = (
    BASE_DIR
    / "datasets"
    / "emotion"
)


# ==========================================
# Distress Labels
# ==========================================

DISTRESS_EMOTIONS = {
    "angry",
    "fearful",
}


LABEL_TO_INDEX = {
    "normal": 0,
    "distress": 1,
}


INDEX_TO_LABEL = {
    0: "normal",
    1: "distress",
}


# ==========================================
# Voice Sample
# ==========================================

class VoiceSample:

    def __init__(
        self,
        path: Path,
        label: str,
    ):

        self.path = path

        self.label = label


# ==========================================
# Label Generator
# ==========================================

def get_distress_label(
    file_path: Path,
) -> str:

    filename = file_path.stem

    parts = filename.split("-")

    emotion_code = parts[2]

    intensity_code = parts[3]

    # Distress only if:
    # Angry (05) OR Fearful (06)
    # AND Strong Intensity (02)

    if (
        emotion_code in ["05", "06"]
        and intensity_code == "02"
    ):
        return "distress"

    return "normal"

# ==========================================
# Actor Loader
# ==========================================

def get_actor_files(
    actor_numbers=None,
):

    files = []

    if actor_numbers is None:

        actor_folders = sorted(

            DATASET_DIR.glob(
                "Actor_*"
            )

        )

    else:

        actor_folders = [

            DATASET_DIR
            / f"Actor_{actor:02d}"

            for actor in actor_numbers

        ]

    if not actor_folders:

        raise FileNotFoundError(

            f"No Actor folders found in:\n"
            f"{DATASET_DIR}"

        )

    for actor_folder in actor_folders:

        if not actor_folder.exists():

            raise FileNotFoundError(

                f"Folder not found:\n"
                f"{actor_folder}"

            )

        actor_files = sorted(

            actor_folder.glob("*.wav")

        )

        files.extend(actor_files)

    return files

# ==========================================
# Dataset Loader
# ==========================================

def load_dataset():

    samples = []

    files = get_actor_files(
    actor_numbers=None
    )

    normal_count = 0

    distress_count = 0

    for file_path in files:

        label = get_distress_label(
            file_path
        )

        if label == "normal":

            normal_count += 1

        else:

            distress_count += 1

        samples.append(

            VoiceSample(

                path=file_path,

                label=label,

            )

        )

    print("\n==============================")

    print("Voice Distress Dataset")

    print("==============================")

    print(
        f"Total Samples : {len(samples)}"
    )

    print(
        f"Normal        : {normal_count}"
    )

    print(
        f"Distress      : {distress_count}"
    )

    return samples


# ==========================================
# Dataset
# ==========================================

class VoiceDatasetV3(Dataset):

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

            sample.label

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

    samples = load_dataset()

    dataset = VoiceDatasetV3(
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
        "Voice Dataset V3"
    )

    print("==============================")

    print(
        f"Samples : {len(dataset)}"
    )

    features, label = dataset[0]

    print(
        f"MFCC Shape : {features.shape}"
    )

    print(
        f"Label : {label}"
    )

    print(
        f"Class : {INDEX_TO_LABEL[label.item()]}"
    )

