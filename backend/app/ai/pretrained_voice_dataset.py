from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from torch.utils.data import Dataset

from transformers import AutoProcessor

from app.ai.voice_dataset_v3 import (
    get_actor_files,
    get_distress_label,
)


# ==========================================
# Configuration
# ==========================================

BASE_DIR = (
    Path(__file__).resolve().parents[3]
)

MODEL_NAME = "facebook/wav2vec2-base"

TARGET_SAMPLE_RATE = 16000


# ==========================================
# Wav2Vec2 Processor
# ==========================================

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


# ==========================================
# Audio Loader
# ==========================================

def load_audio(
    audio_path: Path,
):
    """
    Load WAV audio and convert it to:
        - mono
        - 16 kHz
        - float32
    """

    audio, sample_rate = sf.read(
        str(audio_path),
        dtype="float32",
    )

    # --------------------------------------
    # NumPy → Torch
    # --------------------------------------

    if audio.ndim == 1:

        waveform = torch.from_numpy(
            audio
        ).unsqueeze(0)

    else:

        # (Samples, Channels)
        # →
        # (Channels, Samples)

        waveform = torch.from_numpy(
            audio.T
        )

    # --------------------------------------
    # Stereo → Mono
    # --------------------------------------

    if waveform.size(0) > 1:

        waveform = waveform.mean(
            dim=0,
            keepdim=True,
        )

    # --------------------------------------
    # Resample → 16 kHz
    # --------------------------------------

    if sample_rate != TARGET_SAMPLE_RATE:

        original_length = (
            waveform.shape[-1]
        )

        target_length = int(
            original_length
            * TARGET_SAMPLE_RATE
            / sample_rate
        )

        waveform = F.interpolate(
            waveform.unsqueeze(0),
            size=target_length,
            mode="linear",
            align_corners=False,
        ).squeeze(0)

        sample_rate = TARGET_SAMPLE_RATE

    # --------------------------------------
    # Tensor → NumPy
    # --------------------------------------

    audio_array = (
        waveform
        .squeeze(0)
        .cpu()
        .numpy()
    )

    return (
        audio_array,
        sample_rate,
    )


# ==========================================
# Dataset
# ==========================================

class PretrainedVoiceDistressDataset(
    Dataset
):

    def __init__(
        self,
        files,
    ):

        self.files = list(files)

    # --------------------------------------

    def __len__(self):

        return len(
            self.files
        )

    # --------------------------------------

    def __getitem__(
        self,
        index,
    ):

        audio_path = Path(
            self.files[index]
        )

        # ----------------------------------
        # Load audio
        # ----------------------------------

        audio, sample_rate = load_audio(
            audio_path
        )

        # ----------------------------------
        # Wav2Vec2 processor
        # ----------------------------------

        processed = processor(
            audio,
            sampling_rate=sample_rate,
            return_tensors="pt",
        )

        input_values = (
            processed.input_values
            .squeeze(0)
        )

        # ----------------------------------
        # Label
        # ----------------------------------

        label_name = get_distress_label(
            audio_path
        )

        label = (
            1
            if label_name == "distress"
            else 0
        )

        return (
            input_values,
            torch.tensor(
                label,
                dtype=torch.long,
            ),
        )


# ==========================================
# Actor Split
# ==========================================

def create_voice_splits():

    train_files = get_actor_files(
        range(1, 19)
    )

    validation_files = get_actor_files(
        [19, 20, 21]
    )

    test_files = get_actor_files(
        [22, 23, 24]
    )

    return (
        train_files,
        validation_files,
        test_files,
    )


# ==========================================
# Dataset Statistics
# ==========================================

def count_labels(files):

    normal = 0
    distress = 0

    for file_path in files:

        label = get_distress_label(
            Path(file_path)
        )

        if label == "distress":

            distress += 1

        else:

            normal += 1

    return (
        normal,
        distress,
    )


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "Pretrained Voice Dataset"
    )

    print(
        "=============================="
    )

    # --------------------------------------
    # Create splits
    # --------------------------------------

    (
        train_files,
        validation_files,
        test_files,
    ) = create_voice_splits()

    print(
        f"\nTrain Samples      : "
        f"{len(train_files)}"
    )

    print(
        f"Validation Samples : "
        f"{len(validation_files)}"
    )

    print(
        f"Test Samples       : "
        f"{len(test_files)}"
    )

    # --------------------------------------
    # Label statistics
    # --------------------------------------

    train_normal, train_distress = (
        count_labels(train_files)
    )

    val_normal, val_distress = (
        count_labels(validation_files)
    )

    test_normal, test_distress = (
        count_labels(test_files)
    )

    print(
        "\n=============================="
    )

    print(
        "Label Distribution"
    )

    print(
        "=============================="
    )

    print(
        f"\nTrain:"
        f"\n  Normal   : {train_normal}"
        f"\n  Distress : {train_distress}"
    )

    print(
        f"\nValidation:"
        f"\n  Normal   : {val_normal}"
        f"\n  Distress : {val_distress}"
    )

    print(
        f"\nTest:"
        f"\n  Normal   : {test_normal}"
        f"\n  Distress : {test_distress}"
    )

    # --------------------------------------
    # Create dataset
    # --------------------------------------

    dataset = (
        PretrainedVoiceDistressDataset(
            train_files
        )
    )

    print(
        "\n=============================="
    )

    print(
        "Testing One Sample"
    )

    print(
        "=============================="
    )

    features, label = dataset[0]

    print(
        "\nAudio Tensor Shape :",
        features.shape,
    )

    print(
        "Label              :",
        label.item(),
    )

    print(
        "Class              :",
        (
            "distress"
            if label.item() == 1
            else "normal"
        ),
    )

    print(
        "\nDataset Loader Test Successful!"
    )