from pathlib import Path

import torch
from torch.utils.data import DataLoader

from transformers import AutoProcessor

from app.ai.pretrained_voice_dataset import (
    PretrainedVoiceDistressDataset,
    create_voice_splits,
)


# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "facebook/wav2vec2-base"

BATCH_SIZE = 8

NUM_WORKERS = 0


# ==========================================
# Processor
# ==========================================

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)


# ==========================================
# Dynamic Padding Collator
# ==========================================

class VoiceDataCollator:

    def __init__(self, processor):
        self.processor = processor

    def __call__(self, batch):

        # ======================================
        # Audio
        # ======================================

        audio_features = [
            item[0].numpy()
            for item in batch
        ]

        # ======================================
        # Labels
        # ======================================

        labels = torch.stack(
            [
                item[1]
                for item in batch
            ]
        )

        # ======================================
        # Dynamic Padding
        # ======================================

        inputs = self.processor(
            audio_features,
            sampling_rate=16000,
            return_tensors="pt",
            padding=True,
        )

        # ======================================
        # Create Attention Mask if Processor
        # doesn't provide one
        # ======================================

        if hasattr(inputs, "attention_mask"):

            attention_mask = (
                inputs.attention_mask
            )

        else:

            attention_mask = torch.ones_like(
                inputs.input_values,
                dtype=torch.long,
            )

        # ======================================
        # Return Batch
        # ======================================

        return {
            "input_values":
                inputs.input_values,

            "attention_mask":
                attention_mask,

            "labels":
                labels,
        }

# ==========================================
# Create DataLoaders
# ==========================================

def create_dataloaders():

    (
        train_files,
        validation_files,
        test_files,
    ) = create_voice_splits()

    # --------------------------------------
    # Dataset
    # --------------------------------------

    train_dataset = (
        PretrainedVoiceDistressDataset(
            train_files
        )
    )

    validation_dataset = (
        PretrainedVoiceDistressDataset(
            validation_files
        )
    )

    test_dataset = (
        PretrainedVoiceDistressDataset(
            test_files
        )
    )

    # --------------------------------------
    # Collator
    # --------------------------------------

    collator = VoiceDataCollator(
        processor
    )

    # --------------------------------------
    # DataLoaders
    # --------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        collate_fn=collator,
    )

    return (
        train_loader,
        validation_loader,
        test_loader,
    )


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "Pretrained Voice DataLoader"
    )

    print(
        "=============================="
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders()

    print(
        "\nTrain Batches      :",
        len(train_loader),
    )

    print(
        "Validation Batches :",
        len(validation_loader),
    )

    print(
        "Test Batches       :",
        len(test_loader),
    )

    # --------------------------------------
    # Get one training batch
    # --------------------------------------

    print(
        "\nLoading first training batch..."
    )

    batch = next(
        iter(train_loader)
    )

    print(
        "\n=============================="
    )

    print(
        "Batch Information"
    )

    print(
        "=============================="
    )

    print(
        "\nInput Values Shape :",
        batch[
            "input_values"
        ].shape,
    )

    print(
        "Attention Mask Shape :",
        batch[
            "attention_mask"
        ].shape,
    )

    print(
        "Labels Shape :",
        batch[
            "labels"
        ].shape,
    )

    print(
        "Labels :",
        batch[
            "labels"
        ],
    )

    print(
        "\nDataLoader Test Successful!"
    )