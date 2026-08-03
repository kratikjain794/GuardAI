from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import accuracy_score, f1_score
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.ai.emotion_model import EmotionCNNLSTM
from app.ai.emotion_utils import (
    EMOTION_MAP,
    extract_mfcc,
    get_emotion_from_filename,
)


# ==========================================
# Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = BASE_DIR / "datasets" / "emotion"
MODEL_DIR = BASE_DIR / "trained_models"

# Best model according to validation F1
MODEL_PATH = MODEL_DIR / "emotion_model.pth"

# Latest training checkpoint for resume
LAST_MODEL_PATH = MODEL_DIR / "emotion_last.pth"

BATCH_SIZE = 16
EPOCHS = 30
LEARNING_RATE = 0.001
SEED = 42

# True = resume from emotion_last.pth when available
RESUME_TRAINING = True


# ==========================================
# Reproducibility
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# Labels
# ==========================================

EMOTIONS = list(EMOTION_MAP.values())

LABEL_TO_INDEX = {
    emotion: index
    for index, emotion in enumerate(EMOTIONS)
}


# ==========================================
# Dataset
# ==========================================

class RavdessDataset(Dataset):

    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_path = self.files[index]

        features = extract_mfcc(file_path)

        emotion = get_emotion_from_filename(
            file_path
        )

        label = LABEL_TO_INDEX[emotion]

        features_tensor = torch.from_numpy(
            features
        ).float()

        label_tensor = torch.tensor(
            label,
            dtype=torch.long
        )

        return features_tensor, label_tensor


# ==========================================
# Actor Files
# ==========================================

def get_actor_files(actor_numbers):

    files = []

    for actor_number in actor_numbers:

        actor_folder = (
            DATASET_DIR
            / f"Actor_{actor_number:02d}"
        )

        if not actor_folder.exists():
            raise FileNotFoundError(
                f"Actor folder not found: "
                f"{actor_folder}"
            )

        actor_files = list(
            actor_folder.glob("*.wav")
        )

        if not actor_files:
            raise RuntimeError(
                f"No WAV files found in "
                f"{actor_folder}"
            )

        files.extend(actor_files)

    return sorted(files)


# ==========================================
# Evaluation
# ==========================================

def evaluate(
    model,
    loader,
    criterion,
    device,
):

    model.eval()

    total_loss = 0.0

    predictions = []
    actual_labels = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            total_loss += loss.item()

            predicted = torch.argmax(
                outputs,
                dim=1
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

            actual_labels.extend(
                labels.cpu().tolist()
            )

    average_loss = (
        total_loss / max(len(loader), 1)
    )

    accuracy = accuracy_score(
        actual_labels,
        predictions
    )

    f1 = f1_score(
        actual_labels,
        predictions,
        average="macro",
        zero_division=0
    )

    return average_loss, accuracy, f1


# ==========================================
# Training
# ==========================================

def main():

    print("\n================================")
    print("GuardIA Emotion Model Training")
    print("================================")

    if not DATASET_DIR.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATASET_DIR}"
        )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(f"\nDevice: {device}")

    # --------------------------------------
    # Speaker-independent split
    # --------------------------------------

    train_actors = list(range(1, 19))

    validation_actors = [
        19,
        20,
        21,
    ]

    test_actors = [
        22,
        23,
        24,
    ]

    train_files = get_actor_files(
        train_actors
    )

    validation_files = get_actor_files(
        validation_actors
    )

    test_files = get_actor_files(
        test_actors
    )

    print(
        f"Training samples   : "
        f"{len(train_files)}"
    )

    print(
        f"Validation samples : "
        f"{len(validation_files)}"
    )

    print(
        f"Test samples       : "
        f"{len(test_files)}"
    )

    # --------------------------------------
    # Datasets
    # --------------------------------------

    train_dataset = RavdessDataset(
        train_files
    )

    validation_dataset = RavdessDataset(
        validation_files
    )

    test_dataset = RavdessDataset(
        test_files
    )

    # --------------------------------------
    # DataLoaders
    # --------------------------------------

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )

    validation_loader = DataLoader(
        validation_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    # --------------------------------------
    # Model
    # --------------------------------------

    model = EmotionCNNLSTM(
        num_classes=len(EMOTIONS)
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True
    )

    # ======================================
    # Resume Configuration
    # ======================================

    start_epoch = 1
    best_validation_f1 = -1.0

    if (
        RESUME_TRAINING
        and LAST_MODEL_PATH.exists()
    ):

        print("\n================================")
        print("Previous checkpoint found")
        print("================================")

        checkpoint = torch.load(
            LAST_MODEL_PATH,
            map_location=device
        )

        model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_validation_f1 = (
            checkpoint.get(
                "best_validation_f1",
                -1.0
            )
        )

        print(
            f"Last completed epoch : "
            f"{checkpoint['epoch']}"
        )

        print(
            f"Resume from epoch    : "
            f"{start_epoch}"
        )

        print(
            f"Best validation F1   : "
            f"{best_validation_f1:.4f}"
        )

    else:

        print(
            "\nStarting new training..."
        )

    # ======================================
    # Check if training already completed
    # ======================================

    if start_epoch > EPOCHS:

        print(
            f"\nTraining already completed "
            f"{EPOCHS} epochs."
        )

        print(
            "Skipping training and "
            "running final test..."
        )

    # ======================================
    # Epoch Loop
    # ======================================

    for epoch in range(
        start_epoch,
        EPOCHS + 1
    ):

        model.train()

        total_train_loss = 0.0

        for features, labels in train_loader:

            features = features.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()

            outputs = model(features)

            loss = criterion(
                outputs,
                labels
            )

            loss.backward()

            optimizer.step()

            total_train_loss += (
                loss.item()
            )

        train_loss = (
            total_train_loss
            / max(len(train_loader), 1)
        )

        (
            val_loss,
            val_accuracy,
            val_f1,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        print(
            f"\nEpoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss  : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss    : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy: "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Val F1      : "
            f"{val_f1:.4f}"
        )

        # ==================================
        # Save Best Model
        # ==================================

        if val_f1 > best_validation_f1:

            best_validation_f1 = val_f1

            torch.save(
                {
                    "model_state_dict":
                        model.state_dict(),

                    "label_to_index":
                        LABEL_TO_INDEX,

                    "epoch":
                        epoch,

                    "validation_f1":
                        val_f1,
                },
                MODEL_PATH
            )

            print(
                "Best model saved."
            )

        # ==================================
        # Save Resume Checkpoint
        # ==================================

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "label_to_index":
                    LABEL_TO_INDEX,

                "best_validation_f1":
                    best_validation_f1,

                "train_loss":
                    train_loss,

                "validation_loss":
                    val_loss,

                "validation_accuracy":
                    val_accuracy,

                "validation_f1":
                    val_f1,
            },
            LAST_MODEL_PATH
        )

        print(
            f"Resume checkpoint saved "
            f"at epoch {epoch}."
        )

    # ======================================
    # Final Test
    # ======================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            "Best emotion model was not found. "
            "Training may not have completed "
            "successfully."
        )

    print(
        "\nLoading best model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        test_loss,
        test_accuracy,
        test_f1,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print("\n================================")
    print("FINAL TEST RESULTS")
    print("================================")

    print(
        f"Test Loss     : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test F1 Score : "
        f"{test_f1:.4f}"
    )

    print(
        f"Best Epoch    : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"Best Val F1   : "
        f"{checkpoint['validation_f1']:.4f}"
    )

    print(
        f"\nBest model saved at:\n"
        f"{MODEL_PATH}"
    )

    print(
        f"\nResume checkpoint saved at:\n"
        f"{LAST_MODEL_PATH}"
    )


if __name__ == "__main__":
    main()