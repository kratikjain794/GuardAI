from pathlib import Path
import random

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from torch import nn
from torch.utils.data import DataLoader, Dataset

from app.ai.emotion_model import EmotionCNNLSTM
from app.ai.emotion_utils import (
    extract_mfcc,
    get_emotion_from_filename,
)


# ==========================================
# Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

DATASET_DIR = BASE_DIR / "datasets" / "emotion"
MODEL_DIR = BASE_DIR / "trained_models"

MODEL_PATH = MODEL_DIR / "distress_model_v2.pth"
LAST_MODEL_PATH = MODEL_DIR / "distress_last_v2.pth"

BATCH_SIZE = 16
EPOCHS = 40
LEARNING_RATE = 0.001

WEIGHT_DECAY = 1e-4

EARLY_STOPPING_PATIENCE = 8

SEED = 42

RESUME_TRAINING = True


# ==========================================
# Labels
# ==========================================

DISTRESS_EMOTIONS = {
    "angry",
    "fearful",
}

LABEL_TO_INDEX = {
    "normal": 0,
    "distress": 1,
}


# ==========================================
# Reproducibility
# ==========================================

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)


# ==========================================
# Label
# ==========================================

def get_distress_label(file_path):

    emotion = get_emotion_from_filename(
        file_path
    )

    return (
        1
        if emotion in DISTRESS_EMOTIONS
        else 0
    )


# ==========================================
# MFCC Augmentation
# ==========================================

def augment_mfcc(
    mfcc: np.ndarray,
) -> np.ndarray:

    augmented = mfcc.copy()

    # Random Gaussian noise
    if random.random() < 0.5:

        noise = np.random.normal(
            0,
            0.02,
            augmented.shape,
        ).astype(np.float32)

        augmented += noise

    # Random time masking
    if random.random() < 0.3:

        frames = augmented.shape[1]

        mask_size = random.randint(
            2,
            min(8, frames),
        )

        start = random.randint(
            0,
            frames - mask_size,
        )

        augmented[
            :,
            start:start + mask_size
        ] = 0

    # Random feature masking
    if random.random() < 0.2:

        feature_count = augmented.shape[0]

        mask_size = random.randint(
            1,
            min(4, feature_count),
        )

        start = random.randint(
            0,
            feature_count - mask_size,
        )

        augmented[
            start:start + mask_size,
            :
        ] = 0

    return augmented.astype(
        np.float32
    )


# ==========================================
# Dataset
# ==========================================

class DistressDataset(Dataset):

    def __init__(
        self,
        files,
        augment=False,
    ):

        self.files = files
        self.augment = augment

    def __len__(self):
        return len(self.files)

    def __getitem__(self, index):

        file_path = self.files[index]

        features = extract_mfcc(
            file_path
        )

        if self.augment:
            features = augment_mfcc(
                features
            )

        label = get_distress_label(
            file_path
        )

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
# Actor Files
# ==========================================

def get_actor_files(
    actor_numbers,
):

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

        actor_files = sorted(
            actor_folder.glob("*.wav")
        )

        if not actor_files:
            raise RuntimeError(
                f"No WAV files found in "
                f"{actor_folder}"
            )

        files.extend(actor_files)

    return files


# ==========================================
# Class Weights
# ==========================================

def calculate_class_weights(
    files,
    device,
):

    labels = [
        get_distress_label(file)
        for file in files
    ]

    normal_count = labels.count(0)
    distress_count = labels.count(1)

    total = len(labels)

    normal_weight = (
        total / (2 * normal_count)
    )

    distress_weight = (
        total / (2 * distress_count)
    )

    print("\nClass distribution:")

    print(
        f"Normal   : {normal_count}"
    )

    print(
        f"Distress : {distress_count}"
    )

    print(
        f"Normal weight   : "
        f"{normal_weight:.4f}"
    )

    print(
        f"Distress weight : "
        f"{distress_weight:.4f}"
    )

    return torch.tensor(
        [
            normal_weight,
            distress_weight,
        ],
        dtype=torch.float32,
        device=device,
    )


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
    actual = []

    with torch.no_grad():

        for features, labels in loader:

            features = features.to(device)
            labels = labels.to(device)

            outputs = model(features)

            loss = criterion(
                outputs,
                labels,
            )

            total_loss += loss.item()

            predicted = torch.argmax(
                outputs,
                dim=1,
            )

            predictions.extend(
                predicted.cpu().tolist()
            )

            actual.extend(
                labels.cpu().tolist()
            )

    loss = (
        total_loss
        / max(len(loader), 1)
    )

    accuracy = accuracy_score(
        actual,
        predictions,
    )

    precision = precision_score(
        actual,
        predictions,
        zero_division=0,
    )

    recall = recall_score(
        actual,
        predictions,
        zero_division=0,
    )

    f1 = f1_score(
        actual,
        predictions,
        zero_division=0,
    )

    return (
        loss,
        accuracy,
        precision,
        recall,
        f1,
    )


# ==========================================
# Main
# ==========================================

def main():

    print(
        "\n================================"
    )
    print(
        "GuardIA Distress Model V2"
    )
    print(
        "================================"
    )

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print(
        f"\nDevice: {device}"
    )

    # Speaker-independent split

    train_files = get_actor_files(
        range(1, 19)
    )

    validation_files = get_actor_files(
        [19, 20, 21]
    )

    test_files = get_actor_files(
        [22, 23, 24]
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

    # ======================================
    # Datasets
    # ======================================

    train_dataset = DistressDataset(
        train_files,
        augment=True,
    )

    validation_dataset = DistressDataset(
        validation_files,
        augment=False,
    )

    test_dataset = DistressDataset(
        test_files,
        augment=False,
    )

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

    # ======================================
    # Model
    # ======================================

    model = EmotionCNNLSTM(
        num_classes=2
    ).to(device)

    class_weights = (
        calculate_class_weights(
            train_files,
            device,
        )
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
        )
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    start_epoch = 1
    best_validation_f1 = -1.0
    epochs_without_improvement = 0

    # ======================================
    # Resume
    # ======================================

    if (
        RESUME_TRAINING
        and LAST_MODEL_PATH.exists()
    ):

        print(
            "\nV2 checkpoint found."
        )

        checkpoint = torch.load(
            LAST_MODEL_PATH,
            map_location=device,
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

        if "scheduler_state_dict" in checkpoint:

            scheduler.load_state_dict(
                checkpoint[
                    "scheduler_state_dict"
                ]
            )

        start_epoch = (
            checkpoint["epoch"] + 1
        )

        best_validation_f1 = (
            checkpoint.get(
                "best_validation_f1",
                -1.0,
            )
        )

        epochs_without_improvement = (
            checkpoint.get(
                "epochs_without_improvement",
                0,
            )
        )

        print(
            f"Resume from epoch: "
            f"{start_epoch}"
        )

        print(
            f"Best F1: "
            f"{best_validation_f1:.4f}"
        )

    else:

        print(
            "\nStarting new V2 training..."
        )

    # ======================================
    # Training
    # ======================================

    for epoch in range(
        start_epoch,
        EPOCHS + 1,
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
                labels,
            )

            loss.backward()

            # Prevent exploding gradients
            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=5.0,
            )

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
            val_precision,
            val_recall,
            val_f1,
        ) = evaluate(
            model,
            validation_loader,
            criterion,
            device,
        )

        scheduler.step(
            val_loss
        )

        current_lr = (
            optimizer
            .param_groups[0]["lr"]
        )

        print(
            f"\nEpoch "
            f"{epoch:02d}/{EPOCHS}"
        )

        print(
            f"Train Loss    : "
            f"{train_loss:.4f}"
        )

        print(
            f"Val Loss      : "
            f"{val_loss:.4f}"
        )

        print(
            f"Val Accuracy  : "
            f"{val_accuracy * 100:.2f}%"
        )

        print(
            f"Val Precision : "
            f"{val_precision:.4f}"
        )

        print(
            f"Val Recall    : "
            f"{val_recall:.4f}"
        )

        print(
            f"Val F1        : "
            f"{val_f1:.4f}"
        )

        print(
            f"Learning Rate : "
            f"{current_lr:.6f}"
        )

        # ==================================
        # Best Model
        # ==================================

        if val_f1 > best_validation_f1:

            best_validation_f1 = val_f1

            epochs_without_improvement = 0

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

                    "validation_recall":
                        val_recall,

                    "validation_precision":
                        val_precision,
                },
                MODEL_PATH,
            )

            print(
                "Best V2 model saved."
            )

        else:

            epochs_without_improvement += 1

        # ==================================
        # Resume Checkpoint
        # ==================================

        torch.save(
            {
                "epoch":
                    epoch,

                "model_state_dict":
                    model.state_dict(),

                "optimizer_state_dict":
                    optimizer.state_dict(),

                "scheduler_state_dict":
                    scheduler.state_dict(),

                "best_validation_f1":
                    best_validation_f1,

                "epochs_without_improvement":
                    epochs_without_improvement,
            },
            LAST_MODEL_PATH,
        )

        print(
            "Resume checkpoint saved."
        )

        # ==================================
        # Early Stopping
        # ==================================

        if (
            epochs_without_improvement
            >= EARLY_STOPPING_PATIENCE
        ):

            print(
                "\nEarly stopping triggered."
            )

            print(
                f"No F1 improvement for "
                f"{EARLY_STOPPING_PATIENCE} "
                f"epochs."
            )

            break

    # ======================================
    # Test Best Model
    # ======================================

    print(
        "\nLoading best V2 model..."
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    (
        test_loss,
        test_accuracy,
        test_precision,
        test_recall,
        test_f1,
    ) = evaluate(
        model,
        test_loader,
        criterion,
        device,
    )

    print(
        "\n================================"
    )
    print(
        "DISTRESS V2 TEST RESULTS"
    )
    print(
        "================================"
    )

    print(
        f"Test Loss      : "
        f"{test_loss:.4f}"
    )

    print(
        f"Test Accuracy  : "
        f"{test_accuracy * 100:.2f}%"
    )

    print(
        f"Test Precision : "
        f"{test_precision:.4f}"
    )

    print(
        f"Test Recall    : "
        f"{test_recall:.4f}"
    )

    print(
        f"Test F1 Score  : "
        f"{test_f1:.4f}"
    )

    print(
        f"Best Epoch     : "
        f"{checkpoint['epoch']}"
    )

    print(
        f"\nV2 model saved at:\n"
        f"{MODEL_PATH}"
    )


if __name__ == "__main__":
    main()