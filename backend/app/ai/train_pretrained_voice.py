from pathlib import Path

import torch
import torch.nn as nn

from app.ai.pretrained_voice_distress import (
    PretrainedVoiceDistressModel,
)

from app.ai.pretrained_voice_dataloader import (
    create_dataloaders,
)


# ==========================================
# Configuration
# ==========================================

# Smoke-test ke liye 1 epoch.
# Successful hone ke baad final training ke
# liye inhe increase karenge.
STAGE1_EPOCHS = 1
STAGE2_EPOCHS = 1

STAGE1_LEARNING_RATE = 1e-4

# Fine-tuning ke time backbone ko
# bahut small learning rate dena safer hai.
STAGE2_LEARNING_RATE = 1e-5

WEIGHT_DECAY = 0.01

UNFREEZE_LAYERS = 4

# Class weights
NORMAL_WEIGHT = 1.0
DISTRESS_WEIGHT = 4.0


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_DIR = BASE_DIR / "trained_models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# IMPORTANT:
# Existing checkpoints ko overwrite NAHI karna.

BEST_MODEL_PATH = (
    MODEL_DIR
    / "pretrained_voice_distress_finetuned_best.pth"
)

LAST_MODEL_PATH = (
    MODEL_DIR
    / "pretrained_voice_distress_finetuned_last.pth"
)


# ==========================================
# Device
# ==========================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ==========================================
# Metrics
# ==========================================

def calculate_metrics(
    true_labels,
    predicted_labels,
):

    tp = 0
    tn = 0
    fp = 0
    fn = 0

    for actual, predicted in zip(
        true_labels,
        predicted_labels,
    ):

        if actual == 1 and predicted == 1:
            tp += 1

        elif actual == 0 and predicted == 0:
            tn += 1

        elif actual == 0 and predicted == 1:
            fp += 1

        elif actual == 1 and predicted == 0:
            fn += 1

    total = tp + tn + fp + fn

    accuracy = (
        (tp + tn) / total
        if total > 0
        else 0.0
    )

    precision = (
        tp / (tp + fp)
        if (tp + fp) > 0
        else 0.0
    )

    recall = (
        tp / (tp + fn)
        if (tp + fn) > 0
        else 0.0
    )

    if precision + recall > 0:

        f1 = (
            2
            * precision
            * recall
            / (precision + recall)
        )

    else:

        f1 = 0.0

    return {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "tn": tn,
        "fp": fp,
        "fn": fn,
    }


# ==========================================
# Train One Epoch
# ==========================================

def train_one_epoch(
    model,
    data_loader,
    optimizer,
    criterion,
):

    model.train()

    total_loss = 0.0
    total_samples = 0

    true_labels = []
    predicted_labels = []

    for batch_index, batch in enumerate(
        data_loader,
        start=1,
    ):

        input_values = (
            batch["input_values"]
            .to(DEVICE)
        )

        attention_mask = (
            batch["attention_mask"]
            .to(DEVICE)
        )

        labels = (
            batch["labels"]
            .to(DEVICE)
        )

        optimizer.zero_grad()

        logits = model(
            input_values,
            attention_mask,
        )

        loss = criterion(
            logits,
            labels,
        )

        loss.backward()

        torch.nn.utils.clip_grad_norm_(
            model.parameters(),
            max_norm=1.0,
        )

        optimizer.step()

        predictions = torch.argmax(
            logits,
            dim=1,
        )

        true_labels.extend(
            labels.detach()
            .cpu()
            .tolist()
        )

        predicted_labels.extend(
            predictions.detach()
            .cpu()
            .tolist()
        )

        total_loss += (
            loss.item()
            * labels.size(0)
        )

        total_samples += labels.size(0)

        if (
            batch_index % 10 == 0
            or batch_index == len(data_loader)
        ):

            print(
                f"Batch "
                f"{batch_index}/"
                f"{len(data_loader)} "
                f"| Loss: "
                f"{loss.item():.4f}"
            )

    average_loss = (
        total_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
    )

    return average_loss, metrics


# ==========================================
# Validation
# ==========================================

def evaluate(
    model,
    data_loader,
    criterion,
):

    model.eval()

    total_loss = 0.0
    total_samples = 0

    true_labels = []
    predicted_labels = []

    with torch.inference_mode():

        for batch in data_loader:

            input_values = (
                batch["input_values"]
                .to(DEVICE)
            )

            attention_mask = (
                batch["attention_mask"]
                .to(DEVICE)
            )

            labels = (
                batch["labels"]
                .to(DEVICE)
            )

            logits = model(
                input_values,
                attention_mask,
            )

            loss = criterion(
                logits,
                labels,
            )

            predictions = torch.argmax(
                logits,
                dim=1,
            )

            true_labels.extend(
                labels.cpu().tolist()
            )

            predicted_labels.extend(
                predictions.cpu().tolist()
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            total_samples += labels.size(0)

    average_loss = (
        total_loss / total_samples
        if total_samples > 0
        else 0.0
    )

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
    )

    return average_loss, metrics


# ==========================================
# Checkpoint
# ==========================================

def save_checkpoint(
    path,
    epoch,
    stage,
    model,
    optimizer,
    metrics,
):

    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "model_state_dict":
                model.state_dict(),
            "optimizer_state_dict":
                optimizer.state_dict(),
            "validation_accuracy":
                metrics["accuracy"],
            "validation_precision":
                metrics["precision"],
            "validation_recall":
                metrics["recall"],
            "validation_f1":
                metrics["f1"],
        },
        path,
    )


# ==========================================
# Print Results
# ==========================================

def print_results(
    title,
    loss,
    metrics,
):

    print(
        "\n=============================="
    )

    print(title)

    print(
        "=============================="
    )

    print(
        f"Loss      : {loss:.4f}"
    )

    print(
        f"Accuracy  : "
        f"{metrics['accuracy'] * 100:.2f}%"
    )

    print(
        f"Precision : "
        f"{metrics['precision'] * 100:.2f}%"
    )

    print(
        f"Recall    : "
        f"{metrics['recall'] * 100:.2f}%"
    )

    print(
        f"F1        : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    print(
        "\nConfusion Matrix:"
    )

    print(
        f"TN: {metrics['tn']}"
    )

    print(
        f"FP: {metrics['fp']}"
    )

    print(
        f"FN: {metrics['fn']}"
    )

    print(
        f"TP: {metrics['tp']}"
    )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    print(
        "\n======================================"
    )

    print(
        "GuardAI Voice Fine-Tuning"
    )

    print(
        "======================================"
    )

    print(
        f"\nDevice: {DEVICE}"
    )

    print(
        f"Stage 1 Epochs: {STAGE1_EPOCHS}"
    )

    print(
        f"Stage 2 Epochs: {STAGE2_EPOCHS}"
    )

    print(
        f"Unfreeze Layers: {UNFREEZE_LAYERS}"
    )

    print(
        "\nNew checkpoints:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        LAST_MODEL_PATH
    )


    # ======================================
    # DataLoaders
    # ======================================

    print(
        "\n=============================="
    )

    print(
        "Creating DataLoaders..."
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
        f"\nTrain Batches      : "
        f"{len(train_loader)}"
    )

    print(
        f"Validation Batches : "
        f"{len(validation_loader)}"
    )

    print(
        f"Test Batches       : "
        f"{len(test_loader)}"
    )


    # ======================================
    # Model - Stage 1
    # ======================================

    print(
        "\n=============================="
    )

    print(
        "STAGE 1"
    )

    print(
        "Frozen Wav2Vec2 + Classifier"
    )

    print(
        "=============================="
    )

    model = PretrainedVoiceDistressModel(
        fine_tune=False
    ).to(DEVICE)


    # ======================================
    # Parameters
    # ======================================

    total_parameters = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nTotal Parameters     : "
        f"{total_parameters:,}"
    )

    print(
        f"Stage 1 Trainable    : "
        f"{trainable_parameters:,}"
    )


    # ======================================
    # Weighted Loss
    # ======================================

    class_weights = torch.tensor(
        [
            NORMAL_WEIGHT,
            DISTRESS_WEIGHT,
        ],
        dtype=torch.float32,
        device=DEVICE,
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )


    # ======================================
    # Stage 1 Optimizer
    # ======================================

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters(),
        ),
        lr=STAGE1_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # ======================================
    # Stage 1 Training
    # ======================================

    best_f1 = -1.0

    for epoch in range(
        1,
        STAGE1_EPOCHS + 1,
    ):

        print(
            f"\n========== "
            f"STAGE 1 EPOCH "
            f"{epoch}/{STAGE1_EPOCHS} "
            f"=========="
        )

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
            )
        )

        validation_loss, validation_metrics = (
            evaluate(
                model,
                validation_loader,
                criterion,
            )
        )

        print_results(
            "Stage 1 Training",
            train_loss,
            train_metrics,
        )

        print_results(
            "Stage 1 Validation",
            validation_loss,
            validation_metrics,
        )

        save_checkpoint(
            LAST_MODEL_PATH,
            epoch,
            "stage1",
            model,
            optimizer,
            validation_metrics,
        )

        if (
            validation_metrics["f1"]
            > best_f1
        ):

            best_f1 = (
                validation_metrics["f1"]
            )

            save_checkpoint(
                BEST_MODEL_PATH,
                epoch,
                "stage1",
                model,
                optimizer,
                validation_metrics,
            )

            print(
                "\nNew best Stage 1 model saved!"
            )


    # ======================================
    # Stage 2
    # ======================================

    print(
        "\n======================================"
    )

    print(
        "STAGE 2"
    )

    print(
        "Fine-Tuning Last Wav2Vec2 Layers"
    )

    print(
        "======================================"
    )


    # Unfreeze last 4 layers
    model.unfreeze_last_layers(
        UNFREEZE_LAYERS
    )


    # ======================================
    # Stage 2 Trainable Parameters
    # ======================================

    trainable_parameters = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        f"\nStage 2 Trainable Parameters: "
        f"{trainable_parameters:,}"
    )


    # ======================================
    # New Optimizer
    # ======================================

    optimizer = torch.optim.AdamW(
        filter(
            lambda parameter:
                parameter.requires_grad,
            model.parameters(),
        ),
        lr=STAGE2_LEARNING_RATE,
        weight_decay=WEIGHT_DECAY,
    )


    # ======================================
    # Stage 2 Training
    # ======================================

    for epoch in range(
        1,
        STAGE2_EPOCHS + 1,
    ):

        print(
            f"\n========== "
            f"STAGE 2 EPOCH "
            f"{epoch}/{STAGE2_EPOCHS} "
            f"=========="
        )

        train_loss, train_metrics = (
            train_one_epoch(
                model,
                train_loader,
                optimizer,
                criterion,
            )
        )

        validation_loss, validation_metrics = (
            evaluate(
                model,
                validation_loader,
                criterion,
            )
        )

        print_results(
            "Stage 2 Training",
            train_loss,
            train_metrics,
        )

        print_results(
            "Stage 2 Validation",
            validation_loss,
            validation_metrics,
        )

        save_checkpoint(
            LAST_MODEL_PATH,
            epoch,
            "stage2",
            model,
            optimizer,
            validation_metrics,
        )

        if (
            validation_metrics["f1"]
            > best_f1
        ):

            best_f1 = (
                validation_metrics["f1"]
            )

            save_checkpoint(
                BEST_MODEL_PATH,
                epoch,
                "stage2",
                model,
                optimizer,
                validation_metrics,
            )

            print(
                "\nNew best fine-tuned model saved!"
            )


    # ======================================
    # IMPORTANT
    # ======================================
    #
    # Test set is NOT used during training
    # or model selection.
    #
    # We evaluate it separately after the
    # smoke test.
    # ======================================

    print(
        "\n======================================"
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "======================================"
    )

    print(
        f"\nBest Validation F1: "
        f"{best_f1 * 100:.2f}%"
    )

    print(
        "\nBest Model:"
    )

    print(
        BEST_MODEL_PATH
    )

    print(
        "\nLast Model:"
    )

    print(
        LAST_MODEL_PATH
    )

    print(
        "\nNext step:"
    )

    print(
        "Evaluate the new model on the "
        "unseen TEST set."
    )