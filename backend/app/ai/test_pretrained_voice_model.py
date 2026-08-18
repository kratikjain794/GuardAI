from pathlib import Path

import torch

from app.ai.pretrained_voice_distress import (
    PretrainedVoiceDistressModel,
)

from app.ai.pretrained_voice_dataloader import (
    create_dataloaders,
)


# ==========================================
# Configuration
# ==========================================

BASE_DIR = (
    Path(__file__).resolve().parents[2]
)

MODEL_PATH = (
    BASE_DIR
    / "trained_models"
    / "pretrained_voice_distress_best.pth"
)

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

    total = (
        tp + tn + fp + fn
    )

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

    if (
        precision + recall
    ) > 0:

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
# Main
# ==========================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "Pretrained Voice Model Test"
    )

    print(
        "=============================="
    )

    print(
        f"\nDevice : {DEVICE}"
    )

    print(
        "\nModel :"
    )

    print(
        MODEL_PATH
    )

    # ======================================
    # Check Model
    # ======================================

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"\nModel not found:\n"
            f"{MODEL_PATH}"
        )

    # ======================================
    # Data
    # ======================================

    print(
        "\nCreating test DataLoader..."
    )

    (
        train_loader,
        validation_loader,
        test_loader,
    ) = create_dataloaders()

    print(
        "Test DataLoader ready!"
    )

    # ======================================
    # Model
    # ======================================

    print(
        "\nLoading model..."
    )

    model = (
        PretrainedVoiceDistressModel()
        .to(DEVICE)
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE,
    )

    print(
        "\nCheckpoint Epoch :",
        checkpoint.get(
            "epoch",
            "unknown",
        ),
    )

    print(
        "Validation Accuracy :",
        f"{checkpoint.get('validation_accuracy', 0) * 100:.2f}%",
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ]
    )

    model.eval()

    print(
        "\nModel Loaded Successfully!"
    )

    # ======================================
    # Testing
    # ======================================

    true_labels = []

    predicted_labels = []

    total_samples = 0

    print(
        "\n=============================="
    )

    print(
        "Testing"
    )

    print(
        "=============================="
    )

    with torch.inference_mode():

        for batch_index, batch in enumerate(
            test_loader,
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

            logits = model(
                input_values,
                attention_mask,
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

            total_samples += (
                labels.size(0)
            )

            print(
                f"Tested "
                f"{total_samples}/180"
            )

    # ======================================
    # Metrics
    # ======================================

    metrics = calculate_metrics(
        true_labels,
        predicted_labels,
    )

    print(
        "\n=============================="
    )

    print(
        "Test Results"
    )

    print(
        "=============================="
    )

    print(
        f"\nAccuracy  : "
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
        f"F1 Score  : "
        f"{metrics['f1'] * 100:.2f}%"
    )

    # ======================================
    # Confusion Matrix
    # ======================================

    print(
        "\n=============================="
    )

    print(
        "Confusion Matrix"
    )

    print(
        "=============================="
    )

    print(
        "\n                 Predicted"
    )

    print(
        "              Normal  Distress"
    )

    print(
        f"Actual Normal     "
        f"{metrics['tn']:3d}      "
        f"{metrics['fp']:3d}"
    )

    print(
        f"Actual Distress   "
        f"{metrics['fn']:3d}      "
        f"{metrics['tp']:3d}"
    )

    # ======================================
    # Error Counts
    # ======================================

    print(
        "\n=============================="
    )

    print(
        "Errors"
    )

    print(
        "=============================="
    )

    print(
        f"\nFalse Positives : "
        f"{metrics['fp']}"
    )

    print(
        f"False Negatives : "
        f"{metrics['fn']}"
    )

    # ======================================
    # Class-wise Results
    # ======================================

    normal_total = (
        metrics["tn"]
        + metrics["fp"]
    )

    distress_total = (
        metrics["tp"]
        + metrics["fn"]
    )

    normal_accuracy = (
        metrics["tn"]
        / normal_total
        if normal_total > 0
        else 0.0
    )

    distress_accuracy = (
        metrics["tp"]
        / distress_total
        if distress_total > 0
        else 0.0
    )

    print(
        "\n=============================="
    )

    print(
        "Class-wise Accuracy"
    )

    print(
        "=============================="
    )

    print(
        f"\nNormal   : "
        f"{normal_accuracy * 100:.2f}%"
    )

    print(
        f"Distress : "
        f"{distress_accuracy * 100:.2f}%"
    )

    print(
        "\n=============================="
    )

    print(
        "Test Complete"
    )

    print(
        "=============================="
    )