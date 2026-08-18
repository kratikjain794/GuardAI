from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
)

from app.ai.voice_dataset_v3 import (
    create_dataset,
    get_actor_files,
)

from app.ai.voice_model_v3 import (
    VoiceDistressModelV3,
)


# ==========================================
# Configuration
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "trained_models"
    / "distress_last_v3.pth"
)

TEST_ACTORS = [22, 23, 24]

BATCH_SIZE = 32


# ==========================================
# Load Model
# ==========================================

def load_model():

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )

    print("\n==============================")
    print("Voice Distress Model Testing")
    print("==============================")

    print(
        f"Device : {device}"
    )

    print(
        f"Model  : {MODEL_PATH}"
    )

    if not MODEL_PATH.exists():

        raise FileNotFoundError(
            f"Model not found:\n{MODEL_PATH}"
        )

    model = VoiceDistressModelV3().to(
        device
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device,
    )

    if isinstance(checkpoint, dict):

        if "model_state_dict" in checkpoint:

            model.load_state_dict(
                checkpoint[
                    "model_state_dict"
                ]
            )

        else:

            model.load_state_dict(
                checkpoint
            )

    else:

        model.load_state_dict(
            checkpoint
        )

    model.eval()

    print(
        "\nModel Loaded Successfully!"
    )

    return model, device


# ==========================================
# Test
# ==========================================

def test_model():

    model, device = load_model()

    # -----------------------------
    # Complete Dataset
    # -----------------------------

    dataset = create_dataset()

    samples = dataset.samples

    path_to_index = {
        sample.path: index
        for index, sample in enumerate(
            samples
        )
    }

    # -----------------------------
    # Test Files
    # -----------------------------

    test_files = get_actor_files(
        TEST_ACTORS
    )

    test_indices = []

    for file_path in test_files:

        if file_path not in path_to_index:

            print(
                f"WARNING: File not found "
                f"in dataset: {file_path}"
            )

            continue

        test_indices.append(
            path_to_index[file_path]
        )

    print("\n==============================")
    print("Test Dataset")
    print("==============================")

    print(
        f"Actors       : {TEST_ACTORS}"
    )

    print(
        f"Test Samples : {len(test_indices)}"
    )

    # -----------------------------
    # Actual / Predicted
    # -----------------------------

    y_true = []
    y_pred = []

    # -----------------------------
    # Threshold
    # -----------------------------

    THRESHOLD = 0.40

    print(
        f"Threshold    : {THRESHOLD}"
    )

    # ======================================
    # Prediction Loop
    # ======================================

    with torch.inference_mode():

        for count, index in enumerate(
            test_indices,
            start=1,
        ):

            features, label = dataset[
                index
            ]

            features = (
                features
                .unsqueeze(0)
                .to(device)
            )

            outputs = model(
                features
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            distress_probability = (
                probabilities[0][1].item()
            )

            # --------------------------
            # Decision
            # --------------------------

            if (
                distress_probability
                >= THRESHOLD
            ):

                prediction = 1

            else:

                prediction = 0

            actual = label.item()

            y_true.append(
                actual
            )

            y_pred.append(
                prediction
            )

            # --------------------------
            # Progress
            # --------------------------

            if (
                count % 20 == 0
                or count == len(test_indices)
            ):

                print(
                    f"Tested "
                    f"{count}/"
                    f"{len(test_indices)}"
                )

    # ======================================
    # Metrics
    # ======================================

    accuracy = accuracy_score(
        y_true,
        y_pred,
    )

    precision = precision_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    recall = recall_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    f1 = f1_score(
        y_true,
        y_pred,
        zero_division=0,
    )

    matrix = confusion_matrix(
        y_true,
        y_pred,
    )

    # ======================================
    # Results
    # ======================================

    print("\n==============================")
    print("FINAL RESULTS")
    print("==============================")

    print(
        f"Accuracy  : {accuracy * 100:.2f}%"
    )

    print(
        f"Precision : {precision * 100:.2f}%"
    )

    print(
        f"Recall    : {recall * 100:.2f}%"
    )

    print(
        f"F1 Score  : {f1 * 100:.2f}%"
    )

    # ======================================
    # Confusion Matrix
    # ======================================

    print("\n==============================")
    print("CONFUSION MATRIX")
    print("==============================")

    print(
        "                 Predicted"
    )

    print(
        "              Normal  Distress"
    )

    print(
        f"Actual Normal "
        f"   {matrix[0][0]:5d}"
        f"    {matrix[0][1]:5d}"
    )

    print(
        f"Actual Distress"
        f"   {matrix[1][0]:5d}"
        f"    {matrix[1][1]:5d}"
    )

    # ======================================
    # Error Counts
    # ======================================

    false_positives = matrix[0][1]

    false_negatives = matrix[1][0]

    print("\n==============================")
    print("ERROR ANALYSIS")
    print("==============================")

    print(
        f"False Positives : "
        f"{false_positives}"
    )

    print(
        f"False Negatives : "
        f"{false_negatives}"
    )

    # ======================================
    # Classification Report
    # ======================================

    print("\n==============================")
    print("CLASSIFICATION REPORT")
    print("==============================")

    print(
        classification_report(
            y_true,
            y_pred,
            target_names=[
                "normal",
                "distress",
            ],
            zero_division=0,
        )
    )


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    test_model()