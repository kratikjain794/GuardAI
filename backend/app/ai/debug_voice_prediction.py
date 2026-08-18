from pathlib import Path

import torch

from app.ai.voice_dataset_v3 import create_dataset
from app.ai.voice_model_v3 import VoiceDistressModelV3
from app.ai.checkpoint import load_checkpoint


# ==========================================
# Model Path
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[2]

MODEL_PATH = (
    BASE_DIR
    / "trained_models"
    / "distress_model_v3.pth"
)


# ==========================================
# Load Dataset
# ==========================================

dataset = create_dataset()


# ==========================================
# Load Model
# ==========================================

model = VoiceDistressModelV3()

load_checkpoint(
    MODEL_PATH,
    model,
)

model.eval()


# ==========================================
# Labels
# ==========================================

index_to_label = {
    0: "normal",
    1: "distress",
}


print("\n==============================")
print("DEBUG DISTRESS PREDICTION")
print("==============================")

count = 0

for i, sample in enumerate(dataset.samples):

    # Sirf distress samples test karo
    if sample.label != "distress":
        continue

    features, label = dataset[i]

    with torch.no_grad():

        output = model(
            features.unsqueeze(0)
        )

        probability = torch.softmax(
            output,
            dim=1,
        )

        prediction = torch.argmax(
            probability,
            dim=1,
        ).item()

    print("\n--------------------------------")

    print(
        "File :",
        sample.path.name,
    )

    print(
        "Ground Truth :",
        sample.label,
    )

    print(
        "Prediction :",
        index_to_label[prediction],
    )

    print(
        f"Normal Probability   : {probability[0][0].item():.4f}"
    )

    print(
        f"Distress Probability : {probability[0][1].item():.4f}"
    )

    count += 1

    # Sirf pehle 20 distress samples
    if count == 20:
        break