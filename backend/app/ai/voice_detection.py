from pathlib import Path

import torch

from app.ai.emotion_model import EmotionCNNLSTM
from app.ai.emotion_utils import extract_mfcc


BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    BASE_DIR
    / "trained_models"
    / "distress_model_v2.pth"
)


class VoiceDistressDetector:

    def __init__(self):

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                f"Distress model not found: {MODEL_PATH}"
            )

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        self.model = EmotionCNNLSTM(
            num_classes=2
        ).to(self.device)

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        self.model.load_state_dict(
            checkpoint["model_state_dict"]
        )

        self.model.eval()

        self.index_to_label = {
            0: "normal",
            1: "distress",
        }

    def predict(self, audio_path):

        features = extract_mfcc(
            audio_path
        )

        features = torch.from_numpy(
            features
        ).float()

        # Add batch dimension
        features = features.unsqueeze(0)

        features = features.to(
            self.device
        )

        with torch.no_grad():

            outputs = self.model(
                features
            )

            probabilities = torch.softmax(
                outputs,
                dim=1,
            )

            confidence, predicted = torch.max(
                probabilities,
                dim=1,
            )

        predicted_index = (
            predicted.item()
        )

        label = self.index_to_label[
            predicted_index
        ]

        return {
            "distress_detected":
                label == "distress",

            "label":
                label,

            "confidence":
                round(
                    confidence.item() * 100,
                    2,
                ),
        }