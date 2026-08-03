from pathlib import Path

import torch

from app.ai.emotion_model import EmotionCNNLSTM
from app.ai.emotion_utils import extract_mfcc


BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_PATH = (
    BASE_DIR
    / "trained_models"
    / "emotion_model.pth"
)


class EmotionDetector:
    """
    GuardIA speech emotion detector.

    Loads the trained CNN + BiLSTM model and
    predicts emotion from a WAV audio file.
    """

    def __init__(self):

        self.device = torch.device(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        )

        if not MODEL_PATH.exists():
            raise FileNotFoundError(
                "Emotion model not found. "
                "Run train_emotion.py first."
            )

        checkpoint = torch.load(
            MODEL_PATH,
            map_location=self.device,
        )

        self.label_to_index = checkpoint[
            "label_to_index"
        ]

        self.index_to_label = {
            index: label
            for label, index
            in self.label_to_index.items()
        }

        self.model = EmotionCNNLSTM(
            num_classes=len(
                self.label_to_index
            )
        ).to(self.device)

        self.model.load_state_dict(
            checkpoint[
                "model_state_dict"
            ]
        )

        self.model.eval()

    def predict(
        self,
        audio_path: str | Path,
    ) -> dict:

        audio_path = Path(audio_path)

        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file not found: {audio_path}"
            )

        # MFCC: [40, 94]
        features = extract_mfcc(
            audio_path
        )

        tensor = torch.from_numpy(
            features
        ).float()

        # [40, 94]
        #      ↓
        # [1, 40, 94]
        tensor = tensor.unsqueeze(0)

        tensor = tensor.to(
            self.device
        )

        with torch.no_grad():

            logits = self.model(
                tensor
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )

            confidence, predicted_index = (
                torch.max(
                    probabilities,
                    dim=1
                )
            )

        emotion = self.index_to_label[
            predicted_index.item()
        ]

        confidence_percent = (
            confidence.item() * 100
        )

        return {
            "emotion": emotion,
            "confidence": round(
                confidence_percent,
                2
            ),
        }