import torch
from torch import nn

from app.utils.constants import N_MFCC


class EmotionCNNLSTM(nn.Module):
    """
    CNN + Bidirectional LSTM model
    for speech emotion recognition.

    Input:
        [batch_size, N_MFCC, frames]

    Output:
        [batch_size, num_classes]
    """

    def __init__(
        self,
        num_classes: int = 8,
    ):
        super().__init__()

        # -----------------------------
        # CNN Feature Extraction
        # -----------------------------

        self.cnn = nn.Sequential(
            nn.Conv1d(
                in_channels=N_MFCC,
                out_channels=64,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=64,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.MaxPool1d(kernel_size=2),

            nn.Conv1d(
                in_channels=128,
                out_channels=128,
                kernel_size=3,
                padding=1,
            ),
            nn.BatchNorm1d(128),
            nn.ReLU(),
        )

        # -----------------------------
        # BiLSTM
        # -----------------------------

        self.lstm = nn.LSTM(
            input_size=128,
            hidden_size=128,
            num_layers=1,
            batch_first=True,
            bidirectional=True,
        )

        # -----------------------------
        # Classification
        # -----------------------------

        self.dropout = nn.Dropout(
            p=0.3
        )

        self.classifier = nn.Linear(
            in_features=256,
            out_features=num_classes,
        )

    def forward(
        self,
        x: torch.Tensor,
    ) -> torch.Tensor:

        # x:
        # [batch, 40, 94]

        x = self.cnn(x)

        # CNN output:
        # [batch, 128, frames]

        # LSTM needs:
        # [batch, frames, features]

        x = x.transpose(1, 2)

        lstm_output, _ = self.lstm(x)

        # Use final time step
        x = lstm_output[:, -1, :]

        x = self.dropout(x)

        logits = self.classifier(x)

        return logits