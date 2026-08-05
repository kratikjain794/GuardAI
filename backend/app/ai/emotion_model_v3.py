import torch
import torch.nn as nn


class EmotionCNNBiLSTMV3(nn.Module):
    """
    GuardIA Emotion Model V3

    Input:
        (Batch, 40, 94)

    Output:
        (Batch, 7)
    """

    def __init__(
        self,
        num_classes: int = 7,
    ):
        super().__init__()

        # ==========================================
        # CNN Feature Extractor
        # ==========================================

        self.features = nn.Sequential(

            nn.Conv2d(
                in_channels=1,
                out_channels=32,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(32),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout(0.20),

            # ------------------------------

            nn.Conv2d(
                32,
                64,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(64),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout(0.25),

            # ------------------------------

            nn.Conv2d(
                64,
                128,
                kernel_size=3,
                padding=1,
            ),

            nn.BatchNorm2d(128),

            nn.ReLU(inplace=True),

            nn.MaxPool2d(2),

            nn.Dropout(0.30),
        )

        # ==========================================
        # Automatically calculate LSTM input size
        # ==========================================

        with torch.no_grad():

            dummy = torch.zeros(
                1,
                1,
                40,
                94,
            )

            dummy = self.features(dummy)

            channels = dummy.size(1)

            height = dummy.size(2)

            width = dummy.size(3)

            lstm_input_size = (
                channels * height
            )

        # ==========================================
        # BiLSTM
        # ==========================================

        self.lstm = nn.LSTM(

            input_size=lstm_input_size,

            hidden_size=256,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.30,
        )

        # ==========================================
        # Attention
        # ==========================================

        self.attention = nn.Sequential(

            nn.Linear(
                512,
                128,
            ),

            nn.Tanh(),

            nn.Linear(
                128,
                1,
            ),
        )

        # ==========================================
        # Classifier
        # ==========================================

        self.classifier = nn.Sequential(

            nn.Linear(
                512,
                256,
            ),

            nn.BatchNorm1d(256),

            nn.ReLU(inplace=True),

            nn.Dropout(0.50),

            # ------------------------------

            nn.Linear(
                256,
                128,
            ),

            nn.BatchNorm1d(128),

            nn.ReLU(inplace=True),

            nn.Dropout(0.30),

            # ------------------------------

            nn.Linear(
                128,
                num_classes,
            ),
        )

    # ==========================================
    # Forward
    # ==========================================

    def forward(
        self,
        x,
    ):

        # Input:
        # (B,40,94)

        x = x.unsqueeze(1)

        # (B,1,40,94)

        x = self.features(x)

        # (B,C,H,W)

        batch = x.size(0)

        channels = x.size(1)

        height = x.size(2)

        width = x.size(3)

        x = x.permute(
            0,
            3,
            1,
            2,
        )

        # (B,W,C,H)

        x = x.reshape(
            batch,
            width,
            channels * height,
        )

        # (B,Time,Features)

        lstm_output, _ = self.lstm(x)

        # ======================================
        # Attention
        # ======================================

        attention_scores = self.attention(
            lstm_output
        )

        attention_scores = torch.softmax(
            attention_scores,
            dim=1,
        )

        context = torch.sum(

            attention_scores
            * lstm_output,

            dim=1,
        )

        # ======================================
        # Classification
        # ======================================

        output = self.classifier(
            context
        )

        return output


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    model = EmotionCNNBiLSTMV3()

    print(model)

    sample = torch.randn(
        8,
        40,
        94,
    )

    output = model(sample)

    print("\n==============================")

    print("GuardIA Emotion Model V3")

    print("==============================")

    print(
        "Input Shape :",
        sample.shape,
    )

    print(
        "Output Shape:",
        output.shape,
    )

    print(
        "\nNumber of Parameters:",
        f"{sum(p.numel() for p in model.parameters()):,}",
    )