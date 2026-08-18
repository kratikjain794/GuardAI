import torch

from app.ai.emotion_model_v3 import (
    EmotionCNNBiLSTMV3,
)

# ==========================================
# Voice Distress Model V3
# ==========================================

class VoiceDistressModelV3(
    EmotionCNNBiLSTMV3
):

    def __init__(self):

        super().__init__(
            num_classes=2
        )

# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    model = VoiceDistressModelV3()

    print("\n==============================")
    print("Voice Distress Model V3")
    print("==============================")

    sample = torch.randn(
        8,
        40,
        94,
    )

    output = model(sample)

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