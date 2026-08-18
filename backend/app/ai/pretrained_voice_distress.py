import torch
import torch.nn as nn

from transformers import Wav2Vec2Model


class PretrainedVoiceDistressModel(nn.Module):

    def __init__(
        self,
        num_classes=2,
        fine_tune=False,
        unfreeze_layers=4,
    ):

        super().__init__()

        print("\nLoading pretrained Wav2Vec2...")

        self.backbone = Wav2Vec2Model.from_pretrained(
            "facebook/wav2vec2-base"
        )

        hidden_size = (
            self.backbone.config.hidden_size
        )

        # ======================================
        # Freeze / Fine-tune Configuration
        # ======================================

        self.freeze_backbone()

        if fine_tune:
            self.unfreeze_last_layers(
                unfreeze_layers
            )

        # ======================================
        # Classification Head
        # ======================================

        self.classifier = nn.Sequential(

            nn.Linear(
                hidden_size,
                256,
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                256,
                num_classes,
            ),
        )

    # ==========================================
    # Freeze Entire Wav2Vec2
    # ==========================================

    def freeze_backbone(self):

        for param in self.backbone.parameters():
            param.requires_grad = False

    # ==========================================
    # Unfreeze Last N Transformer Layers
    # ==========================================

    def unfreeze_last_layers(
        self,
        num_layers=4,
    ):

        total_layers = len(
            self.backbone.encoder.layers
        )

        num_layers = min(
            num_layers,
            total_layers,
        )

        start_layer = (
            total_layers - num_layers
        )

        print(
            f"\nUnfreezing last "
            f"{num_layers} Wav2Vec2 layers..."
        )

        for layer_index in range(
            start_layer,
            total_layers,
        ):

            for param in (
                self.backbone
                .encoder
                .layers[layer_index]
                .parameters()
            ):

                param.requires_grad = True

        # Also allow the final layer norm
        # to adapt during fine-tuning.

        for param in (
            self.backbone
            .encoder
            .layer_norm
            .parameters()
        ):

            param.requires_grad = True

    # ==========================================
    # Forward
    # ==========================================

    def forward(
        self,
        input_values,
        attention_mask=None,
    ):

        # IMPORTANT:
        # No torch.no_grad() here.
        #
        # When the last Wav2Vec2 layers are
        # unfrozen, gradients must flow through
        # the backbone.

        outputs = self.backbone(
            input_values=input_values,
            attention_mask=attention_mask,
        )

        hidden_states = (
            outputs.last_hidden_state
        )

        # ======================================
        # Attention-Mask Aware Mean Pooling
        # ======================================

        if attention_mask is not None:

            # Wav2Vec2 changes the temporal
            # resolution, so create a mask
            # matching hidden-state length.

            output_length = (
                hidden_states.size(1)
            )

            input_length = (
                attention_mask.size(1)
            )

            if input_length != output_length:

                mask = torch.nn.functional.interpolate(
                    attention_mask.float()
                    .unsqueeze(1),

                    size=output_length,

                    mode="nearest",
                ).squeeze(1)

                mask = mask.to(
                    hidden_states.dtype
                )

            else:

                mask = attention_mask.to(
                    hidden_states.dtype
                )

            mask = mask.unsqueeze(-1)

            masked_hidden = (
                hidden_states * mask
            )

            pooled = (
                masked_hidden.sum(dim=1)
                /
                mask.sum(dim=1).clamp(
                    min=1e-6
                )
            )

        else:

            pooled = hidden_states.mean(
                dim=1
            )

        # ======================================
        # Classification
        # ======================================

        logits = self.classifier(
            pooled
        )

        return logits


# ==========================================
# Testing
# ==========================================

if __name__ == "__main__":

    print(
        "\n=============================="
    )

    print(
        "Pretrained Voice Distress Model"
    )

    print(
        "=============================="
    )

    # --------------------------------------
    # Baseline mode
    # --------------------------------------

    print(
        "\nCreating frozen baseline..."
    )

    model = (
        PretrainedVoiceDistressModel(
            fine_tune=False
        )
    )

    total_params = sum(
        p.numel()
        for p in model.parameters()
    )

    trainable_params = sum(
        p.numel()
        for p in model.parameters()
        if p.requires_grad
    )

    print(
        "\nTotal Parameters :",
        f"{total_params:,}",
    )

    print(
        "Trainable Parameters :",
        f"{trainable_params:,}",
    )

    print(
        "Frozen Parameters :",
        f"{total_params - trainable_params:,}",
    )

    # --------------------------------------
    # Fine-tuning mode
    # --------------------------------------

    print(
        "\nCreating fine-tuning model..."
    )

    fine_tune_model = (
        PretrainedVoiceDistressModel(
            fine_tune=True,
            unfreeze_layers=4,
        )
    )

    fine_tune_trainable = sum(
        p.numel()
        for p in fine_tune_model.parameters()
        if p.requires_grad
    )

    print(
        "\nFine-tuning Trainable Parameters :",
        f"{fine_tune_trainable:,}",
    )

    # --------------------------------------
    # Forward Test
    # --------------------------------------

    sample = torch.randn(
        2,
        16000,
    )

    attention_mask = torch.ones(
        2,
        16000,
        dtype=torch.long,
    )

    output = fine_tune_model(
        sample,
        attention_mask,
    )

    print(
        "\nInput Shape :",
        sample.shape,
    )

    print(
        "Output Shape :",
        output.shape,
    )

    print(
        "\nModel Test Successful!"
    )