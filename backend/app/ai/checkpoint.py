from pathlib import Path

import torch


# ==========================================
# Paths
# ==========================================

BASE_DIR = Path(__file__).resolve().parents[3]

MODEL_DIR = BASE_DIR / "trained_models"

MODEL_DIR.mkdir(
    parents=True,
    exist_ok=True,
)


# ==========================================
# Save Checkpoint
# ==========================================

def save_checkpoint(
    path,
    model,
    optimizer,
    epoch,
    best_metric,
):

    torch.save(
        {
            "epoch": epoch,
            "best_metric": best_metric,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
        },
        path,
    )


# ==========================================
# Load Checkpoint
# ==========================================

def load_checkpoint(
    path,
    model,
    optimizer=None,
    device="cpu",
):

    checkpoint = torch.load(
        path,
        map_location=device,
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    if (
        optimizer is not None
        and "optimizer_state_dict"
        in checkpoint
    ):

        optimizer.load_state_dict(
            checkpoint[
                "optimizer_state_dict"
            ]
        )

    return (
        checkpoint["epoch"],
        checkpoint["best_metric"],
    )


# ==========================================
# Check Exists
# ==========================================

def checkpoint_exists(
    path,
):

    return Path(path).exists()



if __name__ == "__main__":

    print(
        "Checkpoint module loaded successfully."
    )

    print(
        "Model Directory:"
    )

    print(MODEL_DIR)