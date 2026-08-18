import torch
import torch.nn as nn

from app.ai.focal_loss import (
    FocalLoss,
)

from torch.utils.data import (
    DataLoader,
    Subset,
    WeightedRandomSampler,
)

from app.ai.voice_dataset_v3 import (
    create_dataset,
    get_actor_files,
)

from app.ai.voice_model_v3 import (
    VoiceDistressModelV3,
)

from app.ai.trainer_v3 import (
    TrainerV3,
)

# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 32

EPOCHS = 40

LEARNING_RATE = 3e-4

# ==========================================
# Dataset
# ==========================================

dataset = create_dataset()

samples = dataset.samples

path_to_index = {
    sample.path: index
    for index, sample in enumerate(samples)
}

# ==========================================
# Speaker Independent Split
# ==========================================

train_files = get_actor_files(
    range(1, 19)
)

validation_files = get_actor_files(
    [19, 20, 21]
)

test_files = get_actor_files(
    [22, 23, 24]
)

train_indices = [
    path_to_index[file]
    for file in train_files
]

validation_indices = [
    path_to_index[file]
    for file in validation_files
]

test_indices = [
    path_to_index[file]
    for file in test_files
]

train_dataset = Subset(
    dataset,
    train_indices,
)

train_labels = []

for index in train_indices:

    sample = dataset.samples[index]

    if sample.label == "normal":
        train_labels.append(0)
    else:
        train_labels.append(1)

normal_count = train_labels.count(0)
distress_count = train_labels.count(1)

class_weights = [
    1 / normal_count,
    1 / distress_count,
]

sample_weights = [
    class_weights[label]
    for label in train_labels
]

train_sampler = WeightedRandomSampler(
    weights=sample_weights,
    num_samples=len(sample_weights),
    replacement=True,
)

validation_dataset = Subset(
    dataset,
    validation_indices,
)

test_dataset = Subset(
    dataset,
    test_indices,
)


# ==========================================
# Data Loaders
# ==========================================

train_loader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    sampler=train_sampler,
    shuffle=False,
    num_workers=0,
    drop_last=True,
)

validation_loader = DataLoader(
    validation_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

print("\n==============================")
print("Voice Dataset Split")
print("==============================")
print(f"Train Samples      : {len(train_dataset)}")
print(f"Validation Samples : {len(validation_dataset)}")
print(f"Test Samples       : {len(test_dataset)}")



# ==========================================
# Weighted Loss
# ==========================================

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

normal_samples = 1248
distress_samples = 192

weights = torch.tensor(
    [
        1.0,
        normal_samples / distress_samples,
    ],
    dtype=torch.float32,
    device=device,
)

print("\n==============================")
print("Class Weights")
print("==============================")
print(f"Normal Weight   : {weights[0]:.2f}")
print(f"Distress Weight : {weights[1]:.2f}")

criterion = FocalLoss(
    alpha=weights,
    gamma=2.0,
)
# ==========================================
# Model
# ==========================================

model = VoiceDistressModelV3()

trainer = TrainerV3(
    model=model,
    train_loader=train_loader,
    val_loader=validation_loader,
    test_loader=test_loader,
    learning_rate=LEARNING_RATE,
    epochs=EPOCHS,
    criterion=criterion,
    best_model_name="distress_model_v3.pth",
    last_model_name="distress_last_v3.pth",
    log_dir="runs/voice_v3",
    training_name="Voice Distress V3 Training",
)


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    trainer.fit()