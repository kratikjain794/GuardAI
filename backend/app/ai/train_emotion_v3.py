import torch

from torch.utils.data import (
    DataLoader,
    random_split,
)

from app.ai.emotion_dataset_v3 import (
    create_dataset,
)

from app.ai.emotion_model_v3 import (
    EmotionCNNBiLSTMV3,
)

from app.ai.trainer_v3 import (
    TrainerV3,
)


# ==========================================
# Configuration
# ==========================================

BATCH_SIZE = 32

EPOCHS = 40

LEARNING_RATE = 1e-3

RANDOM_SEED = 42


# ==========================================
# Dataset
# ==========================================

dataset = create_dataset()

total_size = len(dataset)

train_size = int(total_size * 0.8)

val_size = int(total_size * 0.1)

test_size = total_size - train_size - val_size


generator = torch.Generator().manual_seed(
    RANDOM_SEED
)

train_dataset, val_dataset, test_dataset = random_split(

    dataset,

    [
        train_size,
        val_size,
        test_size,
    ],

    generator=generator,

)


# ==========================================
# DataLoaders
# ==========================================

train_loader = DataLoader(

    train_dataset,

    batch_size=BATCH_SIZE,

    shuffle=True,

    drop_last=True,

    num_workers=0,

    pin_memory=torch.cuda.is_available(),

)

val_loader = DataLoader(

    val_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    drop_last=True,

    num_workers=0,

    pin_memory=torch.cuda.is_available(),

)

test_loader = DataLoader(

    test_dataset,

    batch_size=BATCH_SIZE,

    shuffle=False,

    drop_last=False,

    num_workers=0,

    pin_memory=torch.cuda.is_available(),

)


# ==========================================
# Model
# ==========================================

model = EmotionCNNBiLSTMV3()


# ==========================================
# Trainer
# ==========================================

trainer = TrainerV3(

    model=model,

    train_loader=train_loader,

    val_loader=val_loader,

    test_loader=test_loader,

    learning_rate=LEARNING_RATE,

    epochs=EPOCHS,

)


# ==========================================
# Main
# ==========================================

if __name__ == "__main__":

    trainer.fit()