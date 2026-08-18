from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torch.nn.functional as F

from transformers import AutoProcessor

from app.ai.pretrained_voice_distress import (
    PretrainedVoiceDistressModel,
)


# ==========================================
# Configuration
# ==========================================

MODEL_NAME = "facebook/wav2vec2-base"

AUDIO_PATH = (
    Path(__file__).resolve().parents[3]
    / "datasets"
    / "emotion"
    / "Actor_01"
    / "03-01-01-01-01-02-01.wav"
)

TARGET_SAMPLE_RATE = 16000


# ==========================================
# Header
# ==========================================

print("\n==============================")
print("Pretrained Voice Test")
print("==============================")


# ==========================================
# Check Audio
# ==========================================

print(f"\nAudio : {AUDIO_PATH}")

if not AUDIO_PATH.exists():
    raise FileNotFoundError(
        f"Audio file not found:\n{AUDIO_PATH}"
    )


# ==========================================
# Load Processor
# ==========================================

print("\nLoading processor...")

processor = AutoProcessor.from_pretrained(
    MODEL_NAME
)

print(
    "Processor loaded successfully!"
)


# ==========================================
# Load Audio
# ==========================================

print("\nLoading audio...")

audio, sample_rate = sf.read(
    str(AUDIO_PATH),
    dtype="float32",
)

# ------------------------------------------
# Convert NumPy → Torch
# ------------------------------------------

if audio.ndim == 1:

    # Mono
    waveform = torch.from_numpy(
        audio
    ).unsqueeze(0)

else:

    # Stereo / multi-channel
    # Shape: (Samples, Channels)
    # Convert to: (Channels, Samples)

    waveform = torch.from_numpy(
        audio.T
    )

print(
    f"Original Sample Rate : {sample_rate}"
)

print(
    f"Original Shape       : {waveform.shape}"
)


# ==========================================
# Convert Multi-Channel → Mono
# ==========================================

if waveform.size(0) > 1:

    waveform = waveform.mean(
        dim=0,
        keepdim=True,
    )


# ==========================================
# Resample → 16 kHz
# ==========================================

if sample_rate != TARGET_SAMPLE_RATE:

    original_length = waveform.shape[-1]

    target_length = int(
        original_length
        * TARGET_SAMPLE_RATE
        / sample_rate
    )

    waveform = F.interpolate(
        waveform.unsqueeze(0),
        size=target_length,
        mode="linear",
        align_corners=False,
    ).squeeze(0)

    sample_rate = TARGET_SAMPLE_RATE


print(
    f"Final Sample Rate    : {sample_rate}"
)

print(
    f"Final Shape          : {waveform.shape}"
)


# ==========================================
# Convert Audio for Wav2Vec2
# ==========================================

audio_array = (
    waveform
    .squeeze(0)
    .cpu()
    .numpy()
)


# ==========================================
# Prepare Wav2Vec2 Input
# ==========================================

inputs = processor(
    audio_array,
    sampling_rate=sample_rate,
    return_tensors="pt",
    padding=True,
)


input_values = inputs.input_values


print(
    "\nWav2Vec2 Input Shape :",
    input_values.shape,
)


# ==========================================
# Load Pretrained Model
# ==========================================

print(
    "\nLoading pretrained model..."
)

model = PretrainedVoiceDistressModel()

model.eval()

print(
    "Model loaded successfully!"
)


# ==========================================
# Forward Pass
# ==========================================

print(
    "\nRunning model..."
)

with torch.inference_mode():

    logits = model(
        input_values
    )

    probabilities = torch.softmax(
        logits,
        dim=1,
    )


# ==========================================
# Output
# ==========================================

normal_probability = (
    probabilities[0][0].item()
)

distress_probability = (
    probabilities[0][1].item()
)


print("\n==============================")
print("MODEL OUTPUT")
print("==============================")

print(
    "Logits :",
    logits,
)

print(
    f"Normal Probability   : "
    f"{normal_probability:.6f}"
)

print(
    f"Distress Probability : "
    f"{distress_probability:.6f}"
)


# ==========================================
# Temporary Prediction
# ==========================================

prediction = torch.argmax(
    probabilities,
    dim=1,
).item()

if prediction == 0:

    label = "normal"
    confidence = normal_probability

else:

    label = "distress"
    confidence = distress_probability


print("\n==============================")
print("TEMPORARY PREDICTION")
print("==============================")

print(
    f"Predicted : {label}"
)

print(
    f"Confidence : {confidence * 100:.2f}%"
)


# ==========================================
# Important
# ==========================================

print("\n==============================")
print("Pipeline Test Successful!")
print("==============================")

print(
    "NOTE: The classifier is NOT trained yet."
)

print(
    "This prediction is NOT meaningful until "
    "fine-tuning is completed."
)