# ==========================================
# GuardIA Application Constants
# ==========================================

APP_NAME = "GuardIA"
APP_VERSION = "1.0.0"

# ------------------------------------------
# Risk Levels
# ------------------------------------------

RISK_LOW = "LOW"
RISK_MEDIUM = "MEDIUM"
RISK_HIGH = "HIGH"

LOW_RISK_THRESHOLD = 40
HIGH_RISK_THRESHOLD = 70


# ------------------------------------------
# Emergency / SOS
# ------------------------------------------

DEFAULT_SOS_MESSAGE = (
    "Emergency! I may be in danger. "
    "Please check my current location."
)

EMERGENCY_NUMBER_INDIA = "112"


# ------------------------------------------
# Emotion Detection
# ------------------------------------------

EMOTIONS = [
    "neutral",
    "calm",
    "happy",
    "sad",
    "angry",
    "fearful",
    "disgust",
    "surprised",
]

HIGH_RISK_EMOTIONS = {
    "fearful",
    "angry",
}


# ------------------------------------------
# Audio Settings
# ------------------------------------------

AUDIO_SAMPLE_RATE = 16000
AUDIO_DURATION_SECONDS = 3
N_MFCC = 40


# ------------------------------------------
# Monitoring
# ------------------------------------------

VOICE_DISTRESS_THRESHOLD = 0.75
EMOTION_CONFIDENCE_THRESHOLD = 0.70