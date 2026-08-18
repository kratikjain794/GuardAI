from fastapi import APIRouter
from pydantic import BaseModel, Field

from app.utils.constants import (
    HIGH_RISK_THRESHOLD,
    LOW_RISK_THRESHOLD,
    RISK_HIGH,
    RISK_LOW,
    RISK_MEDIUM,
)

router = APIRouter(
    prefix="/risk",
    tags=["Risk Prediction"],
)


# ==========================================
# Risk Input
# ==========================================

class RiskInput(BaseModel):

    hour: int = Field(
        ge=0,
        le=23,
    )

    alone: bool = False

    dark_area: bool = False

    distress_detected: bool = False

    fearful_emotion: bool = False


# ==========================================
# Risk Calculation
# ==========================================

def calculate_risk(
    hour: int,
    alone: bool = False,
    dark_area: bool = False,
    distress_detected: bool = False,
    fearful_emotion: bool = False,
):
    """
    Central risk calculation function.

    This function can be used by:
        - /risk/predict
        - monitoring
        - future SOS automation
    """

    score = 0

    # --------------------------------------
    # Late night / early morning
    # --------------------------------------

    late_hours = (
        hour >= 22
        or hour <= 5
    )

    if late_hours:
        score += 25

    # --------------------------------------
    # User travelling alone
    # --------------------------------------

    if alone:
        score += 20

    # --------------------------------------
    # Dark area
    # --------------------------------------

    if dark_area:
        score += 20

    # --------------------------------------
    # Voice distress
    # --------------------------------------

    if distress_detected:
        score += 25

    # --------------------------------------
    # Fearful emotion
    # --------------------------------------

    if fearful_emotion:
        score += 10

    # --------------------------------------
    # Maximum score
    # --------------------------------------

    score = min(score, 100)

    # --------------------------------------
    # Risk Level
    # --------------------------------------

    if score >= HIGH_RISK_THRESHOLD:

        risk_level = RISK_HIGH

    elif score >= LOW_RISK_THRESHOLD:

        risk_level = RISK_MEDIUM

    else:

        risk_level = RISK_LOW

    # --------------------------------------
    # Signals
    # --------------------------------------

    signals = {

        "late_hours": late_hours,

        "alone": alone,

        "dark_area": dark_area,

        "distress_detected":
            distress_detected,

        "fearful_emotion":
            fearful_emotion,
    }

    return {

        "risk_score": score,

        "risk_level": risk_level,

        "signals": signals,

        "prediction_method":
            "rule_based_baseline_v1",
    }


# ==========================================
# Risk Prediction API
# ==========================================

@router.post("/predict")
async def predict_risk(
    data: RiskInput,
):

    result = calculate_risk(

        hour=data.hour,

        alone=data.alone,

        dark_area=data.dark_area,

        distress_detected=
            data.distress_detected,

        fearful_emotion=
            data.fearful_emotion,
    )

    return result