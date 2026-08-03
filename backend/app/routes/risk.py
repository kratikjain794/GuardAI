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


class RiskInput(BaseModel):
    hour: int = Field(ge=0, le=23)

    alone: bool = False
    dark_area: bool = False
    distress_detected: bool = False
    fearful_emotion: bool = False


@router.post("/predict")
async def predict_risk(data: RiskInput):

    score = 0

    # Late night / early morning
    if data.hour >= 22 or data.hour <= 5:
        score += 25

    # User travelling alone
    if data.alone:
        score += 20

    # Poorly lit environment
    if data.dark_area:
        score += 20

    # Voice distress model signal
    if data.distress_detected:
        score += 25

    # Emotion model signal
    if data.fearful_emotion:
        score += 10

    score = min(score, 100)

    if score >= HIGH_RISK_THRESHOLD:
        risk_level = RISK_HIGH

    elif score >= LOW_RISK_THRESHOLD:
        risk_level = RISK_MEDIUM

    else:
        risk_level = RISK_LOW

    return {
        "risk_score": score,
        "risk_level": risk_level,
        "signals": {
            "late_hours": (
                data.hour >= 22
                or data.hour <= 5
            ),
            "alone": data.alone,
            "dark_area": data.dark_area,
            "distress_detected":
                data.distress_detected,
            "fearful_emotion":
                data.fearful_emotion,
        },
        "prediction_method":
            "rule_based_baseline_v1",
    }