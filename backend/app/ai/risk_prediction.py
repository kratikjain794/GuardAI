from typing import Any


class RiskPredictor:
    """
    GuardIA risk fusion engine.

    Combines:
    - Emotion prediction
    - Voice distress prediction
    - Camera person count

    Important:
    This score is a risk indicator, not proof
    that an emergency is happening.
    """

    HIGH_RISK_EMOTIONS = {
        "fearful",
        "angry",
    }

    MODERATE_RISK_EMOTIONS = {
        "sad",
        "disgust",
        "surprised",
    }

    def predict(
        self,
        emotion: str,
        emotion_confidence: float,
        distress_detected: bool,
        distress_confidence: float,
        person_count: int,
    ) -> dict[str, Any]:

        emotion = emotion.lower().strip()

        # Keep inputs within valid ranges
        emotion_confidence = max(
            0.0,
            min(float(emotion_confidence), 100.0),
        )

        distress_confidence = max(
            0.0,
            min(float(distress_confidence), 100.0),
        )

        person_count = max(
            0,
            int(person_count),
        )

        risk_score = 0.0
        reasons = []

        # ==================================
        # Emotion Signal
        # Maximum contribution: 25
        # ==================================

        if emotion in self.HIGH_RISK_EMOTIONS:

            contribution = (
                25.0
                * emotion_confidence
                / 100.0
            )

            risk_score += contribution

            reasons.append(
                f"High-risk emotion detected: "
                f"{emotion}"
            )

        elif emotion in self.MODERATE_RISK_EMOTIONS:

            contribution = (
                12.0
                * emotion_confidence
                / 100.0
            )

            risk_score += contribution

            reasons.append(
                f"Potential stress emotion detected: "
                f"{emotion}"
            )

        # ==================================
        # Voice Distress Signal
        # Maximum contribution: 55
        # ==================================

        if distress_detected:

            contribution = (
                55.0
                * distress_confidence
                / 100.0
            )

            risk_score += contribution

            reasons.append(
                "Voice distress signal detected"
            )

        # ==================================
        # Camera Context
        # Maximum contribution: 10
        # ==================================
        #
        # Person count alone is NOT treated
        # as danger.
        #
        # It adds a small amount only when
        # distress / concerning emotion is
        # already present.
        # ==================================

        concerning_signal = (
            distress_detected
            or emotion in self.HIGH_RISK_EMOTIONS
        )

        if concerning_signal:

            if person_count >= 5:

                risk_score += 10.0

                reasons.append(
                    "Multiple people detected "
                    "during concerning signals"
                )

            elif person_count >= 2:

                risk_score += 5.0

                reasons.append(
                    "Additional people detected "
                    "during concerning signals"
                )

        # ==================================
        # Signal Agreement Bonus
        # Maximum contribution: 10
        # ==================================

        if (
            distress_detected
            and emotion in self.HIGH_RISK_EMOTIONS
        ):

            risk_score += 10.0

            reasons.append(
                "Voice distress and emotion "
                "signals agree"
            )

        # ==================================
        # Final Score
        # ==================================

        risk_score = min(
            round(risk_score, 2),
            100.0,
        )

        # ==================================
        # Risk Level
        # ==================================

        if risk_score >= 70:

            risk_level = "high"

        elif risk_score >= 40:

            risk_level = "medium"

        else:

            risk_level = "low"

        # ==================================
        # Recommended Action
        # ==================================

        if risk_level == "high":

            recommended_action = (
                "Prompt user to confirm safety "
                "and make SOS immediately available."
            )

        elif risk_level == "medium":

            recommended_action = (
                "Continue monitoring and show "
                "quick-access safety controls."
            )

        else:

            recommended_action = (
                "Continue normal monitoring."
            )

        return {
            "risk_score": risk_score,
            "risk_level": risk_level,
            "reasons": reasons,
            "recommended_action": recommended_action,
        }