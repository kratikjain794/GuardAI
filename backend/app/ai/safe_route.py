from typing import Any


class SafeRouteAnalyzer:
    """
    GuardIA safer-route analysis module.

    Each route can contain:
    - distance_km
    - duration_minutes
    - crime_risk
    - lighting_risk
    - crowd_risk
    - night_risk

    Risk values should be between 0 and 1.
    """

    def __init__(self):

        self.weights = {
            "crime_risk": 0.40,
            "lighting_risk": 0.25,
            "crowd_risk": 0.20,
            "night_risk": 0.15,
        }

    @staticmethod
    def _validate_risk(value: float) -> float:

        if not 0 <= value <= 1:
            raise ValueError(
                "Risk values must be between 0 and 1."
            )

        return float(value)

    def calculate_risk_score(
        self,
        crime_risk: float,
        lighting_risk: float,
        crowd_risk: float,
        night_risk: float,
    ) -> float:

        crime_risk = self._validate_risk(
            crime_risk
        )

        lighting_risk = self._validate_risk(
            lighting_risk
        )

        crowd_risk = self._validate_risk(
            crowd_risk
        )

        night_risk = self._validate_risk(
            night_risk
        )

        score = (
            crime_risk
            * self.weights["crime_risk"]
            + lighting_risk
            * self.weights["lighting_risk"]
            + crowd_risk
            * self.weights["crowd_risk"]
            + night_risk
            * self.weights["night_risk"]
        )

        return round(score * 100, 2)

    @staticmethod
    def get_risk_level(
        score: float,
    ) -> str:

        if score >= 70:
            return "HIGH"

        if score >= 40:
            return "MEDIUM"

        return "LOW"

    def analyze_route(
        self,
        route: dict[str, Any],
    ) -> dict:

        required_fields = [
            "crime_risk",
            "lighting_risk",
            "crowd_risk",
            "night_risk",
        ]

        missing_fields = [
            field
            for field in required_fields
            if field not in route
        ]

        if missing_fields:
            raise ValueError(
                "Missing route fields: "
                + ", ".join(missing_fields)
            )

        score = self.calculate_risk_score(
            crime_risk=route["crime_risk"],
            lighting_risk=route["lighting_risk"],
            crowd_risk=route["crowd_risk"],
            night_risk=route["night_risk"],
        )

        return {
            **route,
            "safety_risk_score": score,
            "risk_level": self.get_risk_level(
                score
            ),
        }

    def find_safest_route(
        self,
        routes: list[dict[str, Any]],
    ) -> dict:

        if not routes:
            raise ValueError(
                "At least one route is required."
            )

        analyzed_routes = [
            self.analyze_route(route)
            for route in routes
        ]

        safest_route = min(
            analyzed_routes,
            key=lambda route:
                route["safety_risk_score"],
        )

        return {
            "recommended_route": safest_route,
            "all_routes": analyzed_routes,
        }