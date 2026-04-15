from __future__ import annotations

from typing import Any, Dict

from app.common.recommendation import compute_recommendation
from app.driving.services.profile_loader import active_profile_name, load_driving_profile


class RecommendationService:
    def __init__(self) -> None:
        self.params = load_driving_profile()
        self.profile_name = active_profile_name()

    def recommend(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        speed_limit = int(payload.get("speed_limit", 60))

        classification = payload.get("classification") or {}
        label = str(classification.get("label", "unknown"))
        confidence = float(classification.get("confidence", 0.0))
        scores_raw = classification.get("scores")
        class_scores = scores_raw if isinstance(scores_raw, dict) else {}

        weather = payload.get("weather") or {}
        temp_c = float(weather.get("temp_c", 1.0))
        precip_mm_h = float(weather.get("precip_mm_h", 0.0))
        humidity = weather.get("humidity")
        try:
            humidity_f = float(humidity) if humidity is not None else None
        except (TypeError, ValueError):
            humidity_f = None

        return compute_recommendation(
            speed_limit=speed_limit,
            label=label,
            confidence=confidence,
            temp_c=temp_c,
            precip_mm_h=precip_mm_h,
            class_scores=class_scores,
            humidity=humidity_f,
            params=self.params,
        )
