from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


AUTO_TUNED_PROFILE: Dict[str, Any] = {
    "w_camera": 11.0,
    "w_weather": 5.0,
    "w_confidence": 38.0,
    "neutral_cam": 0.88,
    "difficulty_factors": {
        1: 1.00,
        2: 1.00,
        3: 1.00,
        4: 0.63,
        5: 0.20,
    },
    "weather_factors": {
        "light_precip": 0.92,
        "moderate_precip": 0.76,
        "heavy_precip": 0.57,
        "near_freeze": 1.00,
        "freeze": 0.85,
    },
}

CLASS_TO_DIFFICULTY: Dict[str, int] = {
    "dry_asphalt": 1,
    "wet_asphalt": 2,
    "melted_snow": 3,
    "water_asphalt": 3,
    "fresh_snow": 4,
    "ice": 5,
}


def tuned_profile() -> Dict[str, Any]:
    """Return a mutable copy of the shared tuned profile."""
    return deepcopy(AUTO_TUNED_PROFILE)
