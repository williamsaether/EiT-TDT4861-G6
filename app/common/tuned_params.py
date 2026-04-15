from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict


AUTO_TUNED_PROFILE: Dict[str, Any] = {
    "w_camera": 5.0,
    "w_weather": 10.06,
    "w_confidence": 40.0,
    "neutral_cam": 0.92,
    "difficulty_factors": {
        1: 1.0,
        2: 1.0,
        3: 1.0,
        4: 0.517,
        5: 0.561,
    },
    "weather_factors": {
        "light_precip": 0.84,
        "moderate_precip": 0.86,
        "heavy_precip": 0.693,
        "near_freeze": 1.0,
        "freeze": 0.85,
    },
    "smooth_window_sec": 3.75,
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
