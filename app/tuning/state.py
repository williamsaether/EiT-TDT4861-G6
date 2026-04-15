from __future__ import annotations

import threading
from typing import Any, Dict, List, Tuple

from app.common.tuned_params import AUTO_TUNED_PROFILE


class TuningState:
    """Thread-safe container for all tunable parameters."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.w_camera: float = float(AUTO_TUNED_PROFILE["w_camera"])
        self.w_weather: float = float(AUTO_TUNED_PROFILE["w_weather"])
        self.w_confidence: float = float(AUTO_TUNED_PROFILE["w_confidence"])
        self.difficulty_factors: Dict[int, float] = {
            int(k): float(v) for k, v in AUTO_TUNED_PROFILE["difficulty_factors"].items()
        }
        self.neutral_cam: float = float(AUTO_TUNED_PROFILE["neutral_cam"])
        self.wf_light_precip: float = float(AUTO_TUNED_PROFILE["weather_factors"]["light_precip"])
        self.wf_mod_precip: float = float(AUTO_TUNED_PROFILE["weather_factors"]["moderate_precip"])
        self.wf_heavy_precip: float = float(AUTO_TUNED_PROFILE["weather_factors"]["heavy_precip"])
        self.wf_near_freeze: float = float(AUTO_TUNED_PROFILE["weather_factors"]["near_freeze"])
        self.wf_freeze: float = float(AUTO_TUNED_PROFILE["weather_factors"]["freeze"])
        self.smooth_window_sec: float = float(AUTO_TUNED_PROFILE.get("smooth_window_sec", 3.0))

    def compute_weather_factor(self, temp_c: float, precip_mm_h: float) -> Tuple[float, List[str]]:
        factor = 1.0
        reasons: List[str] = []
        if precip_mm_h > 10:
            factor *= self.wf_heavy_precip
            reasons.append("heavy rain/snow")
        elif precip_mm_h > 5:
            factor *= self.wf_mod_precip
            reasons.append("moderate rain/snow")
        elif precip_mm_h > 0.5:
            factor *= self.wf_light_precip
            reasons.append("light rain/snow")
        if temp_c < 0:
            factor *= self.wf_freeze
            reasons.append("freezing")
        elif temp_c < 3:
            factor *= self.wf_near_freeze
            reasons.append("near freezing")
        return round(factor, 4), reasons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_camera": self.w_camera,
            "w_weather": self.w_weather,
            "w_confidence": self.w_confidence,
            "difficulty_factors": {str(k): v for k, v in self.difficulty_factors.items()},
            "neutral_cam": self.neutral_cam,
            "wf_light_precip": self.wf_light_precip,
            "wf_mod_precip": self.wf_mod_precip,
            "wf_heavy_precip": self.wf_heavy_precip,
            "wf_near_freeze": self.wf_near_freeze,
            "wf_freeze": self.wf_freeze,
            "smooth_window_sec": self.smooth_window_sec,
        }

    def apply(self, payload: Dict[str, Any]) -> None:
        if "w_camera" in payload:
            self.w_camera = max(0.0, float(payload["w_camera"]))
        if "w_weather" in payload:
            self.w_weather = max(0.0, float(payload["w_weather"]))
        if "w_confidence" in payload:
            self.w_confidence = max(0.0, float(payload["w_confidence"]))
        if "difficulty_factors" in payload:
            df = payload["difficulty_factors"]
            for level in range(1, 6):
                key = str(level)
                if key in df:
                    self.difficulty_factors[level] = max(0.05, min(1.0, float(df[key])))
        if "neutral_cam" in payload:
            self.neutral_cam = max(0.1, min(1.0, float(payload["neutral_cam"])))
        if "wf_light_precip" in payload:
            self.wf_light_precip = max(0.1, min(1.0, float(payload["wf_light_precip"])))
        if "wf_mod_precip" in payload:
            self.wf_mod_precip = max(0.1, min(1.0, float(payload["wf_mod_precip"])))
        if "wf_heavy_precip" in payload:
            self.wf_heavy_precip = max(0.1, min(1.0, float(payload["wf_heavy_precip"])))
        if "wf_near_freeze" in payload:
            self.wf_near_freeze = max(0.1, min(1.0, float(payload["wf_near_freeze"])))
        if "wf_freeze" in payload:
            self.wf_freeze = max(0.1, min(1.0, float(payload["wf_freeze"])))
        if "smooth_window_sec" in payload:
            self.smooth_window_sec = max(0.1, min(15.0, float(payload["smooth_window_sec"])))
