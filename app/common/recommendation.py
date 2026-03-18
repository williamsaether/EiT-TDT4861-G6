from __future__ import annotations

from typing import Any, Dict, Tuple

from app.common.tuned_params import CLASS_TO_DIFFICULTY, AUTO_TUNED_PROFILE


def compute_weather_factor(
    temp_c: float,
    precip_mm_h: float,
    weather_factors: Dict[str, float],
) -> Tuple[float, list[str]]:
    factor = 1.0
    reasons: list[str] = []

    if precip_mm_h > 10:
        factor *= weather_factors["heavy_precip"]
        reasons.append("heavy precipitation")
    elif precip_mm_h > 5:
        factor *= weather_factors["moderate_precip"]
        reasons.append("moderate precipitation")
    elif precip_mm_h > 0.5:
        factor *= weather_factors["light_precip"]
        reasons.append("light precipitation")

    if temp_c < 0:
        factor *= weather_factors["freeze"]
        reasons.append("freezing")
    elif temp_c < 3:
        factor *= weather_factors["near_freeze"]
        reasons.append("near freezing")

    return max(0.2, min(round(factor, 4), 1.0)), reasons


def compute_recommendation(
    speed_limit: int,
    label: str,
    confidence: float,
    temp_c: float,
    precip_mm_h: float,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    profile = params or AUTO_TUNED_PROFILE
    conf = max(0.0, min(confidence, 1.0))

    difficulty = CLASS_TO_DIFFICULTY.get(label, 3)
    camera_factor = float(profile["difficulty_factors"][difficulty])
    weather_factor, reasons = compute_weather_factor(
        temp_c=temp_c,
        precip_mm_h=precip_mm_h,
        weather_factors=profile["weather_factors"],
    )

    neutral_camera = float(profile["neutral_cam"])
    w_camera = float(profile["w_camera"])
    w_weather = float(profile["w_weather"])
    w_confidence = float(profile["w_confidence"])

    w_total = max(w_camera + w_weather + w_confidence, 1e-6)
    w_conf_norm = w_confidence / w_total
    effective_camera = camera_factor + w_conf_norm * (1.0 - conf) * (neutral_camera - camera_factor)

    combined = (w_camera * effective_camera + w_weather * weather_factor) / max(w_camera + w_weather, 1e-6)

    raw = speed_limit * max(0.2, min(combined, 1.0))
    rounded = int(round(raw / 10.0) * 10)
    recommended = max(20, min(rounded, speed_limit))

    return {
        "recommended_speed": recommended,
        "camera_factor": round(camera_factor, 3),
        "effective_camera_factor": round(effective_camera, 3),
        "weather_factor": weather_factor,
        "weather_reasons": reasons,
    }
