from __future__ import annotations

from typing import Any, Dict, Tuple

from app.common.tuned_params import CLASS_TO_DIFFICULTY, AUTO_TUNED_PROFILE


def _normalize_scores(scores: Dict[str, float]) -> Dict[str, float]:
    cleaned: Dict[str, float] = {}
    for key, value in scores.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        cleaned[str(key)] = max(0.0, v)

    total = sum(cleaned.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in cleaned.items()}


def _top_label(scores: Dict[str, float]) -> Tuple[str, float]:
    if not scores:
        return "unknown", 0.0
    label, confidence = max(scores.items(), key=lambda item: item[1])
    return str(label), float(confidence)


def apply_heuristics(
    scores: Dict[str, float],
    *,
    temp_c: float | None,
    humidity: float | None,
    precip_mm_h: float | None,
    speed_limit: float | None,
) -> Dict[str, float]:
    """Apply research heuristics to class scores, then renormalize."""
    adjusted = _normalize_scores(scores)
    if not adjusted:
        return {}

    if temp_c is not None:
        if temp_c >= 7.0 and "ice" in adjusted:
            adjusted["ice"] = 0.0
        if temp_c >= 5.0 and "fresh_snow" in adjusted:
            adjusted["fresh_snow"] = 0.0

        boost_ice = False
        if temp_c < 2.0:
            wet_or_humid = (humidity is not None and humidity > 70.0) or ((precip_mm_h or 0.0) >= 0.5)
            near_freezing = -2.0 < temp_c < 2.0
            boost_ice = wet_or_humid or near_freezing
        if boost_ice and adjusted.get("ice", 0.0) > 0.05:
            adjusted["ice"] *= 1.5

    if speed_limit is not None and speed_limit >= 80.0:
        for label in list(adjusted.keys()):
            if label == "gravel" or label == "mud" or label.endswith("_gravel") or label.endswith("_mud"):
                adjusted[label] *= 0.1

    return _normalize_scores(adjusted)


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
    class_scores: Dict[str, float] | None = None,
    humidity: float | None = None,
    params: Dict[str, Any] | None = None,
) -> Dict[str, Any]:
    profile = params or AUTO_TUNED_PROFILE
    working_label = str(label)
    conf = max(0.0, min(confidence, 1.0))

    if class_scores:
        adjusted_scores = apply_heuristics(
            class_scores,
            temp_c=temp_c,
            humidity=humidity,
            precip_mm_h=precip_mm_h,
            speed_limit=float(speed_limit),
        )
        if adjusted_scores:
            working_label, conf = _top_label(adjusted_scores)

    difficulty = CLASS_TO_DIFFICULTY.get(working_label, 3)
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
