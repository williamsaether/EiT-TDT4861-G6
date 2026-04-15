from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict

from app.common.tuned_params import tuned_profile


ROOT = Path(__file__).resolve().parents[3]
DRIVING_CONFIG_DIR = ROOT / "app" / "driving" / "config"
SETTINGS_PATH = DRIVING_CONFIG_DIR / "settings.json"
PROFILES_DIR = DRIVING_CONFIG_DIR / "profiles"


def _normalize_profile(raw: Dict[str, Any]) -> Dict[str, Any]:
    profile = tuned_profile()
    profile["w_camera"] = float(raw.get("w_camera", profile["w_camera"]))
    profile["w_weather"] = float(raw.get("w_weather", profile["w_weather"]))
    profile["w_confidence"] = float(raw.get("w_confidence", profile["w_confidence"]))
    profile["neutral_cam"] = float(raw.get("neutral_cam", profile["neutral_cam"]))

    diffs = raw.get("difficulty_factors") or {}
    for lvl in range(1, 6):
        key = str(lvl)
        if key in diffs:
            profile["difficulty_factors"][lvl] = float(diffs[key])
        elif lvl in diffs:
            profile["difficulty_factors"][lvl] = float(diffs[lvl])

    weather = raw.get("weather_factors") or {}
    for key in ("light_precip", "moderate_precip", "heavy_precip", "near_freeze", "freeze"):
        if key in weather:
            profile["weather_factors"][key] = float(weather[key])

    try:
        smooth_window_sec = float(raw.get("smooth_window_sec", profile.get("smooth_window_sec", 3.0)))
    except (TypeError, ValueError):
        smooth_window_sec = float(profile.get("smooth_window_sec", 3.0))
    profile["smooth_window_sec"] = max(0.1, min(15.0, smooth_window_sec))

    return profile


def _load_json(path: Path) -> Dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict):
            return data
        return None
    except Exception:
        return None


def load_driving_profile() -> Dict[str, Any]:
    """Load active driving profile from app/driving/config/settings.json.

    Falls back to shared AUTO_TUNED_PROFILE if config is missing/invalid.
    """
    settings = _load_json(SETTINGS_PATH) or {}
    active_name = str(settings.get("active_profile", "")).strip()
    if not active_name:
        return tuned_profile()

    profile_path = PROFILES_DIR / Path(active_name).name
    raw_profile = _load_json(profile_path)
    if raw_profile is None:
        return tuned_profile()

    return _normalize_profile(raw_profile)


def active_profile_name() -> str:
    settings = _load_json(SETTINGS_PATH) or {}
    active_name = str(settings.get("active_profile", "")).strip()
    return active_name or "AUTO_TUNED_PROFILE"
