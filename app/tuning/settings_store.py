from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict

from app.tuning.config import DEFAULT_MODEL_NAME


SETTINGS_PATH = Path(__file__).resolve().parent / "settings.json"
_SETTINGS_LOCK = threading.Lock()


def _default_settings() -> Dict[str, Any]:
    return {
        "tuning_state": {},
        "analyzer": {
            "model": DEFAULT_MODEL_NAME,
            "use_crop": False,
            "smooth_window_sec": 3.0,
        },
        "ui": {},
    }


def load_settings() -> Dict[str, Any]:
    defaults = _default_settings()
    if not SETTINGS_PATH.exists():
        return defaults
    try:
        with open(SETTINGS_PATH, encoding="utf-8") as f:
            raw = json.load(f)
        if not isinstance(raw, dict):
            return defaults
        out = _default_settings()
        if isinstance(raw.get("tuning_state"), dict):
            out["tuning_state"] = raw["tuning_state"]
        if isinstance(raw.get("analyzer"), dict):
            out["analyzer"].update(raw["analyzer"])
        if isinstance(raw.get("ui"), dict):
            out["ui"].update(raw["ui"])
        try:
            out["analyzer"]["smooth_window_sec"] = float(out["analyzer"].get("smooth_window_sec", 3.0))
        except Exception:
            out["analyzer"]["smooth_window_sec"] = 3.0
        return out
    except Exception:
        return defaults


def save_settings(payload: Dict[str, Any]) -> str:
    with _SETTINGS_LOCK:
        with open(SETTINGS_PATH, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(SETTINGS_PATH)
