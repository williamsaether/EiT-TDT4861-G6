"""Orchestration-only data pipeline.

This module coordinates data collection from provider modules:
- Weather: `weather.weather_met`
- Speed limit: `speed_limit.nvdb_speed`
- Camera surface: `video_processor` (optional)

No source-specific API/model logic should live here.
"""

from __future__ import annotations

from collections import deque
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from typing import Any, Callable, Optional
import time


# ---------------------------------------------------------------------------
# Provider imports (thin adapters only)
# ---------------------------------------------------------------------------

try:
    from weather.weather_met import get_weather_data as _weather_provider  # type: ignore
except ImportError:
    _weather_provider = None

try:
    from speed_limit.nvdb_speed import get_speed_limit_data as _speed_provider  # type: ignore
except ImportError:
    _speed_provider = None

try:
    from video_processor import get_surface as _surface_provider  # type: ignore
except ImportError:
    _surface_provider = None


def _normalize_weather(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "temp": raw.get("temp"),
        "humidity": raw.get("humidity"),
        "precipitation": raw.get("precipitation"),
        "status": raw.get("status", "ok"),
    }


def _normalize_speed(raw: dict[str, Any]) -> dict[str, Any]:
    return {
        "fartsgrense": raw.get("fartsgrense"),
        "vei": raw.get("vei"),
        "avstand_meter": raw.get("avstand_meter"),
        "status": raw.get("status", "ok"),
        "message": raw.get("message"),
    }


def _as_score_map(values: Any) -> dict[str, float]:
    if isinstance(values, dict):
        return {str(k): float(v) for k, v in values.items()}
    if isinstance(values, list):
        out: dict[str, float] = {}
        for item in values:
            if isinstance(item, (list, tuple)) and len(item) >= 2:
                out[str(item[0])] = float(item[1])
        return out
    return {}


def _topk(scores: dict[str, float], k: int = 3) -> list[list[Any]]:
    return [[name, float(score)] for name, score in sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]]


class RoadDataPipeline:
    """Stateful multi-rate collector.

    Defaults:
    - Camera surface provider: 5 Hz, rolling average over 3s
    - Speed provider: every 20s
    - Weather provider: every 10min
    """

    def __init__(
        self,
        *,
        weather_fn: Optional[Callable[..., dict[str, Any]]] = None,
        speed_fn: Optional[Callable[..., dict[str, Any]]] = None,
        surface_fn: Optional[Callable[..., dict[str, Any]]] = None,
        surface_hz: float = 5.0,
        surface_window_s: float = 3.0,
        weather_interval_s: float = 600.0,
        speed_interval_s: float = 20.0,
        speed_limit_radius_m: int = 50,
    ):
        self.weather_fn = weather_fn or _weather_provider
        self.speed_fn = speed_fn or _speed_provider
        self.surface_fn = surface_fn or _surface_provider

        if surface_hz <= 0:
            raise ValueError("surface_hz must be > 0")

        self.surface_interval_s = 1.0 / surface_hz
        self.surface_window_s = float(surface_window_s)
        self.weather_interval_s = float(weather_interval_s)
        self.speed_interval_s = float(speed_interval_s)
        self.speed_limit_radius_m = int(speed_limit_radius_m)

        self.last_weather = {"temp": None, "humidity": None, "precipitation": None, "status": "not_updated"}
        self.last_speed = {
            "fartsgrense": None,
            "vei": None,
            "avstand_meter": None,
            "status": "not_updated",
            "message": None,
        }
        self.last_surface: Optional[dict[str, Any]] = None

        self.last_weather_ts: Optional[float] = None
        self.last_speed_ts: Optional[float] = None
        self.last_surface_ts: Optional[float] = None

        self._surface_history: deque[tuple[float, dict[str, Any]]] = deque()

    def _weather_due(self, now: float, force: bool) -> bool:
        return force or self.last_weather_ts is None or (now - self.last_weather_ts) >= self.weather_interval_s

    def _speed_due(self, now: float, force: bool) -> bool:
        return force or self.last_speed_ts is None or (now - self.last_speed_ts) >= self.speed_interval_s

    def _surface_due(self, now: float) -> bool:
        return self.last_surface_ts is None or (now - self.last_surface_ts) >= self.surface_interval_s

    def _trim_surface_history(self, now: float) -> None:
        min_ts = now - self.surface_window_s
        while self._surface_history and self._surface_history[0][0] < min_ts:
            self._surface_history.popleft()

    def _average_surface(self, now: float) -> Optional[dict[str, Any]]:
        self._trim_surface_history(now)
        if not self._surface_history:
            return None

        n = len(self._surface_history)
        groups = {"friction": {}, "surface": {}, "uneven": {}, "winter": {}, "raw_top": {}}

        for _, sample in self._surface_history:
            for key in groups:
                score_map = _as_score_map(sample.get(key, {}))
                for label, value in score_map.items():
                    groups[key][label] = groups[key].get(label, 0.0) + value

        for key in groups:
            for label in list(groups[key].keys()):
                groups[key][label] /= n

        out = {
            "status": "ok",
            "samples_in_window": n,
            "window_seconds": self.surface_window_s,
            "friction": _topk(groups["friction"]),
            "surface": _topk(groups["surface"]),
            "uneven": _topk(groups["uneven"]),
            "winter": _topk(groups["winter"]),
            "raw_top": _topk(groups["raw_top"], k=5),
        }

        # Preserve optional provider metadata from latest sample.
        latest = self._surface_history[-1][1]
        for k in ("model_path", "model_format"):
            if k in latest:
                out[k] = latest[k]

        return out

    def update(
        self,
        *,
        lat: float,
        lon: float,
        altitude: float = 0.0,
        image: Any = None,
        force_refresh: bool = False,
    ) -> dict[str, Any]:
        now = time.time()

        weather_due = self._weather_due(now, force_refresh)
        speed_due = self._speed_due(now, force_refresh)

        if weather_due or speed_due:
            with ThreadPoolExecutor(max_workers=2) as ex:
                fut_weather = None
                fut_speed = None

                if weather_due:
                    if self.weather_fn is None:
                        self.last_weather = {
                            "temp": None,
                            "humidity": None,
                            "precipitation": None,
                            "status": "error: weather provider not available",
                        }
                        self.last_weather_ts = now
                    else:
                        fut_weather = ex.submit(self.weather_fn, lat=lat, lon=lon, altitude=altitude)

                if speed_due:
                    if self.speed_fn is None:
                        self.last_speed = {
                            "fartsgrense": None,
                            "vei": None,
                            "avstand_meter": None,
                            "status": "error: speed provider not available",
                            "message": None,
                        }
                        self.last_speed_ts = now
                    else:
                        fut_speed = ex.submit(self.speed_fn, lat=lat, lon=lon)

                if fut_weather is not None:
                    try:
                        self.last_weather = _normalize_weather(fut_weather.result())
                    except Exception as e:
                        self.last_weather = {
                            "temp": None,
                            "humidity": None,
                            "precipitation": None,
                            "status": f"error: {e}",
                        }
                    self.last_weather_ts = now

                if fut_speed is not None:
                    try:
                        self.last_speed = _normalize_speed(fut_speed.result())
                    except Exception as e:
                        self.last_speed = {
                            "fartsgrense": None,
                            "vei": None,
                            "avstand_meter": None,
                            "status": f"error: {e}",
                            "message": None,
                        }
                    self.last_speed_ts = now

        if image is not None and self._surface_due(now):
            if self.surface_fn is None:
                self.last_surface = {"status": "error: surface provider (video_processor.get_surface) not available"}
                self.last_surface_ts = now
            else:
                try:
                    try:
                        sample = self.surface_fn(image=image)
                    except TypeError:
                        sample = self.surface_fn(image)
                    if not isinstance(sample, dict):
                        raise TypeError(f"surface provider must return dict, got {type(sample)}")
                    self._surface_history.append((now, sample))
                    self.last_surface = self._average_surface(now)
                except Exception as e:
                    self.last_surface = {"status": f"error: {e}"}
                self.last_surface_ts = now

        ts = datetime.now(timezone.utc).isoformat()
        return {
            "timestamp": ts,
            "gps": {"lat": lat, "lon": lon, "altitude": altitude},
            "weather": self.last_weather,
            "weather_age_s": None if self.last_weather_ts is None else round(now - self.last_weather_ts, 3),
            "speed_limit": self.last_speed,
            "speed_limit_age_s": None if self.last_speed_ts is None else round(now - self.last_speed_ts, 3),
            "camera": self.last_surface,
            "camera_age_s": None if self.last_surface_ts is None else round(now - self.last_surface_ts, 3),
        }


def collect_pipeline_input(
    lat: float,
    lon: float,
    altitude: float = 0.0,
    camera_output: Optional[dict[str, Any]] = None,
) -> dict[str, Any]:
    """One-shot collection via imported providers."""
    pipe = RoadDataPipeline()
    snap = pipe.update(lat=lat, lon=lon, altitude=altitude, image=None, force_refresh=True)
    snap["camera"] = camera_output
    return snap


if __name__ == "__main__":
    import json

    result = collect_pipeline_input(lat=63.4305, lon=10.3951, altitude=20)
    print(json.dumps(result, indent=2, ensure_ascii=False))
