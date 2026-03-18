from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict

from app.driving.services.speed_limit_service import SpeedLimitService
from app.driving.services.weather_service import WeatherService


ROOT = Path(__file__).resolve().parents[3]
METADATA_PATH = ROOT / "videos" / "metadata.json"

DEFAULT_CONTEXT = {
    "lat": 63.4305,
    "lon": 10.3951,
    "speed_limit": 60,
    "weather": {
        "temp_c": 1.0,
        "precip_mm_h": 0.0,
        "humidity": 80.0,
        "summary": "clear",
        "source": "default",
    },
}


class ContextService:
    def __init__(self) -> None:
        self.reference_points = self._load_metadata()
        self.speed_limit_service = SpeedLimitService()
        self.weather_service = WeatherService()

    def _load_metadata(self) -> list[dict[str, Any]]:
        if not METADATA_PATH.exists():
            return []

        import json

        with open(METADATA_PATH, encoding="utf-8") as f:
            entries = json.load(f)

        out: list[dict[str, Any]] = []
        for entry in entries:
            out.append(
                {
                    "lat": float(entry.get("lat", 0.0)),
                    "lon": float(entry.get("lon", 0.0)),
                    "temp": float(entry.get("temp", 1.0)),
                    "humidity": float(entry.get("humidity", 80.0)),
                    "precipitation": float(entry.get("precipitation", 0.0)),
                }
            )
        return out

    @staticmethod
    def _distance_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        r = 6371.0
        p1 = math.radians(lat1)
        p2 = math.radians(lat2)
        dp = math.radians(lat2 - lat1)
        dl = math.radians(lon2 - lon1)
        a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
        return 2 * r * math.asin(math.sqrt(a))

    def _nearest_reference(self, lat: float, lon: float) -> dict[str, Any] | None:
        if not self.reference_points:
            return None
        return min(self.reference_points, key=lambda e: self._distance_km(lat, lon, e["lat"], e["lon"]))

    def get_context(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        lat = float(payload.get("lat", DEFAULT_CONTEXT["lat"]))
        lon = float(payload.get("lon", DEFAULT_CONTEXT["lon"]))

        weather = self.weather_service.lookup(lat, lon)
        ref = self._nearest_reference(lat, lon)
        speed_lookup = self.speed_limit_service.lookup(lat, lon)

        if speed_lookup is not None:
            speed_limit = int(speed_lookup["speed_limit"])
            speed_source = str(speed_lookup["source"])
            speed_details = {
                "road_ref": speed_lookup.get("road_ref"),
                "distance_m": speed_lookup.get("distance_m"),
            }
        elif ref is None:
            speed_limit = DEFAULT_CONTEXT["speed_limit"]
            speed_source = "default"
            speed_details = {}
            if weather is None:
                weather = dict(DEFAULT_CONTEXT["weather"])
                weather["source"] = "default"
        else:
            speed_limit = DEFAULT_CONTEXT["speed_limit"]
            speed_source = "default"
            speed_details = {}
            if weather is None:
                weather = {
                    "temp_c": ref["temp"],
                    "humidity": ref["humidity"],
                    "precip_mm_h": ref["precipitation"],
                    "summary": "fallback-metadata",
                    "source": "metadata",
                }

        return {
            "location": {"lat": lat, "lon": lon},
            "speed_limit": speed_limit,
            "speed_limit_source": speed_source,
            "speed_limit_details": speed_details,
            "weather": weather,
        }
