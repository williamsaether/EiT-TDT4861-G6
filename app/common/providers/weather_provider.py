from __future__ import annotations

from typing import Any, Dict, Optional

import requests


class WeatherProvider:
    """Fetch current weather from Open-Meteo for a coordinate."""

    WEATHER_URL = "https://api.open-meteo.com/v1/forecast"

    @staticmethod
    def _weather_code_to_text(code: int) -> str:
        if code == 0:
            return "clear"
        if code in {1, 2, 3}:
            return "cloudy"
        if code in {45, 48}:
            return "fog"
        if code in {51, 53, 55, 61, 63, 65, 80, 81, 82}:
            return "rain"
        if code in {56, 57, 66, 67, 71, 73, 75, 77, 85, 86}:
            return "snow"
        if code in {95, 96, 99}:
            return "thunder"
        return "unknown"

    def lookup(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        params = {
            "latitude": lat,
            "longitude": lon,
            "current": "temperature_2m,relative_humidity_2m,precipitation,weather_code",
            "timezone": "auto",
        }

        try:
            res = requests.get(self.WEATHER_URL, params=params, timeout=3)
            res.raise_for_status()
            payload = res.json().get("current", {})
            weather_code = int(payload.get("weather_code", 0))
            return {
                "temp_c": float(payload.get("temperature_2m", 1.0)),
                "humidity": float(payload.get("relative_humidity_2m", 80.0)),
                "precip_mm_h": float(payload.get("precipitation", 0.0)),
                "summary": self._weather_code_to_text(weather_code),
                "source": "open-meteo",
            }
        except Exception:
            return None
