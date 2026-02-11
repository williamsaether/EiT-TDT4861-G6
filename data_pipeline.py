"""
data_pipeline.py

Gathers data from:
  1. Weather API (Open-Meteo) → temp, humidity
  2. NVDB Speed Limit API      → fartsgrense, vei, avstand_meter
  3. Camera ML model output    → passed in externally (not fetched here)

Packages everything into a single PipelineInput dict ready for feature engineering.

Usage:
    from data_pipeline import collect_pipeline_input

    camera_output = { ... }   # output from the road surface ML model
    data = collect_pipeline_input(lat=63.4305, lon=10.3951, altitude=20, camera_output=camera_output)
"""

import requests
from datetime import datetime, timezone
from typing import Optional

try:
    from pyproj import Transformer
    _HAS_PYPROJ = True
except ImportError:
    Transformer = None
    _HAS_PYPROJ = False

# ---------------------------------------------------------------------------
# Try to import existing repo modules; fall back to built-in implementations
# ---------------------------------------------------------------------------
try:
    from weatherData import get_weather_data as _repo_get_weather  # type: ignore
    _USE_REPO_WEATHER = True
except ImportError:
    _USE_REPO_WEATHER = False

try:
    from speed_limit.speed_limit import get_speed_limit as _repo_get_speed_limit  # type: ignore
    _USE_REPO_SPEED = True
except ImportError:
    _USE_REPO_SPEED = False


# ---------------------------------------------------------------------------
# Weather
# ---------------------------------------------------------------------------

def get_weather(lat: float, lon: float, altitude: Optional[float] = None) -> dict:
    """
    Fetch current temperature (°C) and relative humidity (%) for a location.

    Returns:
        {
            "temp": float,      # degrees Celsius
            "humidity": float   # percent
        }

    Raises:
        RuntimeError if the API call fails.
    """
    if _USE_REPO_WEATHER:
        return _repo_get_weather(lat=lat, lon=lon, altitude=altitude)

    # --- Open-Meteo (free, no API key required) ---
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat,
        "longitude": lon,
        "current": "temperature_2m,relative_humidity_2m",
        "timezone": "auto",
    }
    if altitude is not None:
        # Open-Meteo accepts elevation override for better accuracy
        params["elevation"] = altitude

    resp = requests.get(url, params=params, timeout=10)
    resp.raise_for_status()
    body = resp.json()

    current = body.get("current", {})
    temp = current.get("temperature_2m")
    humidity = current.get("relative_humidity_2m")

    if temp is None or humidity is None:
        raise RuntimeError(f"Unexpected Open-Meteo response: {body}")

    return {"temp": float(temp), "humidity": float(humidity)}


# NVDB public REST API v4
_NVDB_BASE = "https://nvdbapiles.atlas.vegvesen.no"
_FARTSGRENSE_TYPE = 105   # Vegdatatype 105 = Fartsgrense
_NVDB_POSISJON_URL = f"{_NVDB_BASE}/vegnett/api/v4/posisjon"
_NVDB_OBJEKT_URL = f"{_NVDB_BASE}/vegobjekter/api/v4/vegobjekter/{_FARTSGRENSE_TYPE}"

_NVDB_HEADERS = {
    "X-Client": "NTNU_EiT_StudentProject_DataPipeline",
    "Accept": "application/vnd.vegvesen.nvdb-v4+json",
}

_WGS84_TO_5973 = Transformer.from_crs("EPSG:4326", "EPSG:5973", always_xy=True) if _HAS_PYPROJ else None

def get_speed_limit(lat: float, lon: float, search_radius_m: int = 50) -> dict:
    """
    Fetch the current speed limit closest to (lat, lon) using the NVDB API.

    Returns:
        {
            "fartsgrense": int,       # km/h
            "vei": str,               # vegreferanse / veinavn
            "avstand_meter": float,   # distance from GPS point to road geometry
            "status": "ok" | "not_found" | "error"
        }
    """
    if _USE_REPO_SPEED:
        return _repo_get_speed_limit(lat=lat, lon=lon)

    if not _HAS_PYPROJ:
        return {
            "fartsgrense": None,
            "vei": None,
            "avstand_meter": None,
            "status": "error: pyproj is required for NVDB v4 lookup (pip install pyproj)",
        }

    try:
        # --- Step 1: snap to nearest road in SRID 5973 (same approach as repo script) ---
        ost, nord = _WGS84_TO_5973.transform(lon, lat)
        pos_params = {
            "nord": nord,
            "ost": ost,
            "srid": 5973,
            "maks_avstand": max(150, search_radius_m),
            "maks_antall": 1,
        }
        pos_resp = requests.get(_NVDB_POSISJON_URL, params=pos_params, headers=_NVDB_HEADERS, timeout=10)
        pos_resp.raise_for_status()
        pos_data = pos_resp.json()

        if not pos_data:
            return {
                "fartsgrense": None,
                "vei": None,
                "avstand_meter": None,
                "status": "not_found",
            }

        nearest = pos_data[0]
        avstand = nearest.get("avstand")
        vsys = nearest.get("vegsystemreferanse", {})
        vegsys = vsys.get("vegsystem", {})
        veikategori = vegsys.get("vegkategori")
        veinummer = vegsys.get("nummer")
        vei = vsys.get("kortform")

        # --- Step 2: query speed-limit objects in a bbox around snapped point ---
        size = max(50, search_radius_m)
        kartutsnitt = f"{ost-size},{nord-size},{ost+size},{nord+size}"
        obj_params = {
            "kartutsnitt": kartutsnitt,
            "srid": 5973,
            "inkluder": "egenskaper,lokasjon",
            "antall": 20,
        }
        obj_resp = requests.get(_NVDB_OBJEKT_URL, params=obj_params, headers=_NVDB_HEADERS, timeout=10)
        obj_resp.raise_for_status()
        objekter = obj_resp.json().get("objekter", [])

        if not objekter:
            return {
                "fartsgrense": None,
                "vei": vei,
                "avstand_meter": float(avstand) if avstand is not None else None,
                "status": "not_found",
            }

        # Prefer object on the same road (same vegkategori + veinummer)
        for obj in objekter:
            fart_verdi = None
            for eg in obj.get("egenskaper", []):
                if eg.get("id") == 2021:
                    fart_verdi = eg.get("verdi")
                    break
            if fart_verdi is None:
                continue

            match_found = False
            if veikategori is not None and veinummer is not None:
                for vref in obj.get("lokasjon", {}).get("vegsystemreferanser", []):
                    v_sys = vref.get("vegsystem", {})
                    if v_sys.get("vegkategori") == veikategori and v_sys.get("nummer") == veinummer:
                        match_found = True
                        break
            else:
                match_found = True

            if match_found:
                return {
                    "fartsgrense": int(fart_verdi),
                    "vei": vei,
                    "avstand_meter": float(avstand) if avstand is not None else None,
                    "status": "ok",
                }

        # Fallback: first object with speed-limit value
        for obj in objekter:
            for eg in obj.get("egenskaper", []):
                if eg.get("id") == 2021 and eg.get("verdi") is not None:
                    return {
                        "fartsgrense": int(eg.get("verdi")),
                        "vei": vei,
                        "avstand_meter": float(avstand) if avstand is not None else None,
                        "status": "ok",
                    }

        return {
            "fartsgrense": None,
            "vei": vei,
            "avstand_meter": float(avstand) if avstand is not None else None,
            "status": "not_found",
        }

    except requests.RequestException as e:
        return {
            "fartsgrense": None,
            "vei": None,
            "avstand_meter": None,
            "status": f"error: {e}",
        }
    except Exception as e:
        return {
            "fartsgrense": None,
            "vei": None,
            "avstand_meter": None,
            "status": f"error: {e}",
        }


# ---------------------------------------------------------------------------
# Main collection function
# ---------------------------------------------------------------------------

def collect_pipeline_input(
    lat: float,
    lon: float,
    altitude: float = 0.0,
    camera_output: Optional[dict] = None,
    speed_limit_radius_m: int = 50,
) -> dict:
    """
    Collect all input data for the feature engineering pipeline.

    Args:
        lat:                  GPS latitude (decimal degrees)
        lon:                  GPS longitude (decimal degrees)
        altitude:             GPS altitude in metres (used by weather API)
        camera_output:        Dict output from the road surface ML model.
                              Pass None if no camera data is available.
        speed_limit_radius_m: Max search radius for NVDB speed limit lookup.

    Returns a dict with the following structure:
    {
        "timestamp": str,           # ISO-8601 UTC timestamp
        "gps": {
            "lat": float,
            "lon": float,
            "altitude": float
        },
        "weather": {
            "temp": float,          # °C
            "humidity": float       # %
        },
        "speed_limit": {
            "fartsgrense": int,     # km/h  (None if not found)
            "vei": str,             # road reference
            "avstand_meter": float, # distance GPS→road
            "status": str
        },
        "camera": {                 # None if not provided
            "friction": [...],
            "surface": [...],
            "uneven":  [...],
            "winter":  [...],
            "raw_top": [...],
            "raw_all": [...]
        }
    }
    """
    timestamp = datetime.now(timezone.utc).isoformat()

    weather = get_weather(lat=lat, lon=lon, altitude=altitude)
    speed_limit = get_speed_limit(lat=lat, lon=lon, search_radius_m=speed_limit_radius_m)

    return {
        "timestamp": timestamp,
        "gps": {
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
        },
        "weather": weather,
        "speed_limit": speed_limit,
        "camera": camera_output,
    }


# ---------------------------------------------------------------------------
# CLI quick-test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import json

    # Default: city centre of Trondheim
    result = collect_pipeline_input(lat=63.4305, lon=10.3951, altitude=20)
    print(json.dumps(result, indent=2, ensure_ascii=False))
