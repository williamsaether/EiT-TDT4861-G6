from __future__ import annotations

from typing import Any, Dict, Optional

import requests
from pyproj import Transformer


class SpeedLimitService:
    """Resolve speed limit from Norwegian NVDB for a coordinate."""

    NVDB_BASE_URL = "https://nvdbapiles.atlas.vegvesen.no"
    NVDB_POSISJON_URL = f"{NVDB_BASE_URL}/vegnett/api/v4/posisjon"
    NVDB_OBJEKT_URL = f"{NVDB_BASE_URL}/vegobjekter/api/v4/vegobjekter/105"
    NVDB_HEADERS = {
        "X-Client": "NTNU_EiT_StudentProject",
        "Accept": "application/vnd.vegvesen.nvdb-v4+json",
    }

    def __init__(self) -> None:
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:5973", always_xy=True)

    @staticmethod
    def _extract_speed(obj: Dict[str, Any]) -> Optional[int]:
        for prop in obj.get("egenskaper", []):
            if prop.get("id") == 2021:
                try:
                    return int(float(prop["verdi"]))
                except (KeyError, ValueError, TypeError):
                    return None
        return None

    @staticmethod
    def _matches_road(obj: Dict[str, Any], vegkategori: str | None, veinummer: int | None) -> bool:
        if not vegkategori or veinummer is None:
            return True

        for vref in obj.get("lokasjon", {}).get("vegsystemreferanser", []):
            v_sys = vref.get("vegsystem", {})
            if v_sys.get("vegkategori") == vegkategori and v_sys.get("nummer") == veinummer:
                return True
        return False

    def lookup(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        try:
            ost, nord = self.transformer.transform(lon, lat)
        except Exception:
            return None

        pos_params = {
            "nord": nord,
            "ost": ost,
            "srid": 5973,
            "maks_avstand": 150,
            "maks_antall": 1,
        }

        try:
            pos_resp = requests.get(
                self.NVDB_POSISJON_URL,
                params=pos_params,
                headers=self.NVDB_HEADERS,
                timeout=4,
            )
            pos_resp.raise_for_status()
            pos_data = pos_resp.json()
            if not pos_data:
                return None

            match = pos_data[0]
            vsys = match.get("vegsystemreferanse", {})
            vegsystem = vsys.get("vegsystem", {})
            vegkategori = vegsystem.get("vegkategori")
            veinummer = vegsystem.get("nummer")
            road_ref = vsys.get("kortform") or "unknown"
            distance_m = match.get("avstand")

            size = 50
            obj_params = {
                "kartutsnitt": f"{ost-size},{nord-size},{ost+size},{nord+size}",
                "srid": 5973,
                "inkluder": "egenskaper,lokasjon",
                "antall": 20,
            }

            obj_resp = requests.get(
                self.NVDB_OBJEKT_URL,
                params=obj_params,
                headers=self.NVDB_HEADERS,
                timeout=4,
            )
            obj_resp.raise_for_status()
            objects = (obj_resp.json() or {}).get("objekter", [])
            if not objects:
                return None

            fallback_speed: Optional[int] = None
            for obj in objects:
                speed = self._extract_speed(obj)
                if speed is None:
                    continue

                if fallback_speed is None:
                    fallback_speed = speed

                if self._matches_road(obj, vegkategori, veinummer):
                    return {
                        "speed_limit": speed,
                        "source": "nvdb",
                        "road_ref": road_ref,
                        "distance_m": distance_m,
                    }

            if fallback_speed is None:
                return None

            return {
                "speed_limit": fallback_speed,
                "source": "nvdb_nearest",
                "road_ref": road_ref,
                "distance_m": distance_m,
            }
        except Exception:
            return None
