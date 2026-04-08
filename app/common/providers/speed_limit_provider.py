from __future__ import annotations

import threading
from typing import Any, Dict, Optional

import requests
from pyproj import Transformer


class SpeedLimitProvider:
    """Resolve speed limit from Norwegian NVDB for a coordinate."""

    NVDB_BASE_URL = "https://nvdbapiles.atlas.vegvesen.no"
    NVDB_POSISJON_URL = f"{NVDB_BASE_URL}/vegnett/api/v4/posisjon"
    NVDB_OBJEKT_URL = f"{NVDB_BASE_URL}/vegobjekter/api/v4/vegobjekter/105"
    NVDB_HEADERS = {
        "X-Client": "NTNU_EiT_StudentProject",
        "Accept": "application/vnd.vegvesen.nvdb-v4+json",
    }
    MAX_CANDIDATES = 5
    MAX_DISTANCE_M = 40
    LOYALTY_MAX_DISTANCE_M = 30.0
    SWITCH_ADVANTAGE_M = 10.0

    def __init__(self) -> None:
        self.transformer = Transformer.from_crs("EPSG:4326", "EPSG:5973", always_xy=True)
        self._state_lock = threading.Lock()
        self._last_veglenke_id: Optional[int] = None

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

    @staticmethod
    def _extract_veglenke_id(match: Dict[str, Any]) -> Optional[int]:
        vls = match.get("veglenkesekvens", {})
        value = vls.get("veglenkesekvensid")
        try:
            return int(value)
        except (TypeError, ValueError):
            return None

    @staticmethod
    def _distance_m(match: Dict[str, Any], default: float = 1e6) -> float:
        try:
            return float(match.get("avstand", default))
        except (TypeError, ValueError):
            return default

    def _select_match(self, candidates: list[Dict[str, Any]], last_veglenke_id: Optional[int]) -> Dict[str, Any]:
        closest = candidates[0]
        if last_veglenke_id is None:
            return closest

        loyal = next(
            (m for m in candidates if self._extract_veglenke_id(m) == last_veglenke_id),
            None,
        )
        if loyal is None:
            return closest

        dist_old = self._distance_m(loyal)
        dist_new = self._distance_m(closest, default=0.0)
        if dist_old < self.LOYALTY_MAX_DISTANCE_M and (dist_old - dist_new) < self.SWITCH_ADVANTAGE_M:
            return loyal
        return closest

    def _speed_for_match(self, match: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        vls = match.get("veglenkesekvens", {})
        vls_id = vls.get("veglenkesekvensid")
        rel_pos = vls.get("relativPosisjon")
        if vls_id is None or rel_pos is None:
            return None

        obj_params = {
            "veglenkesekvens": f"{rel_pos}@{vls_id}",
            "inkluder": "egenskaper",
            "srid": 5973,
        }
        try:
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
            speed = self._extract_speed(objects[0])
            if speed is None:
                return None

            road_ref = (match.get("vegsystemreferanse", {}) or {}).get("kortform") or "unknown"
            distance_m = match.get("avstand")
            return {
                "speed_limit": speed,
                "source": "nvdb_smart",
                "road_ref": road_ref,
                "distance_m": distance_m,
                "veglenke_id": self._extract_veglenke_id(match),
            }
        except Exception:
            return None

    def _update_last_veglenke_id(self, value: Optional[int]) -> None:
        if value is None:
            return
        with self._state_lock:
            self._last_veglenke_id = value

    def lookup(self, lat: float, lon: float) -> Optional[Dict[str, Any]]:
        try:
            ost, nord = self.transformer.transform(lon, lat)
        except Exception:
            return None

        pos_params = {
            "nord": nord,
            "ost": ost,
            "srid": 5973,
            "maks_avstand": self.MAX_DISTANCE_M,
            "maks_antall": self.MAX_CANDIDATES,
            "trafikantgruppe": "K",
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

            with self._state_lock:
                last_veglenke_id = self._last_veglenke_id
            match = self._select_match(pos_data, last_veglenke_id)
            smart_result = self._speed_for_match(match)
            if smart_result is not None:
                self._update_last_veglenke_id(smart_result.get("veglenke_id"))
                smart_result.pop("veglenke_id", None)
                return smart_result

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
                    self._update_last_veglenke_id(self._extract_veglenke_id(match))
                    return {
                        "speed_limit": speed,
                        "source": "nvdb",
                        "road_ref": road_ref,
                        "distance_m": distance_m,
                    }

            if fallback_speed is None:
                return None

            self._update_last_veglenke_id(self._extract_veglenke_id(match))
            return {
                "speed_limit": fallback_speed,
                "source": "nvdb_nearest",
                "road_ref": road_ref,
                "distance_m": distance_m,
            }
        except Exception:
            return None
