from __future__ import annotations

import os
import threading
import time
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import requests
from flask import Flask, jsonify, render_template, request, send_from_directory
from werkzeug.utils import secure_filename

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNX = False

try:
    from pyproj import Transformer

    HAS_PYPROJ = True
except ImportError:
    Transformer = None
    HAS_PYPROJ = False


app = Flask(__name__, static_folder="static", template_folder="templates")

ROOT = Path(__file__).resolve().parent.parent
UPLOAD_DIR = Path(__file__).resolve().parent / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = ROOT / "rscd_resnet18_v2.onnx"
IMAGE_SIZE = 224
MAX_EVENTS = 300

# When no camera is available or confidence is very low, we fall back to this
# factor (slightly cautious but not penalizing). 1.0 = no reduction from camera.
NEUTRAL_CAMERA_FACTOR = 0.88

WEATHER_API_URL = "https://api.open-meteo.com/v1/forecast"
NVDB_BASE = "https://nvdbapiles.atlas.vegvesen.no"
NVDB_POSISJON_URL = f"{NVDB_BASE}/vegnett/api/v4/posisjon"
NVDB_FARTSGRENSE_URL = f"{NVDB_BASE}/vegobjekter/api/v4/vegobjekter/105"
NVDB_HEADERS = {
    "X-Client": "NTNU_EiT_Dashboard",
    "Accept": "application/vnd.vegvesen.nvdb-v4+json",
}

WGS84_TO_5973 = Transformer.from_crs("EPSG:4326", "EPSG:5973", always_xy=True) if HAS_PYPROJ else None

# v2 model: 12 classes — friction × surface, plus three winter states
WINTER_CLASSES = ["fresh_snow", "melted_snow", "ice"]
FRICTION_CLASSES = ["dry", "wet", "water"]
SURFACE_CLASSES = ["asphalt", "gravel", "mud"]

CATEGORY_LABELS = {
    "friction": FRICTION_CLASSES,
    "surface": SURFACE_CLASSES,
    "winter": WINTER_CLASSES,
}

# Ordered exactly as the v2 checkpoint's class_to_idx (alphabetical)
CLASS_NAMES = [
    "dry_asphalt",
    "dry_gravel",
    "dry_mud",
    "fresh_snow",
    "ice",
    "melted_snow",
    "water_asphalt",
    "water_gravel",
    "water_mud",
    "wet_asphalt",
    "wet_gravel",
    "wet_mud",
]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32) if HAS_NUMPY else None
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32) if HAS_NUMPY else None

# Difficulty 1 (ideal) → 5 (extreme). Default 3 for unknown classes.
DIFFICULTY_LEVEL = {
    "dry_asphalt":  1,
    "wet_asphalt":  2,
    "melted_snow":  3,
    "water_asphalt": 3,
    "dry_gravel":   3,
    "wet_gravel":   4,
    "water_gravel": 4,
    "dry_mud":      4,
    "wet_mud":      4,
    "fresh_snow":   4,
    "water_mud":    5,
    "ice":          5,
}
DIFFICULTY_FACTOR = {1: 1.0, 2: 0.9, 3: 0.7, 4: 0.5, 5: 0.35}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sorted_pairs(scores: Dict[str, float]) -> List[List[Any]]:
    ordered = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [[k, float(v)] for k, v in ordered]


def parse_groups(label_name: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """Return (friction, surface, winter) for a v2 class label."""
    if label_name in WINTER_CLASSES:
        return None, None, label_name

    parts = label_name.split("_", 1)
    friction = parts[0] if parts[0] in FRICTION_CLASSES else None
    surface = parts[1] if len(parts) > 1 and parts[1] in SURFACE_CLASSES else None
    return friction, surface, None


@dataclass
class WeatherSnapshot:
    temperature_c: float
    precipitation_mm_h: float
    wind_speed_kmh: float
    visibility_m: float
    weather_code: int
    status: str
    fetched_at: str


@dataclass
class SpeedSnapshot:
    legal_speed_limit_kmh: Optional[int]
    road_reference: Optional[str]
    distance_m: Optional[float]
    status: str
    fetched_at: str


@dataclass
class CameraSnapshot:
    top_label: str
    confidence: float
    friction: Optional[str]
    surface: Optional[str]
    winter: Optional[str]
    difficulty_level: int
    risk_factor: float
    top_3: List[List[Any]]
    raw_scores: Dict[str, float]
    grouped: Dict[str, List[List[Any]]]
    status: str
    fetched_at: str


class DashboardController:
    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.stop_event = threading.Event()
        self.thread: Optional[threading.Thread] = None

        self.location = {"lat": 63.4305, "lon": 10.3951, "altitude": 20.0}
        self.intervals = {"weather": 300.0, "speed": 10.0, "camera": 1.0, "decision": 2.0}
        self.weight_inputs = {"weather": 45.0, "camera": 45.0, "confidence": 10.0}
        self.weights = {"weather": 0.45, "camera": 0.45, "confidence": 0.10}

        self.default_speed_limit = 80
        self.min_speed_limit = 20
        self.user_max_speed_limit: Optional[int] = None

        # Rolling window: camera predictions within this many seconds are averaged
        # before computing the camera factor. Tunable via the dashboard slider.
        self.camera_window_seconds: float = 3.0
        # Each entry: (monotonic_time, raw_scores_dict)
        self.camera_score_history: List[Tuple[float, Dict[str, float]]] = []

        self.video_filename: Optional[str] = None
        self.video_path: Optional[Path] = None
        self.capture = None

        self.latest_weather: Optional[WeatherSnapshot] = None
        self.latest_speed: Optional[SpeedSnapshot] = None
        self.latest_camera: Optional[CameraSnapshot] = None
        self.latest_packet: Dict[str, Any] = {}

        self.prediction_events: List[Dict[str, Any]] = []
        self.camera_probability_sums: Dict[str, Dict[str, float]] = {
            cat: {label: 0.0 for label in labels} for cat, labels in CATEGORY_LABELS.items()
        }

        self.model_session = None
        self.model_input_name: Optional[str] = None
        self.model_status = "not_loaded"
        self._load_model()

    def _load_model(self) -> None:
        if not HAS_ONNX or not HAS_NUMPY:
            self.model_status = "fallback: missing onnxruntime/numpy"
            return
        if not MODEL_PATH.exists():
            self.model_status = f"fallback: missing model at {MODEL_PATH}"
            return
        try:
            self.model_session = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
            self.model_input_name = self.model_session.get_inputs()[0].name
            self.model_status = "ok"
        except Exception as exc:
            self.model_status = f"fallback: model load error ({exc})"

    def is_running(self) -> bool:
        return bool(self.thread and self.thread.is_alive())

    def set_location(self, lat: float, lon: float, altitude: float) -> None:
        self.location = {"lat": float(lat), "lon": float(lon), "altitude": float(altitude)}

    def set_user_max_speed_limit(self, value: Optional[Any]) -> Optional[int]:
        if value in (None, "", 0, "0"):
            self.user_max_speed_limit = None
            return None
        parsed = int(value)
        if parsed <= 0:
            raise ValueError("max_speed_limit must be > 0")
        self.user_max_speed_limit = parsed
        return parsed

    def set_weights(self, payload: Dict[str, Any]) -> Dict[str, float]:
        weather = float(payload.get("weather", self.weight_inputs["weather"]))
        camera = float(payload.get("camera", self.weight_inputs["camera"]))
        confidence = float(payload.get("confidence", self.weight_inputs["confidence"]))
        if weather < 0 or camera < 0 or confidence < 0:
            raise ValueError("weights must be >= 0")

        total = weather + camera + confidence
        if total <= 0:
            raise ValueError("sum of weights must be > 0")

        self.weight_inputs = {
            "weather": weather,
            "camera": camera,
            "confidence": confidence,
        }
        self.weights = {
            "weather": weather / total,
            "camera": camera / total,
            "confidence": confidence / total,
        }
        return self.weights

    def set_camera_window(self, seconds: float) -> float:
        if seconds < 0.5:
            raise ValueError("camera_window_seconds must be >= 0.5")
        self.camera_window_seconds = float(seconds)
        return self.camera_window_seconds

    def _get_windowed_camera_data(self) -> Optional[Dict[str, Any]]:
        """
        Average raw class scores across all predictions within the rolling window.
        Returns a dict with averaged top_3, top_label, mean_confidence, and frame_count,
        or None if there are no frames in the window.
        """
        now = time.monotonic()
        cutoff = now - self.camera_window_seconds
        window = [(t, s) for t, s in self.camera_score_history if t >= cutoff]
        if not window:
            return None

        # Average probabilities across all frames in the window
        avg_scores: Dict[str, float] = {cls: 0.0 for cls in CLASS_NAMES}
        for _, scores in window:
            for cls, p in scores.items():
                avg_scores[cls] = avg_scores.get(cls, 0.0) + p
        n = len(window)
        avg_scores = {cls: v / n for cls, v in avg_scores.items()}

        # Derive top label, confidence, and top-3 from averaged scores
        sorted_avg = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_label = sorted_avg[0][0]
        top_conf = sorted_avg[0][1]
        top_3 = [[cls, round(p, 4)] for cls, p in sorted_avg[:3]]

        return {
            "top_label": top_label,
            "confidence": top_conf,
            "top_3": top_3,
            "frame_count": n,
            "window_seconds": self.camera_window_seconds,
        }

    def recompute_latest_packet(self) -> None:
        self.latest_packet = self._recommend_packet()

    def _reset_camera_distribution(self) -> None:
        self.prediction_events = []
        self.camera_score_history = []
        self.camera_probability_sums = {cat: {label: 0.0 for label in labels} for cat, labels in CATEGORY_LABELS.items()}

    def set_video(self, filename: str) -> None:
        self.video_filename = filename
        self.video_path = UPLOAD_DIR / filename
        if self.capture is not None:
            self.capture.release()
            self.capture = None
        self.latest_camera = None
        self._reset_camera_distribution()

    def start(self) -> None:
        if self.is_running():
            return
        self.stop_event.clear()
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread:
            self.thread.join(timeout=2)
        if self.capture is not None:
            self.capture.release()
            self.capture = None

    def _normalized_accumulated_camera_distribution(self) -> Dict[str, List[List[Any]]]:
        dist: Dict[str, List[List[Any]]] = {}
        for category, labels in CATEGORY_LABELS.items():
            sums = self.camera_probability_sums[category]
            total = sum(sums.values())
            if total <= 0:
                dist[category] = [[label, 0.0] for label in labels]
            else:
                normalized = {label: sums[label] / total for label in labels}
                dist[category] = sorted_pairs(normalized)
        return dist

    def snapshot(self) -> Dict[str, Any]:
        windowed = self._get_windowed_camera_data()
        return {
            "running": self.is_running(),
            "model_status": self.model_status,
            "location": self.location,
            "weight_inputs": self.weight_inputs,
            "weights": self.weights,
            "intervals": self.intervals,
            "camera_window_seconds": self.camera_window_seconds,
            "windowed_camera": windowed,
            "video": {
                "filename": self.video_filename,
                "url": f"/uploads/{self.video_filename}" if self.video_filename else None,
            },
            "user_max_speed_limit": self.user_max_speed_limit,
            "latest_weather": asdict(self.latest_weather) if self.latest_weather else None,
            "latest_speed": asdict(self.latest_speed) if self.latest_speed else None,
            "latest_camera": asdict(self.latest_camera) if self.latest_camera else None,
            "latest_packet": self.latest_packet,
            "camera_distribution": self._normalized_accumulated_camera_distribution(),
            "recent_predictions": self.prediction_events,
        }

    def _open_capture(self) -> bool:
        if not HAS_CV2:
            return False
        if self.capture is not None and self.capture.isOpened():
            return True
        if self.video_path is None:
            return False
        self.capture = cv2.VideoCapture(str(self.video_path))
        return bool(self.capture is not None and self.capture.isOpened())

    def _next_frame(self):
        if not self._open_capture():
            return None
        ok, frame = self.capture.read()
        if not ok:
            self.capture.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ok, frame = self.capture.read()
        return frame if ok else None

    def _preprocess_frame(self, frame) -> "np.ndarray":
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        arr = rgb.astype(np.float32) / 255.0
        arr = (arr - MEAN) / STD
        arr = np.transpose(arr, (2, 0, 1))
        arr = np.expand_dims(arr, axis=0)
        return arr

    def _predict_camera(self) -> CameraSnapshot:
        empty_grouped = {
            "friction": [[k, 0.0] for k in FRICTION_CLASSES],
            "surface": [[k, 0.0] for k in SURFACE_CLASSES],
            "winter": [[k, 0.0] for k in WINTER_CLASSES],
        }

        def _make_empty(status_msg: str) -> CameraSnapshot:
            return CameraSnapshot(
                top_label="unknown",
                confidence=0.0,
                friction=None,
                surface=None,
                winter=None,
                difficulty_level=3,
                risk_factor=DIFFICULTY_FACTOR[3],
                top_3=[],
                grouped=empty_grouped,
                status=status_msg,
                fetched_at=utc_now(),
            )

        if self.video_path is None:
            return _make_empty("disabled: no video uploaded")
        if not HAS_CV2 or not HAS_NUMPY:
            return _make_empty("error: opencv-python or numpy missing")

        frame = self._next_frame()
        if frame is None:
            return _make_empty("error: unable to read frame")

        if self.model_session is None or self.model_input_name is None:
            label = "dry_asphalt"
            confidence = 0.55
            raw_scores = {"dry_asphalt": 0.55, "wet_asphalt": 0.30, "dry_gravel": 0.15,
                          **{c: 0.0 for c in CLASS_NAMES if c not in ("dry_asphalt", "wet_asphalt", "dry_gravel")}}
            top_3 = [["dry_asphalt", 0.55], ["wet_asphalt", 0.30], ["dry_gravel", 0.15]]
            grouped = {
                "friction": [["dry", 1.0], ["wet", 0.0], ["water", 0.0]],
                "surface": [["asphalt", 1.0], ["gravel", 0.0], ["mud", 0.0]],
                "winter": [["fresh_snow", 0.0], ["melted_snow", 0.0], ["ice", 0.0]],
            }
            status = self.model_status
        else:
            x = self._preprocess_frame(frame)
            logits = self.model_session.run(None, {self.model_input_name: x})[0][0]
            logits = logits - np.max(logits)
            probs = np.exp(logits)
            probs = probs / probs.sum()

            raw_scores = {CLASS_NAMES[i]: float(p) for i, p in enumerate(probs)}
            sorted_scores = sorted(raw_scores.items(), key=lambda x: x[1], reverse=True)
            label = sorted_scores[0][0]
            confidence = sorted_scores[0][1]
            top_3 = [[cls, round(prob, 4)] for cls, prob in sorted_scores[:3]]

            friction_scores = {k: 0.0 for k in FRICTION_CLASSES}
            surface_scores = {k: 0.0 for k in SURFACE_CLASSES}
            winter_scores = {k: 0.0 for k in WINTER_CLASSES}

            for class_name, score in raw_scores.items():
                friction, surface, winter = parse_groups(class_name)
                if friction is not None:
                    friction_scores[friction] += score
                if surface is not None:
                    surface_scores[surface] += score
                if winter is not None:
                    winter_scores[winter] += score

            grouped = {
                "friction": sorted_pairs(friction_scores),
                "surface": sorted_pairs(surface_scores),
                "winter": sorted_pairs(winter_scores),
            }
            status = "ok:model"

        friction, surface, winter = parse_groups(label)
        difficulty = DIFFICULTY_LEVEL.get(label, 3)
        risk = DIFFICULTY_FACTOR[difficulty]

        return CameraSnapshot(
            top_label=label,
            confidence=confidence,
            friction=friction,
            surface=surface,
            winter=winter,
            difficulty_level=difficulty,
            risk_factor=risk,
            top_3=top_3,
            raw_scores=raw_scores,
            grouped=grouped,
            status=status,
            fetched_at=utc_now(),
        )

    def _fetch_weather(self) -> WeatherSnapshot:
        loc = self.location
        params = {
            "latitude": loc["lat"],
            "longitude": loc["lon"],
            "current": "temperature_2m,precipitation,wind_speed_10m,visibility,weather_code",
            "timezone": "auto",
            "elevation": loc["altitude"],
        }
        try:
            res = requests.get(WEATHER_API_URL, params=params, timeout=10)
            res.raise_for_status()
            current = res.json().get("current", {})
            return WeatherSnapshot(
                temperature_c=float(current.get("temperature_2m", 10.0)),
                precipitation_mm_h=float(current.get("precipitation", 0.0)),
                wind_speed_kmh=float(current.get("wind_speed_10m", 0.0)),
                visibility_m=float(current.get("visibility", 10000.0)),
                weather_code=int(current.get("weather_code", 0)),
                status="ok",
                fetched_at=utc_now(),
            )
        except Exception as exc:
            return WeatherSnapshot(
                temperature_c=10.0,
                precipitation_mm_h=0.0,
                wind_speed_kmh=0.0,
                visibility_m=10000.0,
                weather_code=0,
                status=f"error: {exc}",
                fetched_at=utc_now(),
            )

    def _fetch_speed_limit(self) -> SpeedSnapshot:
        if not HAS_PYPROJ or WGS84_TO_5973 is None:
            return SpeedSnapshot(
                legal_speed_limit_kmh=None,
                road_reference=None,
                distance_m=None,
                status="error: pyproj missing",
                fetched_at=utc_now(),
            )

        lat = self.location["lat"]
        lon = self.location["lon"]
        try:
            east, north = WGS84_TO_5973.transform(lon, lat)
            pos_params = {
                "nord": north,
                "ost": east,
                "srid": 5973,
                "maks_avstand": 150,
                "maks_antall": 1,
            }
            pos_resp = requests.get(NVDB_POSISJON_URL, params=pos_params, headers=NVDB_HEADERS, timeout=10)
            pos_resp.raise_for_status()
            roads = pos_resp.json()

            if not roads:
                return SpeedSnapshot(None, None, None, "not_found", utc_now())

            nearest = roads[0]
            distance = nearest.get("avstand")
            road_ref = nearest.get("vegsystemreferanse", {}).get("kortform")

            size = 60
            bbox = f"{east-size},{north-size},{east+size},{north+size}"
            obj_params = {
                "kartutsnitt": bbox,
                "srid": 5973,
                "inkluder": "egenskaper,lokasjon",
                "antall": 20,
            }
            obj_resp = requests.get(NVDB_FARTSGRENSE_URL, params=obj_params, headers=NVDB_HEADERS, timeout=10)
            obj_resp.raise_for_status()
            objects = obj_resp.json().get("objekter", [])

            for obj in objects:
                for prop in obj.get("egenskaper", []):
                    if prop.get("id") == 2021 and prop.get("verdi") is not None:
                        return SpeedSnapshot(
                            legal_speed_limit_kmh=int(prop["verdi"]),
                            road_reference=road_ref,
                            distance_m=float(distance) if distance is not None else None,
                            status="ok",
                            fetched_at=utc_now(),
                        )

            return SpeedSnapshot(
                legal_speed_limit_kmh=None,
                road_reference=road_ref,
                distance_m=float(distance) if distance is not None else None,
                status="not_found",
                fetched_at=utc_now(),
            )
        except Exception as exc:
            return SpeedSnapshot(None, None, None, f"error: {exc}", utc_now())

    def _weather_factor(self) -> Tuple[float, List[str], Dict[str, float]]:
        if self.latest_weather is None:
            return 0.85, ["weather unavailable"], {}

        w = self.latest_weather
        factor = 1.0
        reasons: List[str] = []

        if w.precipitation_mm_h > 10:
            factor *= 0.5
            reasons.append("heavy rain")
        elif w.precipitation_mm_h > 5:
            factor *= 0.7
            reasons.append("moderate rain")
        elif w.precipitation_mm_h > 0.5:
            factor *= 0.85
            reasons.append("light rain")

        if w.temperature_c < 0:
            factor *= 0.6
            reasons.append("freezing")
        elif w.temperature_c < 3:
            factor *= 0.8
            reasons.append("near freezing")

        if w.wind_speed_kmh > 70:
            factor *= 0.6
            reasons.append("strong wind")
        elif w.wind_speed_kmh > 50:
            factor *= 0.8
            reasons.append("moderate wind")

        if w.visibility_m < 50:
            factor *= 0.4
            reasons.append("very poor visibility")
        elif w.visibility_m < 200:
            factor *= 0.6
            reasons.append("poor visibility")
        elif w.visibility_m < 1000:
            factor *= 0.8
            reasons.append("reduced visibility")

        features = {
            "temperature_c": w.temperature_c,
            "precipitation_mm_h": w.precipitation_mm_h,
            "wind_speed_kmh": w.wind_speed_kmh,
            "visibility_m": w.visibility_m,
        }
        return factor, reasons, features

    def _recommend_packet(self) -> Dict[str, Any]:
        weather_factor, weather_reasons, weather_features = self._weather_factor()

        # --- Camera factor: use rolling window average when available ---
        windowed = self._get_windowed_camera_data()
        if windowed:
            top_3 = windowed["top_3"]
            total_p = sum(float(p) for _, p in top_3)
            if total_p > 0:
                raw_cam_factor = sum(
                    (float(p) / total_p) * DIFFICULTY_FACTOR[DIFFICULTY_LEVEL.get(str(cls), 3)]
                    for cls, p in top_3
                )
            else:
                raw_cam_factor = NEUTRAL_CAMERA_FACTOR
            cam_conf = max(0.05, windowed["confidence"])
            cam_reason = (
                f"{windowed['top_label'].replace('_', ' ')} "
                f"({windowed['frame_count']}f/{windowed['window_seconds']:.1f}s)"
            )
        else:
            # No frames in window (no video or window just started)
            raw_cam_factor = NEUTRAL_CAMERA_FACTOR
            cam_conf = 0.0
            cam_reason = "camera unavailable"

        # --- Confidence-blended effective camera factor ---
        # conf_sensitivity (weight["confidence"], 0–1) controls how strongly
        # low model confidence dampens the camera reading toward NEUTRAL_CAMERA_FACTOR.
        # conf_sens=0 → raw factor used regardless of confidence (trust model always)
        # conf_sens=1, cam_conf=0 → use NEUTRAL_CAMERA_FACTOR
        # conf_sens=1, cam_conf=1 → use raw factor (perfect confidence)
        conf_sens = self.weights["confidence"]
        effective_cam = raw_cam_factor + conf_sens * (1.0 - cam_conf) * (NEUTRAL_CAMERA_FACTOR - raw_cam_factor)

        # --- Additive weighted combination ---
        # Combined = (w_weather * weather_factor + w_camera * effective_cam) / (w_weather + w_camera)
        # Weights are normalized between weather and camera; confidence changes the camera input,
        # not the blend ratio, so the sliders mean exactly what users expect.
        w = self.weights
        w_sum = max(w["weather"] + w["camera"], 1e-6)
        combined = (w["weather"] * weather_factor + w["camera"] * effective_cam) / w_sum

        legal_from_api = (
            self.latest_speed.legal_speed_limit_kmh
            if self.latest_speed and self.latest_speed.legal_speed_limit_kmh
            else self.default_speed_limit
        )
        effective_max = self.user_max_speed_limit if self.user_max_speed_limit is not None else legal_from_api

        recommended = int(round((effective_max * combined) / 10.0) * 10)
        recommended = max(self.min_speed_limit, min(recommended, effective_max))
        reduction_pct = ((effective_max - recommended) / max(effective_max, 1)) * 100.0

        # --- Per-factor hypothetical speeds (for UI visualization) ---
        def _clamp_speed(f: float) -> int:
            return max(self.min_speed_limit, min(int(round((effective_max * f) / 10.0) * 10), effective_max))

        weather_only_kmh = _clamp_speed(weather_factor)
        camera_only_kmh = _clamp_speed(effective_cam)

        # --- Factor contributions for donut chart ---
        # Each factor's share of the total speed reduction (in km/h from effective_max)
        w_weather_norm = w["weather"] / w_sum
        w_camera_norm = w["camera"] / w_sum
        weather_reduction_kmh = w_weather_norm * (1.0 - weather_factor) * effective_max
        camera_reduction_kmh = w_camera_norm * (1.0 - effective_cam) * effective_max
        no_reduction_kmh = max(0.0, effective_max - weather_reduction_kmh - camera_reduction_kmh - self.min_speed_limit)

        reasons = weather_reasons + [cam_reason]

        features = {
            "weather_factor": round(weather_factor, 4),
            "raw_camera_factor": round(raw_cam_factor, 4),
            "effective_camera_factor": round(effective_cam, 4),
            "camera_confidence": round(cam_conf, 4),
            "combined_factor": round(combined, 4),
            "difficult_conditions": bool(weather_factor < 0.9 or effective_cam < 0.9),
        }
        features.update(weather_features)

        return {
            "timestamp": utc_now(),
            "gps": self.location,
            "weights": self.weights,
            "weather": asdict(self.latest_weather) if self.latest_weather else None,
            "speed_limit": asdict(self.latest_speed) if self.latest_speed else None,
            "camera": asdict(self.latest_camera) if self.latest_camera else None,
            "features": features,
            "recommendation": {
                "legal_speed_limit_kmh": legal_from_api,
                "user_max_speed_limit_kmh": self.user_max_speed_limit,
                "effective_max_speed_limit_kmh": effective_max,
                "recommended_speed_limit_kmh": recommended,
                "reduction_pct": round(reduction_pct, 1),
                "reason": ", ".join(reasons),
                "weather_only_kmh": weather_only_kmh,
                "camera_only_kmh": camera_only_kmh,
                "factor_contributions": {
                    "weather_kmh": round(weather_reduction_kmh, 1),
                    "camera_kmh": round(camera_reduction_kmh, 1),
                },
            },
        }

    def _run_loop(self) -> None:
        next_weather = 0.0
        next_speed = 0.0
        next_camera = 0.0
        next_decision = 0.0

        while not self.stop_event.is_set():
            now = time.monotonic()

            weather_interval = float(self.intervals["weather"])
            speed_interval = float(self.intervals["speed"])
            camera_interval = float(self.intervals["camera"])
            decision_interval = float(self.intervals["decision"])

            if now >= next_weather:
                self.latest_weather = self._fetch_weather()
                next_weather = now + weather_interval

            if now >= next_speed:
                self.latest_speed = self._fetch_speed_limit()
                next_speed = now + speed_interval

            if now >= next_camera:
                camera = self._predict_camera()
                self.latest_camera = camera
                if camera.status.startswith("ok"):
                    for category in CATEGORY_LABELS:
                        for label, probability in camera.grouped[category]:
                            self.camera_probability_sums[category][label] += float(probability)

                    # Append to rolling window history
                    self.camera_score_history.append((now, camera.raw_scores))
                    # Prune entries older than the window (keep a generous 30 s buffer)
                    cutoff = now - max(self.camera_window_seconds, 30.0)
                    self.camera_score_history = [
                        (t, s) for t, s in self.camera_score_history if t >= cutoff
                    ]

                    self.prediction_events.append(
                        {
                            "timestamp": camera.fetched_at,
                            "label": camera.top_label,
                            "confidence": camera.confidence,
                        }
                    )
                    if len(self.prediction_events) > MAX_EVENTS:
                        self.prediction_events = self.prediction_events[-MAX_EVENTS:]
                next_camera = now + camera_interval

            if now >= next_decision:
                self.latest_packet = self._recommend_packet()
                next_decision = now + decision_interval

            time.sleep(0.05)

        if self.capture is not None:
            self.capture.release()
            self.capture = None


controller = DashboardController()


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/uploads/<path:filename>")
def uploads(filename: str):
    return send_from_directory(UPLOAD_DIR, filename)


@app.route("/api/upload-video", methods=["POST"])
def upload_video():
    if "video" not in request.files:
        return jsonify({"error": "missing file field 'video'"}), 400

    file = request.files["video"]
    if not file.filename:
        return jsonify({"error": "empty filename"}), 400

    filename = secure_filename(file.filename)
    if not filename:
        return jsonify({"error": "invalid filename"}), 400

    save_path = UPLOAD_DIR / filename
    file.save(save_path)

    with controller.lock:
        controller.set_video(filename)

    return jsonify({"status": "ok", "filename": filename, "video_url": f"/uploads/{filename}"})


@app.route("/api/start", methods=["POST"])
def start_pipeline():
    payload = request.get_json(silent=True) or {}

    try:
        lat = float(payload.get("lat", controller.location["lat"]))
        lon = float(payload.get("lon", controller.location["lon"]))
        altitude = float(payload.get("altitude", controller.location["altitude"]))
    except ValueError:
        return jsonify({"error": "lat/lon/altitude must be numeric"}), 400

    with controller.lock:
        controller.set_location(lat, lon, altitude)
        try:
            controller.set_user_max_speed_limit(payload.get("max_speed_limit"))
        except (TypeError, ValueError):
            return jsonify({"error": "max_speed_limit must be empty or a positive integer"}), 400
        controller.recompute_latest_packet()
        controller.start()
        location = dict(controller.location)
        max_speed = controller.user_max_speed_limit

    return jsonify({"status": "running", "location": location, "user_max_speed_limit": max_speed})


@app.route("/api/stop", methods=["POST"])
def stop_pipeline():
    with controller.lock:
        controller.stop()
    return jsonify({"status": "stopped"})


@app.route("/api/weights", methods=["GET", "POST"])
def weights_endpoint():
    if request.method == "GET":
        with controller.lock:
            weight_inputs = dict(controller.weight_inputs)
            weights = dict(controller.weights)
        return jsonify({"weight_inputs": weight_inputs, "weights": weights})

    payload = request.get_json(silent=True) or {}
    try:
        with controller.lock:
            weights = controller.set_weights(payload)
            weight_inputs = dict(controller.weight_inputs)
            controller.recompute_latest_packet()
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    return jsonify({"status": "ok", "weight_inputs": weight_inputs, "weights": weights})


@app.route("/api/config", methods=["POST"])
def config_endpoint():
    payload = request.get_json(silent=True) or {}

    with controller.lock:
        if "lat" in payload or "lon" in payload or "altitude" in payload:
            try:
                lat = float(payload.get("lat", controller.location["lat"]))
                lon = float(payload.get("lon", controller.location["lon"]))
                altitude = float(payload.get("altitude", controller.location["altitude"]))
            except ValueError:
                return jsonify({"error": "lat/lon/altitude must be numeric"}), 400
            controller.set_location(lat, lon, altitude)

        if "max_speed_limit" in payload:
            try:
                controller.set_user_max_speed_limit(payload.get("max_speed_limit"))
            except (TypeError, ValueError):
                return jsonify({"error": "max_speed_limit must be empty or a positive integer"}), 400

        if "camera_window_seconds" in payload:
            try:
                controller.set_camera_window(float(payload["camera_window_seconds"]))
            except (TypeError, ValueError) as exc:
                return jsonify({"error": str(exc)}), 400

        controller.recompute_latest_packet()
        snapshot = controller.snapshot()

    return jsonify({"status": "ok", "config": {
        "location": snapshot["location"],
        "user_max_speed_limit": snapshot["user_max_speed_limit"],
        "camera_window_seconds": snapshot["camera_window_seconds"],
    }})


@app.route("/api/status", methods=["GET"])
def status_endpoint():
    with controller.lock:
        return jsonify(controller.snapshot())


@app.route("/api/reset-histogram", methods=["POST"])
def reset_histogram():
    with controller.lock:
        controller._reset_camera_distribution()
    return jsonify({"status": "ok"})


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
