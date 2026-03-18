from __future__ import annotations

import csv
import os
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from flask import Flask, jsonify, render_template, request, send_from_directory

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


app = Flask(__name__, static_folder="static", template_folder="templates")

ROOT = Path(__file__).resolve().parent.parent
VIDEOS_DIR = ROOT / "videos"
CSV_PATH = ROOT / "merged_survey.csv"
MODELS_DIR = ROOT / "pipeline" / "models"
DEFAULT_MODEL_NAME = "rscd_resnet18_v2.onnx"
METADATA_PATH = VIDEOS_DIR / "metadata.json"


def _available_models() -> List[str]:
    """Return sorted list of .onnx filenames found in MODELS_DIR."""
    if not MODELS_DIR.exists():
        return [DEFAULT_MODEL_NAME]
    return sorted(p.name for p in MODELS_DIR.glob("*.onnx"))

IMAGE_SIZE = 224

# Crop region (as fractions of frame dimensions) used when road-crop mode is on.
# Strips sky/horizon (top 35%), hood/bumper (bottom 15%), and road shoulders (10% each side),
# leaving a rectangle focused on the road surface directly ahead.
CROP_TOP:   float = 0.35
CROP_BOTTOM: float = 0.85
CROP_LEFT:  float = 0.10
CROP_RIGHT: float = 0.90

CLASS_NAMES: List[str] = [
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

# Classes excluded from scoring/display — model outputs are zeroed and renormalised
# so only asphalt and snow/ice categories influence recommendations.
EXCLUDED_CLASSES: frozenset = frozenset([
    "dry_gravel",
    "dry_mud",
    "water_gravel",
    "water_mud",
    "wet_gravel",
    "wet_mud",
])

# Difficulty level per class (1 = safest, 5 = most dangerous)
DIFFICULTY_LEVEL: Dict[str, int] = {
    "dry_asphalt": 1,
    "wet_asphalt": 2,
    "melted_snow": 3,
    "water_asphalt": 3,
    "fresh_snow": 4,
    "ice": 5,
}

DIFFICULTY_NAMES: Dict[int, str] = {
    1: "Dry asphalt",
    2: "Wet asphalt",
    3: "Melted snow / water asphalt",
    4: "Fresh snow",
    5: "Ice",
}

CLASS_COLORS: Dict[str, str] = {
    "dry_asphalt":   "#4ade80",
    "wet_asphalt":   "#60a5fa",
    "fresh_snow":    "#e0f2fe",
    "melted_snow":   "#bae6fd",
    "ice":           "#93c5fd",
    "water_asphalt": "#2563eb",
}

# ── Metadata loading ───────────────────────────────────────────────────────────

def _load_video_metadata() -> Dict[str, Dict[str, Any]]:
    """Load metadata.json: keyed by video_id (v01–v10)."""
    if not METADATA_PATH.exists():
        return {}
    import json
    with open(METADATA_PATH, encoding="utf-8") as f:
        entries = json.load(f)
    result: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        # filename is like "v01.mp4" → strip extension for id
        raw = entry.get("video", "")
        vid_id = Path(raw).stem  # "v01"
        result[vid_id] = entry
    return result


def _compute_weather_factor(temp_c: float, precip_mm_h: float) -> Tuple[float, List[str]]:
    """Compute a [0,1] reduction factor using default thresholds (used at startup only)."""
    factor = 1.0
    reasons: List[str] = []
    if precip_mm_h > 10:
        factor *= 0.50; reasons.append("heavy rain/snow")
    elif precip_mm_h > 5:
        factor *= 0.70; reasons.append("moderate rain/snow")
    elif precip_mm_h > 0.5:
        factor *= 0.85; reasons.append("light rain/snow")
    if temp_c < 0:
        factor *= 0.60; reasons.append("freezing")
    elif temp_c < 3:
        factor *= 0.80; reasons.append("near freezing")
    return round(factor, 4), reasons


# ── Survey data loading ─────────────────────────────────────────────────────────

def _load_survey_targets() -> Dict[str, Dict[str, Any]]:
    """Compute per-video mean recommended speed (rounded to nearest 10) from CSV."""
    buckets: Dict[str, Dict[str, Any]] = {}
    if not CSV_PATH.exists():
        return {}
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            vid = row.get("video_id", "").strip()
            try:
                speed = float(row["recommended_speed"])
                posted = int(float(row["posted_speed_limit"]))
            except (ValueError, KeyError):
                continue
            if vid not in buckets:
                buckets[vid] = {"speeds": [], "posted": posted}
            buckets[vid]["speeds"].append(speed)

    result: Dict[str, Dict[str, Any]] = {}
    for vid, data in buckets.items():
        mean = sum(data["speeds"]) / len(data["speeds"])
        target = int(round(mean / 10) * 10)
        result[vid] = {
            "mean_raw": round(mean, 1),
            "target": target,
            "posted_speed": data["posted"],
            "n_responses": len(data["speeds"]),
            "speeds": data["speeds"],
        }
    return result


def _parse_video_filename(filename: str) -> Dict[str, Any]:
    """Parse filename: {difficulty}-{lat}-{lon}-{speed}.mp4"""
    stem = Path(filename).stem
    parts = stem.split("-")
    try:
        difficulty = parts[0]
        lat = float(parts[1])
        lon = float(parts[2])
        posted_speed = int(parts[3])
    except (IndexError, ValueError):
        difficulty = "unknown"
        lat, lon, posted_speed = 0.0, 0.0, 80
    return {"difficulty": difficulty, "lat": lat, "lon": lon, "posted_speed": posted_speed}


def _build_video_list(
    targets: Dict[str, Dict[str, Any]],
    vid_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not VIDEOS_DIR.exists():
        return []
    files = sorted(f.name for f in VIDEOS_DIR.glob("*.mp4"))
    videos = []
    for i, filename in enumerate(files):
        parsed = _parse_video_filename(filename)
        vid_id = f"v{i + 1:02d}"

        # Prefer metadata.json values over filename-parsed values
        m = vid_meta.get(vid_id, {})
        lat = m.get("lat", parsed["lat"])
        lon = m.get("lon", parsed["lon"])
        altitude = m.get("altitude", 0.0)
        posted_speed = m.get("speedlimit", parsed["posted_speed"])
        difficulty = m.get("difficulty", parsed["difficulty"])

        # Weather from metadata
        temp_c = float(m.get("temp", 10.0))
        humidity = float(m.get("humidity", 70.0))
        precip = float(m.get("precipitation", 0.0))
        weather_factor, weather_reasons = _compute_weather_factor(temp_c, precip)

        meta: Dict[str, Any] = {
            "video_id": vid_id,
            "filename": filename,
            "index": i,
            "difficulty": difficulty,
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
            "posted_speed": posted_speed,
            # Weather
            "temp_c": temp_c,
            "humidity": humidity,
            "precipitation_mm_h": precip,
            "weather_factor": weather_factor,
            "weather_reasons": weather_reasons,
        }

        if vid_id in targets:
            t = targets[vid_id]
            meta["target_speed"] = t["target"]
            meta["mean_raw"] = t["mean_raw"]
            meta["n_responses"] = t["n_responses"]
        else:
            meta["target_speed"] = posted_speed
            meta["mean_raw"] = float(posted_speed)
            meta["n_responses"] = 0

        videos.append(meta)
    return videos


VIDEO_METADATA_JSON = _load_video_metadata()
SURVEY_TARGETS = _load_survey_targets()
VIDEOS = _build_video_list(SURVEY_TARGETS, VIDEO_METADATA_JSON)
VIDEO_BY_ID = {v["video_id"]: v for v in VIDEOS}


# ── Tuning state ────────────────────────────────────────────────────────────────

class TuningState:
    """Thread-safe container for all tunable parameters."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        # Blend weights (raw 0–100 values; normalized internally)
        self.w_camera: float = 60.0
        self.w_weather: float = 30.0
        self.w_confidence: float = 10.0
        # Per-difficulty-level speed reduction factors (0.05–1.0)
        self.difficulty_factors: Dict[int, float] = {1: 1.0, 2: 0.90, 3: 0.70, 4: 0.50, 5: 0.35}
        # Neutral camera factor (fallback when confidence is low)
        self.neutral_cam: float = 0.88
        # Weather condition reduction multipliers
        self.wf_light_precip: float = 0.85    # precip 0.5–5 mm/h
        self.wf_mod_precip: float = 0.70      # precip 5–10 mm/h
        self.wf_heavy_precip: float = 0.50    # precip > 10 mm/h
        self.wf_near_freeze: float = 0.80     # temp 0–3 °C
        self.wf_freeze: float = 0.60          # temp < 0 °C

    def compute_weather_factor(self, temp_c: float, precip_mm_h: float) -> Tuple[float, List[str]]:
        """Compute a [0,1] speed-reduction factor using the current tuning state."""
        factor = 1.0
        reasons: List[str] = []
        if precip_mm_h > 10:
            factor *= self.wf_heavy_precip; reasons.append("heavy rain/snow")
        elif precip_mm_h > 5:
            factor *= self.wf_mod_precip; reasons.append("moderate rain/snow")
        elif precip_mm_h > 0.5:
            factor *= self.wf_light_precip; reasons.append("light rain/snow")
        if temp_c < 0:
            factor *= self.wf_freeze; reasons.append("freezing")
        elif temp_c < 3:
            factor *= self.wf_near_freeze; reasons.append("near freezing")
        return round(factor, 4), reasons

    def to_dict(self) -> Dict[str, Any]:
        return {
            "w_camera": self.w_camera,
            "w_weather": self.w_weather,
            "w_confidence": self.w_confidence,
            "difficulty_factors": {str(k): v for k, v in self.difficulty_factors.items()},
            "neutral_cam": self.neutral_cam,
            "wf_light_precip": self.wf_light_precip,
            "wf_mod_precip": self.wf_mod_precip,
            "wf_heavy_precip": self.wf_heavy_precip,
            "wf_near_freeze": self.wf_near_freeze,
            "wf_freeze": self.wf_freeze,
        }

    def apply(self, payload: Dict[str, Any]) -> None:
        if "w_camera" in payload:
            self.w_camera = max(0.0, float(payload["w_camera"]))
        if "w_weather" in payload:
            self.w_weather = max(0.0, float(payload["w_weather"]))
        if "w_confidence" in payload:
            self.w_confidence = max(0.0, float(payload["w_confidence"]))
        if "difficulty_factors" in payload:
            df = payload["difficulty_factors"]
            for level in range(1, 6):
                key = str(level)
                if key in df:
                    self.difficulty_factors[level] = max(0.05, min(1.0, float(df[key])))
        if "neutral_cam" in payload:
            self.neutral_cam = max(0.1, min(1.0, float(payload["neutral_cam"])))
        if "wf_light_precip" in payload:
            self.wf_light_precip = max(0.1, min(1.0, float(payload["wf_light_precip"])))
        if "wf_mod_precip" in payload:
            self.wf_mod_precip = max(0.1, min(1.0, float(payload["wf_mod_precip"])))
        if "wf_heavy_precip" in payload:
            self.wf_heavy_precip = max(0.1, min(1.0, float(payload["wf_heavy_precip"])))
        if "wf_near_freeze" in payload:
            self.wf_near_freeze = max(0.1, min(1.0, float(payload["wf_near_freeze"])))
        if "wf_freeze" in payload:
            self.wf_freeze = max(0.1, min(1.0, float(payload["wf_freeze"])))


# ── ONNX video analyser ─────────────────────────────────────────────────────────

class VideoAnalyzer:
    """
    Background-processes each video: samples frames, runs ONNX inference,
    stores per-frame probability dicts keyed by timestamp (seconds).
    """

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = None
        self.model_input_name: Optional[str] = None
        self.model_status = "not_loaded"
        self.current_model_name: str = DEFAULT_MODEL_NAME
        self.use_crop: bool = False
        # {video_id: {"frames": [{t, scores}], "status": str, "duration": float}}
        self.data: Dict[str, Dict[str, Any]] = {
            v["video_id"]: {"frames": [], "status": "pending", "duration": 0.0}
            for v in VIDEOS
        }
        self._load_model(DEFAULT_MODEL_NAME)
        threading.Thread(target=self._precompute_all, daemon=True).start()

    def _load_model(self, model_name: str) -> None:
        if not (HAS_ONNX and HAS_NUMPY):
            self.model_status = "missing onnxruntime / numpy"
            return
        path = MODELS_DIR / model_name
        if not path.exists():
            self.model_status = f"model not found: {path}"
            return
        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            self.model = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.model_input_name = self.model.get_inputs()[0].name
            self.model_status = "ok"
            self.current_model_name = model_name
        except Exception as exc:
            self.model_status = f"load error: {exc}"

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        """Load a different ONNX model and re-run inference on all videos."""
        if model_name not in _available_models():
            return {"ok": False, "error": f"Unknown model: {model_name}"}
        with self.lock:
            for vid_id in self.data:
                self.data[vid_id] = {"frames": [], "status": "pending", "duration": 0.0}
            self._load_model(model_name)
        threading.Thread(target=self._precompute_all, daemon=True).start()
        return {"ok": True, "model": model_name, "status": self.model_status}

    def set_crop(self, use_crop: bool) -> Dict[str, Any]:
        """Toggle road-crop mode and re-run inference on all videos."""
        with self.lock:
            self.use_crop = use_crop
            for vid_id in self.data:
                self.data[vid_id] = {"frames": [], "status": "pending", "duration": 0.0}
        threading.Thread(target=self._precompute_all, daemon=True).start()
        return {"ok": True, "use_crop": use_crop}

    def _crop_frame(self, frame: Any) -> Any:
        """Crop to the road-focused rectangle (constants defined at module level)."""
        h, w = frame.shape[:2]
        y1 = int(h * CROP_TOP)
        y2 = int(h * CROP_BOTTOM)
        x1 = int(w * CROP_LEFT)
        x2 = int(w * CROP_RIGHT)
        return frame[y1:y2, x1:x2]

    def _preprocess(self, frame: Any) -> Any:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        arr = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    def _infer(self, frame: Any) -> Dict[str, float]:
        x = self._preprocess(frame)
        logits = self.model.run(None, {self.model_input_name: x})[0][0]
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        # Zero out excluded classes (mud / gravel) then renormalise
        for i, name in enumerate(CLASS_NAMES):
            if name in EXCLUDED_CLASSES:
                probs[i] = 0.0
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return {
            CLASS_NAMES[i]: float(p)
            for i, p in enumerate(probs)
            if CLASS_NAMES[i] not in EXCLUDED_CLASSES
        }

    def _analyze_video(self, video: Dict[str, Any]) -> None:
        vid_id = video["video_id"]
        path = VIDEOS_DIR / video["filename"]

        with self.lock:
            self.data[vid_id]["status"] = "processing"

        if not (HAS_CV2 and HAS_NUMPY):
            with self.lock:
                self.data[vid_id]["status"] = "no_opencv"
            return

        if not path.exists():
            with self.lock:
                self.data[vid_id]["status"] = "file_not_found"
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            with self.lock:
                self.data[vid_id]["status"] = "cannot_open"
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0.0

        # Sample up to 60 frames evenly across the video
        n_samples = min(60, max(10, int(duration * 2)))
        interval = duration / n_samples if n_samples > 0 else 1.0

        frames_data: List[Dict[str, Any]] = []
        for i in range(n_samples):
            t_sec = i * interval
            cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
            ok, frame = cap.read()
            if not ok:
                continue
            if self.use_crop:
                frame = self._crop_frame(frame)
            if self.model is not None and self.model_input_name is not None:
                try:
                    scores = self._infer(frame)
                except Exception:
                    scores = {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}
            else:
                scores = {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}

            frames_data.append({
                "t": round(t_sec, 2),
                "scores": {k: round(v, 4) for k, v in scores.items()},
            })

        cap.release()

        with self.lock:
            self.data[vid_id]["frames"] = frames_data
            self.data[vid_id]["status"] = "done"
            self.data[vid_id]["duration"] = round(duration, 2)

    def _precompute_all(self) -> None:
        for video in VIDEOS:
            try:
                self._analyze_video(video)
            except Exception as exc:
                vid_id = video["video_id"]
                with self.lock:
                    self.data[vid_id]["status"] = f"error: {exc}"

    def get_avg_scores(self, video_id: str) -> Optional[Dict[str, float]]:
        """Return mean class probabilities across all sampled frames."""
        with self.lock:
            frames = self.data.get(video_id, {}).get("frames", [])
        if not frames:
            return None
        avg: Dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
        for frame in frames:
            for cls, p in frame["scores"].items():
                avg[cls] = avg.get(cls, 0.0) + p
        n = len(frames)
        return {c: v / n for c, v in avg.items()}

    def get_scores_matrix(self, video_id: str) -> Optional[Any]:
        """
        Return frame scores as a (n_frames, n_classes) numpy float32 array,
        in CLASS_NAMES order. Used by the vectorised optimizer.
        """
        if not HAS_NUMPY:
            return None
        with self.lock:
            frames = self.data.get(video_id, {}).get("frames", [])
        if not frames:
            return None
        mat = np.array(
            [[frame["scores"].get(cls, 0.0) for cls in CLASS_NAMES] for frame in frames],
            dtype=np.float32,
        )
        return mat

    def status_summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "model_status": self.model_status,
                "current_model": self.current_model_name,
                "use_crop": self.use_crop,
                "videos": {
                    vid: {"status": d["status"], "n_frames": len(d.get("frames", []))}
                    for vid, d in self.data.items()
                },
            }


# ── Recommendation engine ───────────────────────────────────────────────────────

def compute_recommendation(
    avg_scores: Optional[Dict[str, float]],
    posted_speed: int,
    state: TuningState,
    weather_factor: float = 1.0,
) -> Dict[str, Any]:
    """Apply tuning state to averaged frame scores → recommended speed.

    weather_factor comes from the video's metadata (pre-computed from temp + precipitation).
    """
    df = state.difficulty_factors
    neutral = state.neutral_cam

    if avg_scores:
        sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]
        total_p = sum(p for _, p in top_3)
        if total_p > 0:
            raw_cam = sum(
                (p / total_p) * df.get(DIFFICULTY_LEVEL.get(cls, 3), 0.7)
                for cls, p in top_3
            )
        else:
            raw_cam = neutral
        cam_conf = float(top_3[0][1]) if top_3 else 0.0
        top_label = top_3[0][0] if top_3 else "unknown"
        top_3_list = [[cls, round(p, 4)] for cls, p in top_3]
    else:
        raw_cam = neutral
        cam_conf = 0.0
        top_label = "unknown"
        top_3_list = []

    # Confidence damping: pulls camera factor toward neutral when confidence is low
    w_total = max(state.w_camera + state.w_weather + state.w_confidence, 1e-6)
    w_conf_norm = state.w_confidence / w_total
    effective_cam = raw_cam + w_conf_norm * (1.0 - cam_conf) * (neutral - raw_cam)

    # Weighted blend of camera and (per-video) weather
    w_cam = state.w_camera
    w_wea = state.w_weather
    w_sum = max(w_cam + w_wea, 1e-6)
    combined = (w_cam * effective_cam + w_wea * weather_factor) / w_sum

    recommended = int(round((posted_speed * combined) / 10.0) * 10)
    recommended = max(20, min(recommended, posted_speed))

    return {
        "recommended": recommended,
        "top_label": top_label,
        "top_3": top_3_list,
        "raw_cam_factor": round(raw_cam, 4),
        "effective_cam_factor": round(effective_cam, 4),
        "weather_factor": round(weather_factor, 4),
        "combined_factor": round(combined, 4),
        "cam_confidence": round(cam_conf, 4),
    }


# ── Auto-tuning via differential evolution ─────────────────────────────────────

def _continuous_rec(
    avg_scores: Optional[Dict[str, float]],
    posted_speed: int,
    params: Any,
    weather_factor: float,
) -> float:
    """
    Compute the pre-rounding speed recommendation (continuous float).
    params = [w_cam, w_wea, w_conf, df1, df2, df3, df4, df5, neutral_cam]
    """
    w_cam, w_wea, w_conf = float(params[0]), float(params[1]), float(params[2])
    df = {1: params[3], 2: params[4], 3: params[5], 4: params[6], 5: params[7]}
    neutral = float(params[8])

    if avg_scores:
        # Expected value of difficulty factor over full class distribution —
        # more precise than top-3 normalisation for the optimizer.
        raw_cam = sum(
            float(p) * float(df.get(DIFFICULTY_LEVEL.get(cls, 3), 0.7))
            for cls, p in avg_scores.items()
        )
        sorted_sc = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        cam_conf = float(sorted_sc[0][1])
    else:
        raw_cam = float(neutral)
        cam_conf = 0.0

    w_total = max(w_cam + w_wea + w_conf, 1e-6)
    w_conf_norm = w_conf / w_total
    effective_cam = raw_cam + w_conf_norm * (1.0 - cam_conf) * (neutral - raw_cam)

    w_sum = max(w_cam + w_wea, 1e-6)
    combined = (w_cam * effective_cam + w_wea * weather_factor) / w_sum
    return posted_speed * combined


def run_auto_tune() -> Dict[str, Any]:
    """
    Optimize all tuning parameters via scipy differential_evolution.

    Parameter vector (14 elements):
      [0]  w_camera          blend weight
      [1]  w_weather         blend weight
      [2]  w_confidence      blend weight
      [3]  df[1]  dry asphalt
      [4]  df[2]  wet asphalt
      [5]  df[3]  water / melted snow / gravel
      [6]  df[4]  fresh snow / wet gravel / mud
      [7]  df[5]  ice / water mud
      [8]  neutral_cam
      [9]  wf_light_precip   speed factor for light rain/snow  (0.5–5 mm/h)
      [10] wf_mod_precip     speed factor for moderate rain/snow (5–10 mm/h)
      [11] wf_heavy_precip   speed factor for heavy rain/snow   (>10 mm/h)
      [12] wf_near_freeze    speed factor for near-freezing temps (0–3 °C)
      [13] wf_freeze         speed factor for freezing temps    (<0 °C)

    Objective: minimise asymmetric MSE against continuous survey means (mean_raw).
    Ordering constraints enforced via penalty terms:
      df[1] ≥ df[2] ≥ … ≥ df[5]
      wf_light ≥ wf_mod ≥ wf_heavy
      wf_near_freeze ≥ wf_freeze
    """
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return {"error": "scipy not installed — run: pip install scipy"}

    level_vec = np.array(
        [DIFFICULTY_LEVEL.get(cls, 3) for cls in CLASS_NAMES], dtype=np.float32
    )

    video_data = []
    for v in VIDEOS:
        mat = analyzer.get_scores_matrix(v["video_id"])
        conf_vec = mat.max(axis=1) if mat is not None else None
        video_data.append({
            "video_id":         v["video_id"],
            "scores_matrix":    mat,
            "conf_vec":         conf_vec,
            "posted_speed":     v["posted_speed"],
            "mean_raw":         float(v.get("mean_raw", v["posted_speed"])),
            "target_speed":     v["target_speed"],
            "temp_c":           float(v["temp_c"]),
            "precip_mm_h":      float(v["precipitation_mm_h"]),
            "difficulty":       v["difficulty"],
        })

    def _weather_factor_from_params(temp_c: float, precip: float, params: Any) -> float:
        """Compute weather factor from optimisable weather-condition params."""
        factor = 1.0
        if precip > 10:
            factor *= float(params[11])   # wf_heavy_precip
        elif precip > 5:
            factor *= float(params[10])   # wf_mod_precip
        elif precip > 0.5:
            factor *= float(params[9])    # wf_light_precip
        if temp_c < 0:
            factor *= float(params[13])   # wf_freeze
        elif temp_c < 3:
            factor *= float(params[12])   # wf_near_freeze
        return factor

    def _median_rec(vd: Dict[str, Any], params: Any) -> float:
        """
        Compute the median per-frame recommended speed (continuous, pre-rounding)
        for one video given a parameter vector.
        """
        wf = _weather_factor_from_params(vd["temp_c"], vd["precip_mm_h"], params)

        mat = vd["scores_matrix"]
        if mat is None or len(mat) == 0:
            return float(vd["posted_speed"]) * float(params[8])  # neutral_cam fallback

        w_cam   = float(params[0])
        w_wea   = float(params[1])
        w_conf  = float(params[2])
        neutral = float(params[8])
        df_vec  = np.array(
            [float(params[3 + (int(lv) - 1)]) for lv in level_vec], dtype=np.float32
        )

        raw_cam_vec = mat @ df_vec                              # (n_frames,)
        conf_vec    = vd["conf_vec"]
        w_total     = max(w_cam + w_wea + w_conf, 1e-6)
        w_cn        = w_conf / w_total
        eff_cam     = raw_cam_vec + w_cn * (1.0 - conf_vec) * (neutral - raw_cam_vec)
        w_sum       = max(w_cam + w_wea, 1e-6)
        combined    = (w_cam * eff_cam + w_wea * wf) / w_sum

        return float(np.median(float(vd["posted_speed"]) * combined))

    DIFF_WEIGHT = {"easy": (4.0, 1.0), "hard": (1.0, 4.0)}

    def loss(params: Any) -> float:
        # df ordering penalty: df[1] ≥ df[2] ≥ … ≥ df[5]  (indices 3–7)
        mono_df = sum(
            max(0.0, float(params[i + 1]) - float(params[i])) ** 2
            for i in range(3, 7)
        )
        # weather ordering penalties
        mono_wf = (
            max(0.0, float(params[10]) - float(params[9]))  ** 2 +   # mod > light
            max(0.0, float(params[11]) - float(params[10])) ** 2 +   # heavy > mod
            max(0.0, float(params[13]) - float(params[12])) ** 2     # freeze > near_freeze
        )

        total = 0.0
        for vd in video_data:
            rec_median = _median_rec(vd, params)
            err = rec_median - vd["mean_raw"]
            under_w, over_w = DIFF_WEIGHT.get(vd["difficulty"], (2.0, 2.0))
            total += (under_w if err < 0 else over_w) * err ** 2

        return total / max(len(video_data), 1) + 20.0 * mono_df + 20.0 * mono_wf

    bounds = [
        (5.0,  100.0),   # [0]  w_camera
        (0.1,  100.0),   # [1]  w_weather
        (0.1,   40.0),   # [2]  w_confidence
        (0.85,  1.00),   # [3]  df[1]  dry asphalt
        (0.75,  1.00),   # [4]  df[2]  wet asphalt
        (0.65,  1.00),   # [5]  df[3]  water / melted snow / gravel
        (0.45,  0.95),   # [6]  df[4]  fresh snow / wet gravel / mud
        (0.20,  0.70),   # [7]  df[5]  ice / water mud
        (0.50,  0.92),   # [8]  neutral_cam
        (0.60,  1.00),   # [9]  wf_light_precip
        (0.40,  0.95),   # [10] wf_mod_precip
        (0.20,  0.80),   # [11] wf_heavy_precip
        (0.60,  1.00),   # [12] wf_near_freeze
        (0.30,  0.85),   # [13] wf_freeze
    ]

    result = differential_evolution(
        loss,
        bounds,
        seed=42,
        maxiter=3000,
        tol=1e-10,
        popsize=25,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,
        workers=1,
    )

    p = result.x
    optimal_params = {
        "w_camera":         round(float(p[0]), 2),
        "w_weather":        round(float(p[1]), 2),
        "w_confidence":     round(float(p[2]), 2),
        "difficulty_factors": {
            "1": round(float(p[3]), 3),
            "2": round(float(p[4]), 3),
            "3": round(float(p[5]), 3),
            "4": round(float(p[6]), 3),
            "5": round(float(p[7]), 3),
        },
        "neutral_cam":      round(float(p[8]),  3),
        "wf_light_precip":  round(float(p[9]),  3),
        "wf_mod_precip":    round(float(p[10]), 3),
        "wf_heavy_precip":  round(float(p[11]), 3),
        "wf_near_freeze":   round(float(p[12]), 3),
        "wf_freeze":        round(float(p[13]), 3),
    }

    # Per-video comparison at optimal params (median of per-frame recs)
    per_video = []
    for vd in video_data:
        rec_cont = _median_rec(vd, p)
        rec_rounded = int(round(rec_cont / 10.0) * 10)
        rec_rounded = max(20, min(rec_rounded, vd["posted_speed"]))
        delta = rec_rounded - vd["target_speed"]
        per_video.append({
            "video_id":    vd["video_id"],
            "difficulty":  vd["difficulty"],
            "target":      vd["target_speed"],
            "mean_raw":    round(vd["mean_raw"], 1),
            "recommended": rec_rounded,
            "delta":       delta,
            "match":       delta == 0,
        })

    n_match_exact = sum(1 for v in per_video if v["match"])
    n_match_near  = sum(1 for v in per_video if abs(v["delta"]) <= 10)

    return {
        "params":         optimal_params,
        "converged":      bool(result.success),
        "final_loss":     round(float(result.fun), 4),
        "n_iterations":   int(result.nit),
        "n_match":        n_match_exact,
        "n_match_near":   n_match_near,
        "total":          len(per_video),
        "per_video":      per_video,
    }


def _median_recommended(video: Dict[str, Any], state: TuningState) -> int:
    """
    Compute the recommended speed as the median of per-frame continuous
    recommendations, rounded to the nearest 10 km/h.
    Falls back to the avg-scores recommendation when no frame data is available.
    """
    wf, _ = state.compute_weather_factor(video["temp_c"], video["precipitation_mm_h"])
    mat = analyzer.get_scores_matrix(video["video_id"])
    if mat is None or len(mat) == 0:
        rec = compute_recommendation(
            analyzer.get_avg_scores(video["video_id"]),
            video["posted_speed"],
            state,
            weather_factor=wf,
        )
        return int(rec["recommended"])

    w_cam  = state.w_camera
    w_wea  = state.w_weather
    w_conf = state.w_confidence
    neutral = state.neutral_cam
    posted  = float(video["posted_speed"])
    wf      = float(wf)

    lv = np.array(
        [DIFFICULTY_LEVEL.get(cls, 3) for cls in CLASS_NAMES], dtype=np.float32
    )
    df_vec = np.array(
        [state.difficulty_factors.get(int(l), 0.8) for l in lv],
        dtype=np.float32,
    )

    raw_cam  = mat @ df_vec                                    # (n_frames,)
    conf     = mat.max(axis=1)                                 # (n_frames,)
    w_total  = max(w_cam + w_wea + w_conf, 1e-6)
    w_cn     = w_conf / w_total
    eff_cam  = raw_cam + w_cn * (1.0 - conf) * (neutral - raw_cam)
    w_sum    = max(w_cam + w_wea, 1e-6)
    combined = (w_cam * eff_cam + w_wea * wf) / w_sum
    recs     = posted * combined
    median_r = float(np.median(recs))

    rounded = int(round(median_r / 10.0) * 10)
    return max(20, min(rounded, int(posted)))


def all_recommendations(state: TuningState) -> List[Dict[str, Any]]:
    results = []
    for video in VIDEOS:
        # Re-compute weather factor using current tuning state multipliers
        wf, wf_reasons = state.compute_weather_factor(video["temp_c"], video["precipitation_mm_h"])
        avg = analyzer.get_avg_scores(video["video_id"])
        rec = compute_recommendation(avg, video["posted_speed"], state, weather_factor=wf)
        recommended_speed = _median_recommended(video, state)
        target = video["target_speed"]
        delta = recommended_speed - target
        results.append({
            "video_id": video["video_id"],
            "filename": video["filename"],
            "difficulty": video["difficulty"],
            "posted_speed": video["posted_speed"],
            "target_speed": target,
            "mean_raw": video.get("mean_raw", float(target)),
            "n_responses": video.get("n_responses", 0),
            "recommended_speed": recommended_speed,
            "delta": delta,
            "match": delta == 0,
            "details": rec,
            "weather": {
                "temp_c": video["temp_c"],
                "humidity": video["humidity"],
                "precipitation_mm_h": video["precipitation_mm_h"],
                "weather_factor": wf,
                "reasons": wf_reasons,
            },
        })
    return results


# ── App init ────────────────────────────────────────────────────────────────────

tuning_state = TuningState()
analyzer = VideoAnalyzer()


# ── Routes ──────────────────────────────────────────────────────────────────────

@app.route("/")
def index() -> Any:
    return render_template("index.html")


@app.route("/videos/<path:filename>")
def serve_video(filename: str) -> Any:
    return send_from_directory(str(VIDEOS_DIR), filename, conditional=True)


@app.route("/api/init", methods=["GET"])
def api_init() -> Any:
    """Return everything the UI needs on first load."""
    with tuning_state.lock:
        state = tuning_state.to_dict()
        recs = all_recommendations(tuning_state)
    n_match = sum(1 for r in recs if r["match"])
    return jsonify({
        "videos": VIDEOS,
        "state": state,
        "recommendations": recs,
        "n_match": n_match,
        "total": len(recs),
        "analyzer": analyzer.status_summary(),
        "difficulty_names": DIFFICULTY_NAMES,
        "class_colors": CLASS_COLORS,
    })


@app.route("/api/recommendations", methods=["GET"])
def api_recommendations() -> Any:
    with tuning_state.lock:
        recs = all_recommendations(tuning_state)
        state = tuning_state.to_dict()
    n_match = sum(1 for r in recs if r["match"])
    return jsonify({"recommendations": recs, "n_match": n_match, "total": len(recs), "state": state})


@app.route("/api/tune", methods=["POST"])
def api_tune() -> Any:
    """Update tuning parameters and return updated recommendations immediately."""
    payload = request.get_json(silent=True) or {}
    with tuning_state.lock:
        tuning_state.apply(payload)
        state = tuning_state.to_dict()
        recs = all_recommendations(tuning_state)
    n_match = sum(1 for r in recs if r["match"])
    return jsonify({
        "state": state,
        "recommendations": recs,
        "n_match": n_match,
        "total": len(recs),
    })


@app.route("/api/analyzer-status", methods=["GET"])
def api_analyzer_status() -> Any:
    return jsonify(analyzer.status_summary())


@app.route("/api/models", methods=["GET"])
def api_list_models() -> Any:
    """Return available ONNX models and the currently active one."""
    return jsonify({
        "models": _available_models(),
        "current": analyzer.current_model_name,
    })


@app.route("/api/model", methods=["POST"])
def api_switch_model() -> Any:
    """Switch to a different ONNX model and re-run inference on all videos."""
    payload = request.get_json(force=True, silent=True) or {}
    model_name = payload.get("model", "").strip()
    if not model_name:
        return jsonify({"ok": False, "error": "model name required"}), 400
    result = analyzer.switch_model(model_name)
    if not result["ok"]:
        return jsonify(result), 400
    return jsonify(result)


@app.route("/api/crop", methods=["POST"])
def api_set_crop() -> Any:
    """Toggle road-crop preprocessing and re-run inference on all videos."""
    payload = request.get_json(force=True, silent=True) or {}
    use_crop = bool(payload.get("use_crop", False))
    result = analyzer.set_crop(use_crop)
    return jsonify(result)


@app.route("/api/video/<video_id>/predictions", methods=["GET"])
def api_video_predictions(video_id: str) -> Any:
    """Return all pre-computed frame predictions for a single video."""
    with analyzer.lock:
        data = dict(analyzer.data.get(video_id, {}))
    if not data:
        return jsonify({"error": "video not found"}), 404
    video_meta = VIDEO_BY_ID.get(video_id)
    if not video_meta:
        return jsonify({"error": "video metadata not found"}), 404
    with tuning_state.lock:
        wf, wf_reasons = tuning_state.compute_weather_factor(
            video_meta["temp_c"], video_meta["precipitation_mm_h"]
        )
        avg = analyzer.get_avg_scores(video_id)
        rec = compute_recommendation(avg, video_meta["posted_speed"], tuning_state, weather_factor=wf)
    return jsonify({
        "video_id": video_id,
        "frames": data.get("frames", []),
        "status": data.get("status", "unknown"),
        "duration": data.get("duration", 0.0),
        "avg_recommendation": rec,
        "class_names": CLASS_NAMES,
        "difficulty_level": DIFFICULTY_LEVEL,
        "class_colors": CLASS_COLORS,
        "weather": {
            "temp_c": video_meta["temp_c"],
            "humidity": video_meta["humidity"],
            "precipitation_mm_h": video_meta["precipitation_mm_h"],
            "weather_factor": wf,
            "reasons": wf_reasons,
        },
    })


@app.route("/api/video/<video_id>/frame-recommendation", methods=["POST"])
def api_frame_recommendation(video_id: str) -> Any:
    """Compute recommendation for a specific frame's scores (used by live view)."""
    payload = request.get_json(silent=True) or {}
    scores: Dict[str, float] = payload.get("scores", {})
    video_meta = VIDEO_BY_ID.get(video_id)
    if not video_meta:
        return jsonify({"error": "video not found"}), 404
    with tuning_state.lock:
        wf, _ = tuning_state.compute_weather_factor(
            video_meta["temp_c"], video_meta["precipitation_mm_h"]
        )
        rec = compute_recommendation(
            scores if scores else None,
            video_meta["posted_speed"],
            tuning_state,
            weather_factor=wf,
        )
    return jsonify(rec)


@app.route("/api/auto-tune", methods=["POST"])
def api_auto_tune() -> Any:
    """
    Run differential_evolution to find optimal parameters, apply them,
    and return updated recommendations.
    """
    # Don't start while videos are still being processed
    status = analyzer.status_summary()
    pending = [
        vid for vid, d in status["videos"].items()
        if d["status"] not in ("done", "error", "file_not_found", "no_opencv")
    ]
    if pending:
        return jsonify({"error": f"Videos still processing: {pending}. Please wait."}), 400

    try:
        result = run_auto_tune()
        if "error" in result:
            return jsonify(result), 500
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500

    # Snapshot current state for before/after comparison
    with tuning_state.lock:
        recs_before = all_recommendations(tuning_state)
        n_match_before = sum(1 for r in recs_before if r["match"])

    # Apply optimal params
    with tuning_state.lock:
        tuning_state.apply(result["params"])
        state = tuning_state.to_dict()
        recs_after = all_recommendations(tuning_state)

    n_match_after = sum(1 for r in recs_after if r["match"])

    n_near_before = sum(1 for r in recs_before if abs(r["delta"]) <= 10)
    n_near_after  = sum(1 for r in recs_after  if abs(r["delta"]) <= 10)

    return jsonify({
        "params":           result["params"],
        "converged":        result["converged"],
        "final_loss":       result["final_loss"],
        "n_iterations":     result["n_iterations"],
        "per_video":        result["per_video"],
        "n_match_before":   n_match_before,
        "n_match_after":    n_match_after,
        "n_near_before":    n_near_before,
        "n_near_after":     n_near_after,
        "total":            len(recs_after),
        "state":            state,
        "recommendations":  recs_after,
    })


if __name__ == "__main__":
    host = os.environ.get("HOST", "127.0.0.1")
    port = int(os.environ.get("PORT", "8000"))
    app.run(host=host, port=port, debug=False)
