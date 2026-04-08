from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Dict, List, Tuple

from app.common.crop import CROP_BOTTOM, CROP_LEFT, CROP_RIGHT, CROP_TOP

ROOT = Path(__file__).resolve().parent.parent.parent
VIDEOS_DIR = ROOT / "videos"
CSV_PATH = ROOT / "merged_survey.csv"
MODELS_DIR = ROOT / "models" / "onnx"
DEFAULT_MODEL_NAME = "rscd_resnet18_v2.onnx"
METADATA_PATH = VIDEOS_DIR / "metadata.json"

IMAGE_SIZE = 224

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

EXCLUDED_CLASSES: frozenset = frozenset([
    "dry_gravel",
    "dry_mud",
    "water_gravel",
    "water_mud",
    "wet_gravel",
    "wet_mud",
])

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
    "dry_asphalt": "#4ade80",
    "wet_asphalt": "#60a5fa",
    "fresh_snow": "#e0f2fe",
    "melted_snow": "#bae6fd",
    "ice": "#93c5fd",
    "water_asphalt": "#2563eb",
}


def available_models() -> List[str]:
    if not MODELS_DIR.exists():
        return [DEFAULT_MODEL_NAME]
    return sorted(p.name for p in MODELS_DIR.glob("*.onnx"))


def load_video_metadata() -> Dict[str, Dict[str, Any]]:
    if not METADATA_PATH.exists():
        return {}

    import json

    with open(METADATA_PATH, encoding="utf-8") as f:
        entries = json.load(f)

    result: Dict[str, Dict[str, Any]] = {}
    for entry in entries:
        raw = entry.get("video", "")
        vid_id = Path(raw).stem
        result[vid_id] = entry
    return result


def compute_weather_factor(temp_c: float, precip_mm_h: float) -> Tuple[float, List[str]]:
    factor = 1.0
    reasons: List[str] = []
    if precip_mm_h > 10:
        factor *= 0.50
        reasons.append("heavy rain/snow")
    elif precip_mm_h > 5:
        factor *= 0.70
        reasons.append("moderate rain/snow")
    elif precip_mm_h > 0.5:
        factor *= 0.85
        reasons.append("light rain/snow")
    if temp_c < 0:
        factor *= 0.60
        reasons.append("freezing")
    elif temp_c < 3:
        factor *= 0.80
        reasons.append("near freezing")
    return round(factor, 4), reasons


def load_survey_targets() -> Dict[str, Dict[str, Any]]:
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


def parse_video_filename(filename: str) -> Dict[str, Any]:
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


def build_video_list(
    targets: Dict[str, Dict[str, Any]],
    vid_meta: Dict[str, Dict[str, Any]],
) -> List[Dict[str, Any]]:
    if not VIDEOS_DIR.exists():
        return []
    files_by_id: Dict[str, str] = {}
    for p in VIDEOS_DIR.glob("*.mp4"):
        files_by_id[Path(p.name).stem] = p.name

    # Restrict tuning set to metadata entries flagged for training.
    training_ids = sorted(
        vid_id
        for vid_id, meta in vid_meta.items()
        if bool(meta.get("forTraining", True))
    )

    videos = []
    for i, vid_id in enumerate(training_ids):
        filename = files_by_id.get(vid_id)
        if not filename:
            continue
        parsed = parse_video_filename(filename)

        m = vid_meta.get(vid_id, {})
        lat = m.get("lat", parsed["lat"])
        lon = m.get("lon", parsed["lon"])
        altitude = m.get("altitude", 0.0)
        posted_speed = m.get("speedlimit", parsed["posted_speed"])
        difficulty = m.get("difficulty", parsed["difficulty"])

        temp_c = float(m.get("temp", 10.0))
        humidity = float(m.get("humidity", 70.0))
        precip = float(m.get("precipitation", 0.0))
        weather_factor, weather_reasons = compute_weather_factor(temp_c, precip)

        meta: Dict[str, Any] = {
            "video_id": vid_id,
            "filename": filename,
            "index": i,
            "difficulty": difficulty,
            "lat": lat,
            "lon": lon,
            "altitude": altitude,
            "posted_speed": posted_speed,
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


VIDEO_METADATA_JSON = load_video_metadata()
SURVEY_TARGETS = load_survey_targets()
VIDEOS = build_video_list(SURVEY_TARGETS, VIDEO_METADATA_JSON)
VIDEO_BY_ID = {v["video_id"]: v for v in VIDEOS}
