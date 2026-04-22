from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]
METADATA_PATH = ROOT / "videos" / "metadata.json"
VIDEOS_DIR = ROOT / "videos"


class ExampleVideoService:
    """Provides allow-listed example videos for the driving demo."""

    def __init__(self) -> None:
        self.examples = self._load_examples()
        self.allowed_names = {v["video"] for v in self.examples}

    def _load_examples(self) -> List[Dict[str, Any]]:
        if not METADATA_PATH.exists():
            return []

        import json

        with open(METADATA_PATH, encoding="utf-8") as f:
            entries = json.load(f)

        examples: List[Dict[str, Any]] = []
        for entry in entries:
            include_as_example = bool(entry.get("forExample", False)) or not bool(entry.get("forTraining", True))
            if not include_as_example:
                continue
            filename = str(entry.get("video", "")).strip()
            if not filename:
                continue
            if not (VIDEOS_DIR / filename).exists():
                continue

            examples.append(
                {
                    "video": filename,
                    "difficulty": entry.get("difficulty", "unknown"),
                    "lat": float(entry.get("lat", 0.0)),
                    "lon": float(entry.get("lon", 0.0)),
                    "altitude": float(entry.get("altitude", 0.0)),
                    "temp": float(entry.get("temp", 1.0)),
                    "humidity": float(entry.get("humidity", 80.0)),
                    "precipitation": float(entry.get("precipitation", 0.0)),
                    "forTraining": bool(entry.get("forTraining", True)),
                }
            )

        return examples

    def list_examples(self) -> List[Dict[str, Any]]:
        return self.examples

    def is_allowed(self, filename: str) -> bool:
        return filename in self.allowed_names
