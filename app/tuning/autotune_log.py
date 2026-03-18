from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict


ROOT = Path(__file__).resolve().parents[2]
AUTO_TUNE_LOG_DIR = ROOT / "logs" / "autotune"


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_run_id() -> str:
    # Compact UTC timestamp id used in filenames and API payloads.
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")


def write_autotune_log(run_id: str, payload: Dict[str, Any]) -> str:
    AUTO_TUNE_LOG_DIR.mkdir(parents=True, exist_ok=True)
    out_path = AUTO_TUNE_LOG_DIR / f"{run_id}.json"
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return str(out_path)
