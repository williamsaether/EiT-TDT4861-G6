from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.common.recommendation import apply_heuristics
from app.tuning.config import CLASS_NAMES, DIFFICULTY_LEVEL
from app.tuning.state import TuningState


def _normalize_scores(scores: Optional[Dict[str, float]]) -> Dict[str, float]:
    if not scores:
        return {}
    out: Dict[str, float] = {}
    for key, value in scores.items():
        try:
            v = float(value)
        except (TypeError, ValueError):
            continue
        out[str(key)] = max(0.0, v)
    total = sum(out.values())
    if total <= 0:
        return {}
    return {k: v / total for k, v in out.items()}


def _frame_scores_from_matrix_row(row: Any) -> Dict[str, float]:
    return {cls: float(row[idx]) for idx, cls in enumerate(CLASS_NAMES)}


def compute_recommendation(
    avg_scores: Optional[Dict[str, float]],
    posted_speed: int,
    state: TuningState,
    weather_factor: float = 1.0,
    temp_c: float | None = None,
    humidity: float | None = None,
    precip_mm_h: float | None = None,
) -> Dict[str, Any]:
    df = state.difficulty_factors
    neutral = state.neutral_cam

    working_scores = _normalize_scores(avg_scores)
    if working_scores and temp_c is not None and precip_mm_h is not None:
        working_scores = apply_heuristics(
            working_scores,
            temp_c=temp_c,
            humidity=humidity,
            precip_mm_h=precip_mm_h,
            speed_limit=float(posted_speed),
        )

    if working_scores:
        sorted_scores = sorted(working_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]
        cam_conf = float(top_3[0][1]) if top_3 else 0.0
        top_label = top_3[0][0] if top_3 else "unknown"
        raw_cam = float(df.get(DIFFICULTY_LEVEL.get(top_label, 3), neutral))
        top_3_list = [[cls, round(p, 4)] for cls, p in top_3]
    else:
        raw_cam = neutral
        cam_conf = 0.0
        top_label = "unknown"
        top_3_list = []

    w_total = max(state.w_camera + state.w_weather + state.w_confidence, 1e-6)
    w_conf_norm = state.w_confidence / w_total
    effective_cam = raw_cam + w_conf_norm * (1.0 - cam_conf) * (neutral - raw_cam)

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


def median_recommended(video: Dict[str, Any], state: TuningState, analyzer: Any) -> int:
    wf, _ = state.compute_weather_factor(video["temp_c"], video["precipitation_mm_h"])
    mat = analyzer.get_scores_matrix(video["video_id"])
    if mat is None or len(mat) == 0:
        rec = compute_recommendation(
            analyzer.get_avg_scores(video["video_id"]),
            video["posted_speed"],
            state,
            weather_factor=wf,
            temp_c=float(video["temp_c"]),
            humidity=float(video["humidity"]) if video.get("humidity") is not None else None,
            precip_mm_h=float(video["precipitation_mm_h"]),
        )
        return int(rec["recommended"])

    w_cam = state.w_camera
    w_wea = state.w_weather
    w_conf = state.w_confidence
    neutral = state.neutral_cam
    posted = float(video["posted_speed"])
    wf = float(wf)

    temp_c = float(video["temp_c"])
    humidity = float(video["humidity"]) if video.get("humidity") is not None else None
    precip_mm_h = float(video["precipitation_mm_h"])
    speed_limit = float(video["posted_speed"])

    raw_cam_values: List[float] = []
    conf_values: List[float] = []
    for row in mat:
        frame_scores = _normalize_scores(_frame_scores_from_matrix_row(row))
        if frame_scores:
            frame_scores = apply_heuristics(
                frame_scores,
                temp_c=temp_c,
                humidity=humidity,
                precip_mm_h=precip_mm_h,
                speed_limit=speed_limit,
            )
        if frame_scores:
            top_label, top_conf = max(frame_scores.items(), key=lambda item: item[1])
            cam_factor = float(state.difficulty_factors.get(DIFFICULTY_LEVEL.get(top_label, 3), neutral))
            raw_cam_values.append(cam_factor)
            conf_values.append(float(top_conf))
        else:
            raw_cam_values.append(float(neutral))
            conf_values.append(0.0)

    raw_cam = np.array(raw_cam_values, dtype=np.float32)
    conf = np.array(conf_values, dtype=np.float32)

    # Temporal smoothing over frame-level class scores (backend setting).
    duration_s = max(float(analyzer.get_duration(video["video_id"])), 0.0)
    if duration_s > 0 and state.smooth_window_sec > 0:
        fps_eff = len(mat) / duration_s
        window_frames = max(1, int(round(state.smooth_window_sec * fps_eff)))
        if window_frames > 1:
            kernel = np.ones(window_frames, dtype=np.float32) / float(window_frames)
            raw_cam = np.convolve(raw_cam, kernel, mode="same")
            conf = np.convolve(conf, kernel, mode="same")

    w_total = max(w_cam + w_wea + w_conf, 1e-6)
    w_cn = w_conf / w_total
    eff_cam = raw_cam + w_cn * (1.0 - conf) * (neutral - raw_cam)
    w_sum = max(w_cam + w_wea, 1e-6)
    combined = (w_cam * eff_cam + w_wea * wf) / w_sum
    recs = posted * combined
    median_r = float(np.median(recs))

    rounded = int(round(median_r / 10.0) * 10)
    return max(20, min(rounded, int(posted)))


def all_recommendations(state: TuningState, analyzer: Any, videos: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    results = []
    for video in videos:
        wf, wf_reasons = state.compute_weather_factor(video["temp_c"], video["precipitation_mm_h"])
        avg = analyzer.get_avg_scores(video["video_id"])
        rec = compute_recommendation(
            avg,
            video["posted_speed"],
            state,
            weather_factor=wf,
            temp_c=float(video["temp_c"]),
            humidity=float(video["humidity"]) if video.get("humidity") is not None else None,
            precip_mm_h=float(video["precipitation_mm_h"]),
        )
        recommended_speed = median_recommended(video, state, analyzer)
        target = video["target_speed"]
        delta = recommended_speed - target
        results.append(
            {
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
            }
        )
    return results
