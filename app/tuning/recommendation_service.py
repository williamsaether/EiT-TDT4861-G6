from __future__ import annotations

from typing import Any, Dict, List, Optional

import numpy as np

from app.tuning.config import CLASS_NAMES, DIFFICULTY_LEVEL
from app.tuning.state import TuningState


def compute_recommendation(
    avg_scores: Optional[Dict[str, float]],
    posted_speed: int,
    state: TuningState,
    weather_factor: float = 1.0,
) -> Dict[str, Any]:
    df = state.difficulty_factors
    neutral = state.neutral_cam

    if avg_scores:
        sorted_scores = sorted(avg_scores.items(), key=lambda x: x[1], reverse=True)
        top_3 = sorted_scores[:3]
        total_p = sum(p for _, p in top_3)
        if total_p > 0:
            raw_cam = sum((p / total_p) * df.get(DIFFICULTY_LEVEL.get(cls, 3), 0.7) for cls, p in top_3)
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
        )
        return int(rec["recommended"])

    w_cam = state.w_camera
    w_wea = state.w_weather
    w_conf = state.w_confidence
    neutral = state.neutral_cam
    posted = float(video["posted_speed"])
    wf = float(wf)

    lv = np.array([DIFFICULTY_LEVEL.get(cls, 3) for cls in CLASS_NAMES], dtype=np.float32)
    df_vec = np.array([state.difficulty_factors.get(int(l), 0.8) for l in lv], dtype=np.float32)

    raw_cam = mat @ df_vec
    conf = mat.max(axis=1)

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
        rec = compute_recommendation(avg, video["posted_speed"], state, weather_factor=wf)
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
