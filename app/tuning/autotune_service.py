from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional

import numpy as np

from app.tuning.config import CLASS_NAMES, DIFFICULTY_LEVEL


def continuous_rec(
    avg_scores: Optional[Dict[str, float]],
    posted_speed: int,
    params: Any,
    weather_factor: float,
) -> float:
    w_cam, w_wea, w_conf = float(params[0]), float(params[1]), float(params[2])
    df = {1: params[3], 2: params[4], 3: params[5], 4: params[6], 5: params[7]}
    neutral = float(params[8])

    if avg_scores:
        raw_cam = sum(float(p) * float(df.get(DIFFICULTY_LEVEL.get(cls, 3), 0.7)) for cls, p in avg_scores.items())
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


def run_auto_tune(
    analyzer: Any,
    videos: List[Dict[str, Any]],
    smooth_window_sec: float = 3.0,
    progress_cb: Optional[Callable[[Dict[str, Any]], None]] = None,
    should_stop_cb: Optional[Callable[[], bool]] = None,
    maxiter: int = 3000,
    popsize: int = 25,
) -> Dict[str, Any]:
    try:
        from scipy.optimize import differential_evolution
    except ImportError:
        return {"error": "scipy not installed — run: pip install scipy"}

    level_vec = np.array([DIFFICULTY_LEVEL.get(cls, 3) for cls in CLASS_NAMES], dtype=np.float32)

    video_data = []
    for v in videos:
        mat = analyzer.get_scores_matrix(v["video_id"])
        conf_vec = mat.max(axis=1) if mat is not None else None
        video_data.append(
            {
                "video_id": v["video_id"],
                "scores_matrix": mat,
                "conf_vec": conf_vec,
                "duration_s": float(analyzer.get_duration(v["video_id"])),
                "posted_speed": v["posted_speed"],
                "mean_raw": float(v.get("mean_raw", v["posted_speed"])),
                "target_speed": v["target_speed"],
                "temp_c": float(v["temp_c"]),
                "precip_mm_h": float(v["precipitation_mm_h"]),
                "difficulty": v["difficulty"],
            }
        )

    def _weather_factor_from_params(temp_c: float, precip: float, params: Any) -> float:
        factor = 1.0
        if precip > 10:
            factor *= float(params[11])
        elif precip > 5:
            factor *= float(params[10])
        elif precip > 0.5:
            factor *= float(params[9])
        if temp_c < 0:
            factor *= float(params[13])
        elif temp_c < 3:
            factor *= float(params[12])
        return factor

    def _smooth_window_from_params(params: Any) -> float:
        # Backward-compatible fallback if smooth window is not part of the vector.
        if len(params) > 14:
            return max(0.1, float(params[14]))
        return max(0.1, float(smooth_window_sec))

    def _median_rec(vd: Dict[str, Any], params: Any) -> float:
        wf = _weather_factor_from_params(vd["temp_c"], vd["precip_mm_h"], params)
        mat = vd["scores_matrix"]
        if mat is None or len(mat) == 0:
            return float(vd["posted_speed"]) * float(params[8])

        w_cam = float(params[0])
        w_wea = float(params[1])
        w_conf = float(params[2])
        neutral = float(params[8])
        df_vec = np.array([float(params[3 + (int(lv) - 1)]) for lv in level_vec], dtype=np.float32)

        raw_cam_vec = mat @ df_vec
        conf_vec = vd["conf_vec"]

        duration_s = max(float(vd.get("duration_s", 0.0)), 0.0)
        active_smooth_window = _smooth_window_from_params(params)
        if duration_s > 0 and active_smooth_window > 0:
            fps_eff = len(mat) / duration_s
            window_frames = max(1, int(round(active_smooth_window * fps_eff)))
            if window_frames > 1:
                kernel = np.ones(window_frames, dtype=np.float32) / float(window_frames)
                raw_cam_vec = np.convolve(raw_cam_vec, kernel, mode="same")
                conf_vec = np.convolve(conf_vec, kernel, mode="same")

        w_total = max(w_cam + w_wea + w_conf, 1e-6)
        w_cn = w_conf / w_total
        eff_cam = raw_cam_vec + w_cn * (1.0 - conf_vec) * (neutral - raw_cam_vec)
        w_sum = max(w_cam + w_wea, 1e-6)
        combined = (w_cam * eff_cam + w_wea * wf) / w_sum

        return float(np.median(float(vd["posted_speed"]) * combined))

    DIFF_WEIGHT = {"easy": (4.0, 1.0), "hard": (1.0, 4.0)}

    def loss(params: Any) -> float:
        mono_df = sum(max(0.0, float(params[i + 1]) - float(params[i])) ** 2 for i in range(3, 7))
        mono_wf = (
            max(0.0, float(params[10]) - float(params[9])) ** 2
            + max(0.0, float(params[11]) - float(params[10])) ** 2
            + max(0.0, float(params[13]) - float(params[12])) ** 2
        )

        total_cont = 0.0
        total_discrete = 0.0
        for vd in video_data:
            rec_median = _median_rec(vd, params)
            rec_cont = max(20.0, min(rec_median, float(vd["posted_speed"])))
            rec_rounded = int(round(rec_cont / 10.0) * 10)
            rec_rounded = max(20, min(rec_rounded, int(vd["posted_speed"])))

            # Keep some pressure toward survey mean to preserve general fit.
            err = rec_cont - vd["mean_raw"]
            under_w, over_w = DIFF_WEIGHT.get(vd["difficulty"], (2.0, 2.0))
            mean_term = (under_w if err < 0 else over_w) * err**2

            # Directly optimize toward target-speed behavior.
            target_err = rec_cont - float(vd["target_speed"])
            target_term = target_err**2

            # Strong discrete penalty to maximize exact rounded matches.
            abs_delta = abs(rec_rounded - int(vd["target_speed"]))
            if abs_delta == 0:
                discrete = 0.0
            elif abs_delta == 10:
                discrete = 12.0
            elif abs_delta == 20:
                discrete = 30.0
            else:
                discrete = 55.0 + 2.0 * max(0.0, abs_delta - 20.0)

            total_cont += 0.35 * mean_term + 0.65 * target_term
            total_discrete += discrete

        n = max(len(video_data), 1)
        return (total_cont / n) + (total_discrete / n) + 20.0 * mono_df + 20.0 * mono_wf

    def _exact_near(params: Any) -> Dict[str, int]:
        exact = 0
        near = 0
        for vd in video_data:
            rec_cont = _median_rec(vd, params)
            rec_rounded = int(round(rec_cont / 10.0) * 10)
            rec_rounded = max(20, min(rec_rounded, vd["posted_speed"]))
            delta = rec_rounded - vd["target_speed"]
            if delta == 0:
                exact += 1
            if abs(delta) <= 10:
                near += 1
        return {"exact": exact, "near": near}

    bounds = [
        (5.0, 100.0),
        (0.1, 100.0),
        (0.1, 40.0),
        (0.85, 1.00),
        (0.75, 1.00),
        (0.65, 1.00),
        (0.45, 0.95),
        (0.20, 0.70),
        (0.50, 0.92),
        (0.60, 1.00),
        (0.40, 0.95),
        (0.20, 0.80),
        (0.60, 1.00),
        (0.30, 0.85),
        (0.5, 10.0),
    ]

    n_generations = {"count": 0}
    best = {"loss": float("inf"), "stagnation": 0, "reason": ""}
    best_candidate: Dict[str, Any] = {
        "params": None,
        "loss": float("inf"),
        "exact": -1,
        "near": -1,
    }

    def _is_better_candidate(candidate: Dict[str, Any], current_best: Dict[str, Any]) -> bool:
        if int(candidate["exact"]) != int(current_best["exact"]):
            return int(candidate["exact"]) > int(current_best["exact"])
        if int(candidate["near"]) != int(current_best["near"]):
            return int(candidate["near"]) > int(current_best["near"])
        return float(candidate["loss"]) < float(current_best["loss"]) - 1e-9

    def _callback(xk: Any, convergence: float) -> bool:
        n_generations["count"] += 1
        current_loss = float(loss(xk))
        metrics = _exact_near(xk)

        candidate = {
            "params": np.array(xk, dtype=np.float64).copy(),
            "loss": current_loss,
            "exact": int(metrics["exact"]),
            "near": int(metrics["near"]),
        }
        if _is_better_candidate(candidate, best_candidate):
            best_candidate.update(candidate)

        if current_loss + 1e-4 < best["loss"]:
            best["loss"] = current_loss
            best["stagnation"] = 0
        else:
            best["stagnation"] += 1

        if progress_cb:
            progress_cb(
                {
                    "iteration": n_generations["count"],
                    "maxiter": maxiter,
                    "progress": min(1.0, n_generations["count"] / max(maxiter, 1)),
                    "best_loss": round(float(best_candidate["loss"]), 4),
                    "best_exact": int(best_candidate["exact"]),
                    "best_near": int(best_candidate["near"]),
                    "convergence": float(convergence),
                }
            )

        if should_stop_cb and should_stop_cb():
            best["reason"] = "stopped_by_user"
            return True

        # Early stop if optimum on exact matches is reached.
        if metrics["exact"] >= len(video_data):
            best["reason"] = "all_exact_matched"
            return True

        # Early stop if no meaningful improvement for many generations.
        if best["stagnation"] >= 250:
            best["reason"] = "stagnation"
            return True

        return False

    result = differential_evolution(
        loss,
        bounds,
        seed=42,
        maxiter=maxiter,
        tol=1e-10,
        popsize=popsize,
        mutation=(0.5, 1.5),
        recombination=0.9,
        polish=True,
        workers=1,
        callback=_callback,
    )

    final_candidate = {
        "params": np.array(result.x, dtype=np.float64).copy(),
        "loss": float(result.fun),
        "exact": int(_exact_near(result.x)["exact"]),
        "near": int(_exact_near(result.x)["near"]),
    }
    selected_from = "final"
    if best_candidate["params"] is not None and _is_better_candidate(best_candidate, final_candidate):
        p = np.array(best_candidate["params"], dtype=np.float64)
        selected_from = "best_checkpoint"
    else:
        p = np.array(final_candidate["params"], dtype=np.float64)

    optimal_params = {
        "w_camera": round(float(p[0]), 2),
        "w_weather": round(float(p[1]), 2),
        "w_confidence": round(float(p[2]), 2),
        "difficulty_factors": {
            "1": round(float(p[3]), 3),
            "2": round(float(p[4]), 3),
            "3": round(float(p[5]), 3),
            "4": round(float(p[6]), 3),
            "5": round(float(p[7]), 3),
        },
        "neutral_cam": round(float(p[8]), 3),
        "wf_light_precip": round(float(p[9]), 3),
        "wf_mod_precip": round(float(p[10]), 3),
        "wf_heavy_precip": round(float(p[11]), 3),
        "wf_near_freeze": round(float(p[12]), 3),
        "wf_freeze": round(float(p[13]), 3),
        "smooth_window_sec": round(float(_smooth_window_from_params(p)), 2),
    }

    per_video = []
    for vd in video_data:
        rec_cont = _median_rec(vd, p)
        rec_rounded = int(round(rec_cont / 10.0) * 10)
        rec_rounded = max(20, min(rec_rounded, vd["posted_speed"]))
        delta = rec_rounded - vd["target_speed"]
        per_video.append(
            {
                "video_id": vd["video_id"],
                "difficulty": vd["difficulty"],
                "target": vd["target_speed"],
                "mean_raw": round(vd["mean_raw"], 1),
                "recommended": rec_rounded,
                "delta": delta,
                "match": delta == 0,
            }
        )

    n_match_exact = sum(1 for v in per_video if v["match"])
    n_match_near = sum(1 for v in per_video if abs(v["delta"]) <= 10)

    return {
        "params": optimal_params,
        "converged": bool(result.success),
        "final_loss": round(float(loss(p)), 4),
        "n_iterations": int(result.nit),
        "stop_reason": best["reason"],
        "selected_from": selected_from,
        "n_match": n_match_exact,
        "n_match_near": n_match_near,
        "total": len(per_video),
        "per_video": per_video,
    }
