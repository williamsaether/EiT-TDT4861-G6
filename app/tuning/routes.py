from __future__ import annotations

import threading
import time
from typing import Any

from flask import Blueprint, jsonify, render_template, request, send_from_directory

from app.tuning.analyzer import VideoAnalyzer
from app.tuning.autotune_log import build_run_id, utc_now_iso, write_autotune_log
from app.tuning.autotune_service import run_auto_tune
from app.tuning.config import (
    CLASS_COLORS,
    CLASS_NAMES,
    CROP_BOTTOM,
    CROP_LEFT,
    CROP_RIGHT,
    CROP_TOP,
    DIFFICULTY_LEVEL,
    DIFFICULTY_NAMES,
    VIDEO_BY_ID,
    VIDEOS,
    VIDEOS_DIR,
    available_models,
)
from app.tuning.recommendation_service import all_recommendations, compute_recommendation
from app.tuning.settings_store import load_settings, save_settings
from app.tuning.state import TuningState


tuning_bp = Blueprint(
    "tuning",
    __name__,
    url_prefix="/tuning",
    static_folder="static",
    template_folder="templates",
)


tuning_state = TuningState()
analyzer = VideoAnalyzer()
_autotune_lock = threading.Lock()
_autotune_job: dict[str, Any] = {
    "run_id": None,
    "running": False,
    "status": "idle",
    "progress": 0.0,
    "iteration": 0,
    "maxiter": 0,
    "best_loss": None,
    "best_exact": 0,
    "best_near": 0,
    "message": "",
    "result": None,
    "stop_requested": False,
    "started_at": None,
    "finished_at": None,
    "log_file": None,
}


def api_error(message: str, status: int = 400) -> Any:
    return jsonify({"ok": False, "error": message}), status


def _snapshot_settings() -> dict[str, Any]:
    with tuning_state.lock:
        t_state = tuning_state.to_dict()
    return {
        "tuning_state": t_state,
        "analyzer": {
            "model": analyzer.current_model_name,
            "use_crop": bool(analyzer.use_crop),
            "smooth_window_sec": float(tuning_state.smooth_window_sec),
        },
        "ui": {},
    }


def _persist_settings() -> str:
    return save_settings(_snapshot_settings())


def _apply_loaded_settings() -> None:
    loaded = load_settings()
    t_payload = loaded.get("tuning_state") or {}
    if isinstance(t_payload, dict):
        with tuning_state.lock:
            tuning_state.apply(t_payload)

    analyzer_cfg = loaded.get("analyzer") or {}
    if isinstance(analyzer_cfg, dict):
        desired_model = str(analyzer_cfg.get("model", "")).strip()
        if desired_model and desired_model != analyzer.current_model_name:
            analyzer.switch_model(desired_model)
        desired_crop = bool(analyzer_cfg.get("use_crop", False))
        if desired_crop != analyzer.use_crop:
            analyzer.set_crop(desired_crop)
        if "smooth_window_sec" in analyzer_cfg:
            with tuning_state.lock:
                tuning_state.apply({"smooth_window_sec": analyzer_cfg["smooth_window_sec"]})

    # Backward compatibility for previously stored UI-only smoothing.
    ui_cfg = loaded.get("ui") or {}
    if isinstance(ui_cfg, dict) and "smooth_window_sec" in ui_cfg:
        with tuning_state.lock:
            tuning_state.apply({"smooth_window_sec": ui_cfg["smooth_window_sec"]})


_apply_loaded_settings()


@tuning_bp.route("/")
def index() -> Any:
    return render_template("tuning/index.html")


@tuning_bp.route("/videos/<path:filename>")
def serve_video(filename: str) -> Any:
    return send_from_directory(str(VIDEOS_DIR), filename, conditional=True)


@tuning_bp.route("/api/init", methods=["GET"])
def api_init() -> Any:
    with tuning_state.lock:
        state = tuning_state.to_dict()
        recs = all_recommendations(tuning_state, analyzer, VIDEOS)
    n_match = sum(1 for r in recs if r["match"])
    return jsonify(
        {
            "videos": VIDEOS,
            "state": state,
            "recommendations": recs,
            "n_match": n_match,
            "total": len(recs),
            "analyzer": analyzer.status_summary(),
            "crop_rect": {
                "top": CROP_TOP,
                "bottom": CROP_BOTTOM,
                "left": CROP_LEFT,
                "right": CROP_RIGHT,
            },
            "crop_overrides": analyzer.get_crop_overrides(),
            "settings": _snapshot_settings(),
            "difficulty_names": DIFFICULTY_NAMES,
            "class_colors": CLASS_COLORS,
        }
    )


@tuning_bp.route("/api/recommendations", methods=["GET"])
def api_recommendations() -> Any:
    with tuning_state.lock:
        recs = all_recommendations(tuning_state, analyzer, VIDEOS)
        state = tuning_state.to_dict()
    n_match = sum(1 for r in recs if r["match"])
    return jsonify({"recommendations": recs, "n_match": n_match, "total": len(recs), "state": state})


@tuning_bp.route("/api/tune", methods=["POST"])
def api_tune() -> Any:
    payload = request.get_json(silent=True) or {}
    with tuning_state.lock:
        tuning_state.apply(payload)
        state = tuning_state.to_dict()
        recs = all_recommendations(tuning_state, analyzer, VIDEOS)
    _persist_settings()
    n_match = sum(1 for r in recs if r["match"])
    return jsonify({"state": state, "recommendations": recs, "n_match": n_match, "total": len(recs)})


@tuning_bp.route("/api/analyzer-status", methods=["GET"])
def api_analyzer_status() -> Any:
    return jsonify(analyzer.status_summary())


@tuning_bp.route("/api/models", methods=["GET"])
def api_list_models() -> Any:
    return jsonify({"models": available_models(), "current": analyzer.current_model_name})


@tuning_bp.route("/api/model", methods=["POST"])
def api_switch_model() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    model_name = payload.get("model", "").strip()
    if not model_name:
        return api_error("Missing 'model'", 400)
    result = analyzer.switch_model(model_name)
    if result.get("ok"):
        _persist_settings()
    return jsonify(result)


@tuning_bp.route("/api/crop", methods=["POST"])
def api_set_crop() -> Any:
    payload = request.get_json(force=True, silent=True) or {}
    use_crop = bool(payload.get("use_crop", False))
    result = analyzer.set_crop(use_crop)
    if result.get("ok"):
        _persist_settings()
    return jsonify(result)


@tuning_bp.route("/api/settings", methods=["GET"])
def api_settings_get() -> Any:
    return jsonify({"ok": True, "settings": _snapshot_settings()})


@tuning_bp.route("/api/settings", methods=["POST"])
def api_settings_set() -> Any:
    payload = request.get_json(silent=True) or {}
    analyzer_cfg = payload.get("analyzer") or {}
    if isinstance(analyzer_cfg, dict) and "smooth_window_sec" in analyzer_cfg:
        try:
            with tuning_state.lock:
                tuning_state.apply({"smooth_window_sec": analyzer_cfg["smooth_window_sec"]})
        except Exception:
            return api_error("Invalid analyzer.smooth_window_sec", 400)

    # Backward-compatible input path.
    ui = payload.get("ui") or {}
    if isinstance(ui, dict) and "smooth_window_sec" in ui:
        try:
            with tuning_state.lock:
                tuning_state.apply({"smooth_window_sec": ui["smooth_window_sec"]})
        except Exception:
            return api_error("Invalid ui.smooth_window_sec", 400)
    path = _persist_settings()
    return jsonify({"ok": True, "settings": _snapshot_settings(), "settings_file": path})


@tuning_bp.route("/api/video/<video_id>/predictions", methods=["GET"])
def api_video_predictions(video_id: str) -> Any:
    with analyzer.lock:
        data = dict(analyzer.data.get(video_id, {}))
    if not data:
        return api_error("video not found", 404)

    video_meta = VIDEO_BY_ID.get(video_id)
    if not video_meta:
        return api_error("video metadata not found", 404)

    with tuning_state.lock:
        wf, wf_reasons = tuning_state.compute_weather_factor(
            video_meta["temp_c"], video_meta["precipitation_mm_h"]
        )
        avg = analyzer.get_avg_scores(video_id)
        rec = compute_recommendation(
            avg,
            video_meta["posted_speed"],
            tuning_state,
            weather_factor=wf,
            temp_c=float(video_meta["temp_c"]),
            humidity=float(video_meta["humidity"]) if video_meta.get("humidity") is not None else None,
            precip_mm_h=float(video_meta["precipitation_mm_h"]),
        )

    return jsonify(
        {
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
        }
    )


@tuning_bp.route("/api/video/<video_id>/frame-recommendation", methods=["POST"])
def api_frame_recommendation(video_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    scores = payload.get("scores", {})
    video_meta = VIDEO_BY_ID.get(video_id)
    if not video_meta:
        return api_error("video not found", 404)

    with tuning_state.lock:
        wf, _ = tuning_state.compute_weather_factor(video_meta["temp_c"], video_meta["precipitation_mm_h"])
        rec = compute_recommendation(
            scores if scores else None,
            video_meta["posted_speed"],
            tuning_state,
            weather_factor=wf,
            temp_c=float(video_meta["temp_c"]),
            humidity=float(video_meta["humidity"]) if video_meta.get("humidity") is not None else None,
            precip_mm_h=float(video_meta["precipitation_mm_h"]),
        )
    return jsonify(rec)


@tuning_bp.route("/api/video/<video_id>/crop", methods=["GET"])
def api_video_crop_get(video_id: str) -> Any:
    if video_id not in VIDEO_BY_ID:
        return api_error("video not found", 404)
    overrides = analyzer.get_crop_overrides()
    return jsonify(
        {
            "ok": True,
            "video_id": video_id,
            "use_crop": analyzer.use_crop,
            "crop": analyzer.get_effective_crop(video_id),
            "has_override": video_id in overrides,
        }
    )


@tuning_bp.route("/api/video/<video_id>/crop", methods=["POST"])
def api_video_crop_set(video_id: str) -> Any:
    payload = request.get_json(silent=True) or {}
    rect = payload.get("crop") or {}
    if not isinstance(rect, dict):
        return api_error("Invalid 'crop' payload", 400)
    result = analyzer.set_video_crop(video_id, rect)
    if not result.get("ok"):
        return api_error(result.get("error", "failed to set crop"), 404)
    return jsonify(result)


@tuning_bp.route("/api/video/<video_id>/crop", methods=["DELETE"])
def api_video_crop_clear(video_id: str) -> Any:
    result = analyzer.clear_video_crop(video_id)
    if not result.get("ok"):
        return api_error(result.get("error", "failed to clear crop"), 404)
    return jsonify(result)


@tuning_bp.route("/api/auto-tune", methods=["POST"])
def api_auto_tune() -> Any:
    # Backward-compatible alias: starting async auto-tune job.
    return api_auto_tune_start()


@tuning_bp.route("/api/auto-tune/start", methods=["POST"])
def api_auto_tune_start() -> Any:
    status = analyzer.status_summary()
    pending = [
        vid
        for vid, d in status["videos"].items()
        if d["status"] not in ("done", "error", "file_not_found", "no_opencv")
    ]
    if pending:
        return api_error(f"Videos still processing: {pending}. Please wait.", 400)

    with _autotune_lock:
        if _autotune_job["running"]:
            return api_error("Auto-tune already running", 409)
        run_id = build_run_id()
        _autotune_job.update(
            {
                "run_id": run_id,
                "running": True,
                "status": "running",
                "progress": 0.0,
                "iteration": 0,
                "maxiter": 3000,
                "best_loss": None,
                "best_exact": 0,
                "best_near": 0,
                "message": "Auto-tune started",
                "result": None,
                "stop_requested": False,
                "started_at": time.time(),
                "finished_at": None,
                "log_file": None,
            }
        )

    def _progress_update(p: dict[str, Any]) -> None:
        with _autotune_lock:
            _autotune_job["iteration"] = int(p.get("iteration", 0))
            _autotune_job["maxiter"] = int(p.get("maxiter", _autotune_job["maxiter"]))
            _autotune_job["progress"] = float(p.get("progress", 0.0))
            _autotune_job["best_loss"] = p.get("best_loss")
            _autotune_job["best_exact"] = int(p.get("best_exact", 0))
            _autotune_job["best_near"] = int(p.get("best_near", 0))

    def _should_stop() -> bool:
        with _autotune_lock:
            return bool(_autotune_job["stop_requested"])

    def _runner() -> None:
        run_id: str = ""
        started_ts: float | None = None
        try:
            with tuning_state.lock:
                state_before = tuning_state.to_dict()
                recs_before = all_recommendations(tuning_state, analyzer, VIDEOS)
                n_match_before = sum(1 for r in recs_before if r["match"])
                n_near_before = sum(1 for r in recs_before if abs(r["delta"]) <= 10)

            with _autotune_lock:
                run_id = str(_autotune_job.get("run_id") or build_run_id())
                started_ts = _autotune_job.get("started_at")

            result = run_auto_tune(
                analyzer,
                VIDEOS,
                smooth_window_sec=float(tuning_state.smooth_window_sec),
                progress_cb=_progress_update,
                should_stop_cb=_should_stop,
                maxiter=3000,
                popsize=25,
            )
            if "error" in result:
                raise RuntimeError(result["error"])

            with tuning_state.lock:
                tuning_state.apply(result["params"])
                state = tuning_state.to_dict()
                recs_after = all_recommendations(tuning_state, analyzer, VIDEOS)
            _persist_settings()

            n_match_after = sum(1 for r in recs_after if r["match"])
            n_near_after = sum(1 for r in recs_after if abs(r["delta"]) <= 10)

            payload = {
                "run_id": run_id,
                "params": result["params"],
                "converged": result["converged"],
                "final_loss": result["final_loss"],
                "n_iterations": result["n_iterations"],
                "stop_reason": result.get("stop_reason", ""),
                "selected_from": result.get("selected_from", "final"),
                "per_video": result["per_video"],
                "n_match_before": n_match_before,
                "n_match_after": n_match_after,
                "n_near_before": n_near_before,
                "n_near_after": n_near_after,
                "total": len(recs_after),
                "state": state,
                "recommendations": recs_after,
            }

            log_payload = {
                "run_id": run_id,
                "kind": "auto_tune",
                "status": "done",
                "created_at_utc": utc_now_iso(),
                "started_at_unix": started_ts,
                "finished_at_unix": time.time(),
                "inputs": {
                    "maxiter": 3000,
                    "popsize": 25,
                    "total_videos": len(VIDEOS),
                    "video_ids": [v["video_id"] for v in VIDEOS],
                    "state_before": state_before,
                },
                "result": payload,
            }
            log_file = write_autotune_log(run_id, log_payload)

            with _autotune_lock:
                _autotune_job["running"] = False
                _autotune_job["status"] = "done"
                _autotune_job["message"] = "Auto-tune completed"
                _autotune_job["progress"] = 1.0
                _autotune_job["result"] = payload
                _autotune_job["finished_at"] = time.time()
                _autotune_job["log_file"] = log_file
        except Exception as exc:
            log_file = None
            try:
                with tuning_state.lock:
                    state_on_error = tuning_state.to_dict()
                with _autotune_lock:
                    run_id = str(_autotune_job.get("run_id") or run_id or build_run_id())
                    started_ts = _autotune_job.get("started_at")
                log_payload = {
                    "run_id": run_id,
                    "kind": "auto_tune",
                    "status": "error",
                    "created_at_utc": utc_now_iso(),
                    "started_at_unix": started_ts,
                    "finished_at_unix": time.time(),
                    "error": str(exc),
                    "inputs": {
                        "maxiter": 3000,
                        "popsize": 25,
                        "total_videos": len(VIDEOS),
                        "video_ids": [v["video_id"] for v in VIDEOS],
                    },
                    "state_on_error": state_on_error,
                }
                log_file = write_autotune_log(run_id, log_payload)
            except Exception:
                log_file = None
            with _autotune_lock:
                _autotune_job["running"] = False
                _autotune_job["status"] = "error"
                _autotune_job["message"] = str(exc)
                _autotune_job["finished_at"] = time.time()
                _autotune_job["log_file"] = log_file

    threading.Thread(target=_runner, daemon=True).start()
    with _autotune_lock:
        return jsonify({"ok": True, "status": "running", "run_id": _autotune_job["run_id"]})


@tuning_bp.route("/api/auto-tune/status", methods=["GET"])
def api_auto_tune_status() -> Any:
    with _autotune_lock:
        return jsonify(
            {
                "ok": True,
                "running": _autotune_job["running"],
                "run_id": _autotune_job["run_id"],
                "status": _autotune_job["status"],
                "progress": _autotune_job["progress"],
                "iteration": _autotune_job["iteration"],
                "maxiter": _autotune_job["maxiter"],
                "best_loss": _autotune_job["best_loss"],
                "best_exact": _autotune_job["best_exact"],
                "best_near": _autotune_job["best_near"],
                "message": _autotune_job["message"],
                "result": _autotune_job["result"],
                "stop_requested": _autotune_job["stop_requested"],
                "started_at": _autotune_job["started_at"],
                "finished_at": _autotune_job["finished_at"],
                "log_file": _autotune_job["log_file"],
            }
        )


@tuning_bp.route("/api/auto-tune/stop", methods=["POST"])
def api_auto_tune_stop() -> Any:
    with _autotune_lock:
        if not _autotune_job["running"]:
            return api_error("No running auto-tune job", 409)
        _autotune_job["stop_requested"] = True
        _autotune_job["message"] = "Stop requested"
    return jsonify({"ok": True, "status": "stopping"})
