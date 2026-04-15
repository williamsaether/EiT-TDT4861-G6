from __future__ import annotations

from pathlib import Path

from flask import Blueprint, jsonify, render_template, request, send_from_directory

from app.tuning.config import (
    CLASS_NAMES,
    DEFAULT_MODEL_NAME,
    EXCLUDED_CLASSES,
    IMAGE_SIZE,
    MODELS_DIR,
    available_models,
)
from app.driving.services.context_service import ContextService
from app.driving.services.example_video_service import ExampleVideoService
from app.driving.services.recommendation_service import RecommendationService


api_bp = Blueprint(
    "driving",
    __name__,
    url_prefix="/driving",
    static_folder="../static",
    template_folder="../templates",
)
context_service = ContextService()
recommendation_service = RecommendationService()
example_video_service = ExampleVideoService()


@api_bp.route("/")
def index() -> str:
    return render_template("driving/index.html")


@api_bp.route("/examples")
def examples() -> str:
    return render_template("driving/examples.html")


@api_bp.route("/videos/<path:filename>")
def serve_example_video(filename: str):
    clean_name = Path(filename).name
    if clean_name != filename or not example_video_service.is_allowed(clean_name):
        return jsonify({"ok": False, "error": "video not allowed"}), 404
    return send_from_directory(str(Path(__file__).resolve().parents[3] / "videos"), clean_name, conditional=True)


@api_bp.route("/models/<path:filename>")
def serve_onnx_model(filename: str):
    clean_name = Path(filename).name
    models = set(available_models())
    if clean_name != filename or clean_name not in models:
        return jsonify({"ok": False, "error": "model not allowed"}), 404
    return send_from_directory(str(MODELS_DIR), clean_name, conditional=True)


@api_bp.route("/api/examples", methods=["GET"])
def list_examples():
    return jsonify({"ok": True, "examples": example_video_service.list_examples()})


@api_bp.route("/api/models", methods=["GET"])
def list_models():
    models = available_models()
    current = DEFAULT_MODEL_NAME if DEFAULT_MODEL_NAME in models else (models[0] if models else "")
    return jsonify(
        {
            "ok": True,
            "models": models,
            "current": current,
            "class_names": CLASS_NAMES,
            "excluded_classes": sorted(EXCLUDED_CLASSES),
            "image_size": IMAGE_SIZE,
        }
    )


@api_bp.route("/api/context", methods=["POST"])
def context():
    payload = request.get_json(silent=True) or {}
    context_data = context_service.get_context(payload)
    return jsonify({"ok": True, **context_data})


@api_bp.route("/api/recommend", methods=["POST"])
def recommend():
    payload = request.get_json(silent=True) or {}
    result = recommendation_service.recommend(payload)
    return jsonify({"ok": True, **result})
