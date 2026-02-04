from pathlib import Path
import io
import json
import time

from flask import Flask, request, jsonify, send_from_directory
import numpy as np
from PIL import Image
import onnxruntime as ort

app = Flask(__name__, static_folder="static", template_folder="templates")

ROOT = Path(__file__).resolve().parent.parent
DATA_ROOT = ROOT / "data" / "RSCD dataset-1million"
TRAIN_DIR = DATA_ROOT / "train"
MODEL_PATH = ROOT / "rscd_resnet18.onnx"

IMAGE_SIZE = 224

FRICTION_CLASSES = ["dry", "wet", "water"]
SURFACE_CLASSES = ["asphalt", "concrete", "gravel", "mud"]
WINTER_CLASSES = ["fresh_snow", "melted_snow", "ice"]
UNEVEN_CLASSES = ["smooth", "slight", "severe"]

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES_FALLBACK = [
    "dry_asphalt_severe",
    "dry_asphalt_slight",
    "dry_asphalt_smooth",
    "dry_concrete_severe",
    "dry_concrete_slight",
    "dry_concrete_smooth",
    "dry_gravel",
    "dry_mud",
    "fresh_snow",
    "ice",
    "melted_snow",
    "water_asphalt_severe",
    "water_asphalt_slight",
    "water_asphalt_smooth",
    "water_concrete_severe",
    "water_concrete_slight",
    "water_concrete_smooth",
    "water_gravel",
    "water_mud",
    "wet_asphalt_severe",
    "wet_asphalt_slight",
    "wet_asphalt_smooth",
    "wet_concrete_severe",
    "wet_concrete_slight",
    "wet_concrete_smooth",
    "wet_gravel",
    "wet_mud",
]

def get_class_names():
    if not TRAIN_DIR.exists():
        print(f"Missing train dir: {TRAIN_DIR}. Using fallback class list.")
        return CLASS_NAMES_FALLBACK
    class_names = sorted([p.name for p in TRAIN_DIR.iterdir() if p.is_dir()])
    return class_names


def parse_groups(label_name: str):
    parts = label_name.split("_")
    friction = None
    surface = None
    uneven = None
    winter = None

    if label_name in WINTER_CLASSES:
        winter = label_name
        return friction, surface, uneven, winter

    if parts and parts[0] in FRICTION_CLASSES:
        friction = parts[0]

    for p in parts[1:]:
        if p in SURFACE_CLASSES:
            surface = p
        elif p in UNEVEN_CLASSES:
            uneven = p

    return friction, surface, uneven, winter


CLASS_NAMES = get_class_names()
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
INDEX_GROUPS = {i: parse_groups(name) for i, name in IDX_TO_CLASS.items()}

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Missing ONNX model at {MODEL_PATH}. "
        "Run export_for_web.ipynb to create rscd_resnet18.onnx."
    )

SESSION = ort.InferenceSession(str(MODEL_PATH), providers=["CPUExecutionProvider"])
INPUT_NAME = SESSION.get_inputs()[0].name


def preprocess(image: Image.Image) -> np.ndarray:
    image = image.convert("RGB")
    image = image.resize((IMAGE_SIZE, IMAGE_SIZE))
    arr = np.asarray(image).astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    arr = np.expand_dims(arr, axis=0)
    return arr


def topk(scores, k=3):
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]


def predict_grouped(image: Image.Image):
    x = preprocess(image)
    logits = SESSION.run(None, {INPUT_NAME: x})[0]
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs = probs[0]

    friction_scores = {k: 0.0 for k in FRICTION_CLASSES}
    surface_scores = {k: 0.0 for k in SURFACE_CLASSES}
    winter_scores = {k: 0.0 for k in WINTER_CLASSES}
    uneven_scores = {k: 0.0 for k in UNEVEN_CLASSES}

    for idx, p in enumerate(probs):
        friction, surface, uneven, winter = INDEX_GROUPS[idx]
        if friction is not None:
            friction_scores[friction] += float(p)
        if surface is not None:
            surface_scores[surface] += float(p)
        if uneven is not None:
            uneven_scores[uneven] += float(p)
        if winter is not None:
            winter_scores[winter] += float(p)

    return {
        "friction": topk(friction_scores),
        "surface": topk(surface_scores),
        "uneven": topk(uneven_scores),
        "winter": topk(winter_scores),
        "raw_top": topk({IDX_TO_CLASS[i]: float(p) for i, p in enumerate(probs)}, k=5),
    }


@app.route("/")
def index():
    return send_from_directory("templates", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return jsonify({"error": "missing image"}), 400
    file = request.files["image"]
    image = Image.open(io.BytesIO(file.read()))
    result = predict_grouped(image)
    return jsonify(result)


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=8000, debug=False)
