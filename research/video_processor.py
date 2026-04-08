import cv2
import numpy as np
import onnxruntime as ort
from PIL import Image
import time
import argparse
from pathlib import Path
import json
from typing import Optional

# Local imports
try:
    from weather import weather_met
    from speed_limit import nvdb_speed
except ImportError:
    pass

# ================= Constants from app.py =================
IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

FRICTION_CLASSES = ["dry", "wet", "water"]
SURFACE_CLASSES = ["asphalt", "concrete", "gravel", "mud"]
WINTER_CLASSES = ["fresh_snow", "melted_snow", "ice"]
UNEVEN_CLASSES = ["smooth", "slight", "severe"]

CLASS_NAMES_FALLBACK = [
    "dry_asphalt_severe", "dry_asphalt_slight", "dry_asphalt_smooth",
    "dry_concrete_severe", "dry_concrete_slight", "dry_concrete_smooth",
    "dry_gravel", "dry_mud",
    "fresh_snow", "ice", "melted_snow",
    "water_asphalt_severe", "water_asphalt_slight", "water_asphalt_smooth",
    "water_concrete_severe", "water_concrete_slight", "water_concrete_smooth",
    "water_gravel", "water_mud",
    "wet_asphalt_severe", "wet_asphalt_slight", "wet_asphalt_smooth",
    "wet_concrete_severe", "wet_concrete_slight", "wet_concrete_smooth",
    "wet_gravel", "wet_mud",
]

# Use fallback as default since we don't assume training dir structure exists here
CLASS_NAMES = CLASS_NAMES_FALLBACK 
IDX_TO_CLASS = {i: name for i, name in enumerate(CLASS_NAMES)}
ROOT = Path(__file__).resolve().parents[1]
DEFAULT_ONNX_PATH = ROOT / "models" / "onnx" / "rscd_resnet18.onnx"

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

INDEX_GROUPS = {i: parse_groups(name) for i, name in IDX_TO_CLASS.items()}


# ================= Helper Functions =================

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

_SESSION = None
_INPUT_NAME = None

def get_surface(image: Image.Image, model_path: Optional[str] = None):
    """
    Predict surface conditions from an image using the ONNX model.
    """
    global _SESSION, _INPUT_NAME
    
    if _SESSION is None:
        model_file = Path(model_path) if model_path else DEFAULT_ONNX_PATH
        if not model_file.exists():
            raise FileNotFoundError(f"Model not found at {model_file}")
        _SESSION = ort.InferenceSession(str(model_file), providers=["CPUExecutionProvider"])
        _INPUT_NAME = _SESSION.get_inputs()[0].name

    x = preprocess(image)
    logits = _SESSION.run(None, {_INPUT_NAME: x})[0]
    probs = np.exp(logits - np.max(logits, axis=1, keepdims=True))
    probs = probs / probs.sum(axis=1, keepdims=True)
    probs = probs[0]

    raw = {
        "friction": {k: 0.0 for k in FRICTION_CLASSES},
        "surface": {k: 0.0 for k in SURFACE_CLASSES},
        "winter": {k: 0.0 for k in WINTER_CLASSES},
        "uneven": {k: 0.0 for k in UNEVEN_CLASSES}
    }

    for idx, p in enumerate(probs):
        friction, surface, uneven, winter = INDEX_GROUPS[idx]
        if friction is not None:
            raw["friction"][friction] += float(p)
        if surface is not None:
            raw["surface"][surface] += float(p)
        if uneven is not None:
            raw["uneven"][uneven] += float(p)
        if winter is not None:
            raw["winter"][winter] += float(p)

    # Re-calculate topk based on adjusted raw scores
    return {
        "friction": topk(raw["friction"], k=1)[0],
        "surface": topk(raw["surface"], k=1)[0],
        "uneven": topk(raw["uneven"], k=1)[0],
        "winter": topk(raw["winter"], k=1)[0],
        "raw_scores": raw
    }
