from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

try:
    import onnxruntime as ort
except ImportError as exc:  # pragma: no cover
    raise SystemExit(
        "onnxruntime is not installed. Install it first, e.g. `pip install onnxruntime`."
    ) from exc


IMAGE_SIZE = 224
MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)

CLASS_NAMES_12 = [
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

CLASS_NAMES_27 = [
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run ONNX inference on one image.")
    parser.add_argument("--image", required=True, type=Path, help="Path to image file")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("models/onnx/rscd_resnet18_v2.onnx"),
        help="Path to ONNX model file",
    )
    parser.add_argument("--topk", type=int, default=5, help="How many top predictions to print")
    parser.add_argument(
        "--labels",
        type=Path,
        default=None,
        help="Optional JSON file with class names list. Overrides auto label mapping.",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print machine-readable JSON instead of text table",
    )
    return parser.parse_args()


def load_image_rgb(path: Path) -> np.ndarray:
    try:
        from PIL import Image

        image = Image.open(path).convert("RGB")
        return np.asarray(image)
    except ImportError:
        pass

    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit(
            "Neither Pillow nor OpenCV is installed. Install one of them "
            "(e.g. `pip install pillow`)."
        ) from exc

    img_bgr = cv2.imread(str(path))
    if img_bgr is None:
        raise FileNotFoundError(f"Could not read image: {path}")
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    return img_rgb


def preprocess(img_rgb: np.ndarray) -> np.ndarray:
    try:
        import cv2
    except ImportError as exc:  # pragma: no cover
        raise SystemExit("OpenCV is required for resizing. Install with `pip install opencv-python`.") from exc

    resized = cv2.resize(img_rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
    arr = resized.astype(np.float32) / 255.0
    arr = (arr - MEAN) / STD
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0)


def softmax(logits: np.ndarray) -> np.ndarray:
    x = logits - np.max(logits)
    ex = np.exp(x)
    return ex / ex.sum()


def resolve_class_names(num_classes: int, labels_path: Path | None) -> list[str]:
    if labels_path is not None:
        data: Any = json.loads(labels_path.read_text(encoding="utf-8"))
        if not isinstance(data, list) or len(data) != num_classes:
            raise ValueError(f"--labels must be a JSON list with exactly {num_classes} items")
        return [str(x) for x in data]

    if num_classes == len(CLASS_NAMES_12):
        return CLASS_NAMES_12
    if num_classes == len(CLASS_NAMES_27):
        return CLASS_NAMES_27
    return [f"class_{i}" for i in range(num_classes)]


def run_inference(model_path: Path, input_tensor: np.ndarray) -> np.ndarray:
    if not model_path.exists():
        raise FileNotFoundError(f"Model not found: {model_path}")

    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    out = session.run(None, {input_name: input_tensor})[0]

    if out.ndim == 2:
        return out[0]
    if out.ndim == 1:
        return out
    raise ValueError(f"Unexpected output shape from model: {out.shape}")


def main() -> None:
    args = parse_args()
    img = load_image_rgb(args.image)
    x = preprocess(img)
    logits = run_inference(args.model, x)
    probs = softmax(logits)

    class_names = resolve_class_names(num_classes=probs.shape[0], labels_path=args.labels)
    pairs = list(zip(class_names, probs.tolist()))
    pairs.sort(key=lambda item: item[1], reverse=True)

    topk = max(1, min(args.topk, len(pairs)))
    top_pairs = pairs[:topk]
    top_label, top_conf = top_pairs[0]

    result = {
        "image": str(args.image),
        "model": str(args.model),
        "predicted_label": top_label,
        "confidence": float(top_conf),
        "topk": [{"label": lbl, "score": float(score)} for lbl, score in top_pairs],
    }

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
        return

    print(f"Image: {args.image}")
    print(f"Model: {args.model}")
    print(f"Predicted: {top_label} ({top_conf:.4f})")
    print("Top predictions:")
    for idx, (label, score) in enumerate(top_pairs, start=1):
        print(f"  {idx:>2}. {label:<24} {score:.4f}")


if __name__ == "__main__":
    main()
