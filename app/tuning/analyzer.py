from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from app.tuning.config import (
    CLASS_NAMES,
    CROP_BOTTOM,
    CROP_LEFT,
    CROP_RIGHT,
    CROP_TOP,
    DEFAULT_MODEL_NAME,
    EXCLUDED_CLASSES,
    IMAGE_SIZE,
    MODELS_DIR,
    VIDEOS,
    VIDEOS_DIR,
    available_models,
)

try:
    import cv2

    HAS_CV2 = True
except ImportError:
    cv2 = None  # type: ignore[assignment]
    HAS_CV2 = False

try:
    import numpy as np

    HAS_NUMPY = True
except ImportError:
    np = None  # type: ignore[assignment]
    HAS_NUMPY = False

try:
    import onnxruntime as ort

    HAS_ONNX = True
except ImportError:
    ort = None  # type: ignore[assignment]
    HAS_ONNX = False


class VideoAnalyzer:
    """Background-processes each video and stores per-frame class scores."""

    def __init__(self) -> None:
        self.lock = threading.Lock()
        self.model = None
        self.model_input_name: Optional[str] = None
        self.model_status = "not_loaded"
        self.current_model_name: str = DEFAULT_MODEL_NAME
        self.use_crop: bool = False
        self._crop_overrides_path = Path(__file__).resolve().parent / "crop_overrides.json"
        self.video_crop_overrides: Dict[str, Dict[str, float]] = self._load_crop_overrides()
        self.data: Dict[str, Dict[str, Any]] = {
            v["video_id"]: {"frames": [], "status": "pending", "duration": 0.0} for v in VIDEOS
        }
        self._load_model(DEFAULT_MODEL_NAME)
        threading.Thread(target=self._precompute_all, daemon=True).start()

    @staticmethod
    def _normalize_crop_rect(rect: Dict[str, Any]) -> Dict[str, float]:
        left = float(rect.get("left", CROP_LEFT))
        right = float(rect.get("right", CROP_RIGHT))
        top = float(rect.get("top", CROP_TOP))
        bottom = float(rect.get("bottom", CROP_BOTTOM))

        left = max(0.0, min(0.95, left))
        right = max(0.05, min(1.0, right))
        top = max(0.0, min(0.95, top))
        bottom = max(0.05, min(1.0, bottom))

        # Keep a sane minimum crop size.
        if right - left < 0.05:
            right = min(1.0, left + 0.05)
            left = max(0.0, right - 0.05)
        if bottom - top < 0.05:
            bottom = min(1.0, top + 0.05)
            top = max(0.0, bottom - 0.05)

        return {
            "left": round(left, 4),
            "right": round(right, 4),
            "top": round(top, 4),
            "bottom": round(bottom, 4),
        }

    def _load_crop_overrides(self) -> Dict[str, Dict[str, float]]:
        if not self._crop_overrides_path.exists():
            return {}
        try:
            with open(self._crop_overrides_path, encoding="utf-8") as f:
                raw = json.load(f)
            if not isinstance(raw, dict):
                return {}
            out: Dict[str, Dict[str, float]] = {}
            valid_ids = {v["video_id"] for v in VIDEOS}
            for vid, rect in raw.items():
                if vid in valid_ids and isinstance(rect, dict):
                    out[vid] = self._normalize_crop_rect(rect)
            return out
        except Exception:
            return {}

    def _persist_crop_overrides(self) -> None:
        try:
            with open(self._crop_overrides_path, "w", encoding="utf-8") as f:
                json.dump(self.video_crop_overrides, f, ensure_ascii=False, indent=2)
        except Exception:
            # Non-fatal: overrides still work in-memory.
            pass

    def _load_model(self, model_name: str) -> None:
        if not (HAS_ONNX and HAS_NUMPY):
            self.model_status = "missing onnxruntime / numpy"
            return
        path = MODELS_DIR / model_name
        if not path.exists():
            self.model_status = f"model not found: {path}"
            return
        try:
            opts = ort.SessionOptions()
            opts.inter_op_num_threads = 2
            opts.intra_op_num_threads = 2
            self.model = ort.InferenceSession(
                str(path), sess_options=opts, providers=["CPUExecutionProvider"]
            )
            self.model_input_name = self.model.get_inputs()[0].name
            self.model_status = "ok"
            self.current_model_name = model_name
        except Exception as exc:
            self.model_status = f"load error: {exc}"

    def switch_model(self, model_name: str) -> Dict[str, Any]:
        if model_name not in available_models():
            return {"ok": False, "error": f"Unknown model: {model_name}"}
        with self.lock:
            for vid_id in self.data:
                self.data[vid_id] = {"frames": [], "status": "pending", "duration": 0.0}
            self._load_model(model_name)
        threading.Thread(target=self._precompute_all, daemon=True).start()
        return {"ok": True, "model": model_name, "status": self.model_status}

    def set_crop(self, use_crop: bool) -> Dict[str, Any]:
        with self.lock:
            self.use_crop = use_crop
            for vid_id in self.data:
                self.data[vid_id] = {"frames": [], "status": "pending", "duration": 0.0}
        threading.Thread(target=self._precompute_all, daemon=True).start()
        return {"ok": True, "use_crop": use_crop}

    def get_effective_crop(self, video_id: str) -> Dict[str, float]:
        return dict(
            self.video_crop_overrides.get(
                video_id,
                {"top": CROP_TOP, "bottom": CROP_BOTTOM, "left": CROP_LEFT, "right": CROP_RIGHT},
            )
        )

    def get_crop_overrides(self) -> Dict[str, Dict[str, float]]:
        return {vid: dict(rect) for vid, rect in self.video_crop_overrides.items()}

    def _video_meta_by_id(self, video_id: str) -> Optional[Dict[str, Any]]:
        for v in VIDEOS:
            if v["video_id"] == video_id:
                return v
        return None

    def set_video_crop(self, video_id: str, rect: Dict[str, Any]) -> Dict[str, Any]:
        video = self._video_meta_by_id(video_id)
        if not video:
            return {"ok": False, "error": f"Unknown video: {video_id}"}
        norm = self._normalize_crop_rect(rect)
        with self.lock:
            self.video_crop_overrides[video_id] = norm
            self._persist_crop_overrides()
            self.data[video_id] = {"frames": [], "status": "pending", "duration": 0.0}
        threading.Thread(target=self._analyze_video, args=(video,), daemon=True).start()
        return {"ok": True, "video_id": video_id, "crop": norm, "override": True}

    def clear_video_crop(self, video_id: str) -> Dict[str, Any]:
        video = self._video_meta_by_id(video_id)
        if not video:
            return {"ok": False, "error": f"Unknown video: {video_id}"}
        with self.lock:
            self.video_crop_overrides.pop(video_id, None)
            self._persist_crop_overrides()
            self.data[video_id] = {"frames": [], "status": "pending", "duration": 0.0}
        threading.Thread(target=self._analyze_video, args=(video,), daemon=True).start()
        return {"ok": True, "video_id": video_id, "crop": self.get_effective_crop(video_id), "override": False}

    def _crop_frame(self, frame: Any, rect: Dict[str, float]) -> Any:
        h, w = frame.shape[:2]
        y1 = int(h * rect["top"])
        y2 = int(h * rect["bottom"])
        x1 = int(w * rect["left"])
        x2 = int(w * rect["right"])
        return frame[y1:y2, x1:x2]

    def _preprocess(self, frame: Any) -> Any:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        arr = rgb.astype(np.float32) / 255.0
        mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
        std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
        arr = (arr - mean) / std
        arr = np.transpose(arr, (2, 0, 1))
        return np.expand_dims(arr, axis=0)

    @staticmethod
    def _sharpness_score(frame: Any) -> float:
        """Estimate frame sharpness via variance of Laplacian."""
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        return float(cv2.Laplacian(gray, cv2.CV_64F).var())

    def _read_sharpest_near_time(self, cap: Any, fps: float, t_sec: float) -> Any:
        """Read a small neighborhood around target time and keep sharpest frame.

        This reduces noise from motion-blurred frames in low-fps videos.
        """
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total <= 0 or fps <= 0:
            cap.set(cv2.CAP_PROP_POS_MSEC, t_sec * 1000)
            ok, frame = cap.read()
            return frame if ok else None

        target_idx = int(round(t_sec * fps))
        # Try a small +-2 frame window around target frame.
        offsets = (-2, -1, 0, 1, 2)
        best_frame = None
        best_score = -1.0
        for off in offsets:
            idx = max(0, min(total - 1, target_idx + off))
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok or frame is None:
                continue
            score = self._sharpness_score(frame)
            if score > best_score:
                best_score = score
                best_frame = frame
        return best_frame

    def _infer(self, frame: Any) -> Dict[str, float]:
        x = self._preprocess(frame)
        logits = self.model.run(None, {self.model_input_name: x})[0][0]
        logits = logits - np.max(logits)
        probs = np.exp(logits)
        probs = probs / probs.sum()
        for i, name in enumerate(CLASS_NAMES):
            if name in EXCLUDED_CLASSES:
                probs[i] = 0.0
        total = probs.sum()
        if total > 0:
            probs = probs / total
        return {
            CLASS_NAMES[i]: float(p)
            for i, p in enumerate(probs)
            if CLASS_NAMES[i] not in EXCLUDED_CLASSES
        }

    def _analyze_video(self, video: Dict[str, Any]) -> None:
        vid_id = video["video_id"]
        path = VIDEOS_DIR / video["filename"]

        with self.lock:
            self.data[vid_id]["status"] = "processing"

        if not (HAS_CV2 and HAS_NUMPY):
            with self.lock:
                self.data[vid_id]["status"] = "no_opencv"
            return

        if not path.exists():
            with self.lock:
                self.data[vid_id]["status"] = "file_not_found"
            return

        cap = cv2.VideoCapture(str(path))
        if not cap.isOpened():
            with self.lock:
                self.data[vid_id]["status"] = "cannot_open"
            return

        fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total / fps if fps > 0 else 0.0

        n_samples = min(60, max(10, int(duration * 2)))
        interval = duration / n_samples if n_samples > 0 else 1.0
        crop_rect = self.get_effective_crop(vid_id)

        frames_data: List[Dict[str, Any]] = []
        for i in range(n_samples):
            t_sec = i * interval
            frame = self._read_sharpest_near_time(cap, fps, t_sec)
            if frame is None:
                continue
            if self.use_crop:
                frame = self._crop_frame(frame, crop_rect)
            if self.model is not None and self.model_input_name is not None:
                try:
                    scores = self._infer(frame)
                except Exception:
                    scores = {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}
            else:
                scores = {c: 1.0 / len(CLASS_NAMES) for c in CLASS_NAMES}

            frames_data.append({"t": round(t_sec, 2), "scores": {k: round(v, 4) for k, v in scores.items()}})

        cap.release()

        with self.lock:
            self.data[vid_id]["frames"] = frames_data
            self.data[vid_id]["status"] = "done"
            self.data[vid_id]["duration"] = round(duration, 2)

    def _precompute_all(self) -> None:
        for video in VIDEOS:
            try:
                self._analyze_video(video)
            except Exception as exc:
                vid_id = video["video_id"]
                with self.lock:
                    self.data[vid_id]["status"] = f"error: {exc}"

    def get_avg_scores(self, video_id: str) -> Optional[Dict[str, float]]:
        with self.lock:
            frames = self.data.get(video_id, {}).get("frames", [])
        if not frames:
            return None
        avg: Dict[str, float] = {c: 0.0 for c in CLASS_NAMES}
        for frame in frames:
            for cls, p in frame["scores"].items():
                avg[cls] = avg.get(cls, 0.0) + p
        n = len(frames)
        return {c: v / n for c, v in avg.items()}

    def get_scores_matrix(self, video_id: str) -> Optional[Any]:
        if not HAS_NUMPY:
            return None
        with self.lock:
            frames = self.data.get(video_id, {}).get("frames", [])
        if not frames:
            return None
        return np.array(
            [[frame["scores"].get(cls, 0.0) for cls in CLASS_NAMES] for frame in frames],
            dtype=np.float32,
        )

    def get_duration(self, video_id: str) -> float:
        with self.lock:
            return float(self.data.get(video_id, {}).get("duration", 0.0) or 0.0)

    def status_summary(self) -> Dict[str, Any]:
        with self.lock:
            return {
                "model_status": self.model_status,
                "current_model": self.current_model_name,
                "use_crop": self.use_crop,
                "crop_overrides": self.get_crop_overrides(),
                "videos": {
                    vid: {"status": d["status"], "n_frames": len(d.get("frames", []))}
                    for vid, d in self.data.items()
                },
            }
