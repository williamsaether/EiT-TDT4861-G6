from __future__ import annotations

# Normalized crop rectangle used by camera-model inference.
# Videos are normalized to 1000x500, so this focuses on the drivable lane area.
CROP_TOP: float = 0.60
CROP_BOTTOM: float = 0.95
CROP_LEFT: float = 0.35
CROP_RIGHT: float = 0.65
