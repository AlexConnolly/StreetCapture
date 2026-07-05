"""Per-crop quality metrics used to gate and rank artifacts."""

from __future__ import annotations

import cv2
import numpy as np


def sharpness(crop) -> float:
    """Variance of the Laplacian — higher = sharper / more in focus."""
    if crop is None or crop.size == 0:
        return 0.0
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    return float(cv2.Laplacian(gray, cv2.CV_64F).var())


def visibility(bbox, frame_w, frame_h, margin: int = 3) -> float:
    """Fraction of the bbox that sits inside the frame (1.0 = fully visible).

    An object clipped by the frame edge scores below 1.0; we treat the bbox as
    its detector-reported extent and measure how much of that area is actually
    on-screen.
    """
    x1, y1, x2, y2 = bbox
    full = max(1.0, (x2 - x1) * (y2 - y1))
    ix1 = max(x1, margin)
    iy1 = max(y1, margin)
    ix2 = min(x2, frame_w - margin)
    iy2 = min(y2, frame_h - margin)
    inside = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    return float(min(1.0, inside / full))


def area_frac(bbox, frame_w, frame_h) -> float:
    x1, y1, x2, y2 = bbox
    return float(max(0.0, (x2 - x1) * (y2 - y1)) / max(1.0, frame_w * frame_h))


def motion_distance(positions) -> float:
    """Total path length travelled by the object centroid (pixels)."""
    if not positions or len(positions) < 2:
        return 0.0
    pts = np.asarray(positions, dtype=np.float32)
    return float(np.linalg.norm(np.diff(pts, axis=0), axis=1).sum())
