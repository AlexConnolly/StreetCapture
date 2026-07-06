"""Person re-identification embeddings.

CLIP embeds "a person on a street" generically, so different pedestrians land at
~0.7 cosine and the entity layer over-merges them into blobs. This wraps a
dedicated ReID model (ultralytics' yolo26*-reid, auto-downloaded) which is
trained to separate people by appearance — different people sit at ~0.2 cosine.

Used ONLY for person entity resolution, on the async artifact thread (CPU, a few
crops/sec). Falls back to disabled (persons keep using CLIP) if unavailable.
"""

from __future__ import annotations

import threading

import numpy as np


class ReIDEmbedder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.enabled = False
        self._reid = None
        self._lock = threading.Lock()
        if not cfg.reid_enabled:
            return
        try:
            from ultralytics.trackers.utils.reid import ReID
            # device="cpu" avoids a noisy (failed) CUDA-EP probe; the model is
            # tiny and only runs on the low-rate artifact loop.
            self._reid = ReID(cfg.reid_model, device="cpu")
            self.enabled = True
            print(f"[reid] person ReID ready ({cfg.reid_model}, cpu)")
        except Exception as e:  # noqa: BLE001
            print(f"[reid] unavailable ({e}); persons fall back to CLIP entities")

    def embed(self, crop_bgr):
        """L2-normalised ReID vector for a person crop, or None."""
        if not self.enabled or crop_bgr is None or crop_bgr.size == 0:
            return None
        try:
            h, w = crop_bgr.shape[:2]
            det = np.array([[w / 2.0, h / 2.0, float(w), float(h)]], dtype=np.float32)
            with self._lock:
                feat = self._reid(crop_bgr, det)[0].astype("float32")
            n = np.linalg.norm(feat)
            if n < 1e-9:
                return None
            return [round(float(x), 6) for x in (feat / n)]
        except Exception as e:  # noqa: BLE001
            print(f"[reid] embed failed ({e})")
            return None
