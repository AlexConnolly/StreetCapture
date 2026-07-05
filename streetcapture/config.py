"""Runtime configuration.

Every field can be overridden with an environment variable (see the ``_env``
defaults below) or a CLI flag in ``main.py``. Defaults are chosen so the system
runs against a local webcam / video file with zero setup; point ``source`` at
your Tapo RTSP URL for the real camera.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


def _env(name: str, default: str) -> str:
    return os.environ.get(name, default)


@dataclass
class Config:
    # --- Source -----------------------------------------------------------
    # RTSP URL, a webcam index ("0"), or a path to a video file.
    # Tapo C100 example: rtsp://user:pass@192.168.1.50:554/stream1
    source: str = _env("STREETCAPTURE_SOURCE", "0")

    # --- Detection --------------------------------------------------------
    model: str = _env("STREETCAPTURE_MODEL", "yolov8n.pt")   # nano only, per spec
    device: str = _env("STREETCAPTURE_DEVICE", "")           # "" auto, "cpu", "0" (gpu)
    conf: float = float(_env("STREETCAPTURE_CONF", "0.35"))

    # --- FPS caps ---------------------------------------------------------
    live_fps: float = float(_env("STREETCAPTURE_LIVE_FPS", "5"))
    artifact_fps: float = float(_env("STREETCAPTURE_ARTIFACT_FPS", "2"))

    # --- Track lifecycle --------------------------------------------------
    stay_seconds: float = float(_env("STREETCAPTURE_STAY_SECONDS", "8"))
    forget_seconds: float = float(_env("STREETCAPTURE_FORGET_SECONDS", "3"))
    max_positions: int = int(_env("STREETCAPTURE_MAX_POSITIONS", "300"))

    # --- Artifact gating (a completed track becomes an Artifact only if it
    #     clears ALL of these) --------------------------------------------
    artifact_min_duration: float = float(_env("STREETCAPTURE_ART_MIN_DURATION", "2.0"))
    artifact_min_confidence: float = float(_env("STREETCAPTURE_ART_MIN_CONF", "0.5"))
    artifact_min_area_frac: float = float(_env("STREETCAPTURE_ART_MIN_AREA", "0.004"))  # bbox area / frame area
    artifact_min_sharpness: float = float(_env("STREETCAPTURE_ART_MIN_SHARP", "25.0"))  # var-of-Laplacian
    artifact_min_visibility: float = float(_env("STREETCAPTURE_ART_MIN_VIS", "0.55"))   # 1.0 = fully inside frame
    artifact_edge_margin: int = int(_env("STREETCAPTURE_ART_EDGE_MARGIN", "3"))         # px, for visibility calc

    # --- Representative images -------------------------------------------
    rep_images_min: int = int(_env("STREETCAPTURE_REP_MIN", "3"))
    rep_images_max: int = int(_env("STREETCAPTURE_REP_MAX", "10"))
    sample_buffer: int = int(_env("STREETCAPTURE_SAMPLE_BUFFER", "24"))   # candidate crops kept per track
    crop_max_dim: int = int(_env("STREETCAPTURE_CROP_MAX_DIM", "256"))    # stored crop is downscaled to this

    # --- Embeddings ------------------------------------------------------
    embed_enabled: bool = _env("STREETCAPTURE_EMBED", "1") == "1"
    embed_model: str = _env("STREETCAPTURE_EMBED_MODEL", "ViT-B-32")
    embed_pretrained: str = _env("STREETCAPTURE_EMBED_PRETRAINED", "laion2b_s34b_b79k")

    # --- Storage ---------------------------------------------------------
    artifacts_dir: Path = Path(_env("STREETCAPTURE_ARTIFACTS_DIR", "artifacts"))

    # --- Display ---------------------------------------------------------
    show_live: bool = _env("STREETCAPTURE_SHOW_LIVE", "1") == "1"
    show_dashboard: bool = _env("STREETCAPTURE_SHOW_DASHBOARD", "1") == "1"

    @property
    def db_path(self) -> Path:
        return self.artifacts_dir / "artifact.db"

    @property
    def images_dir(self) -> Path:
        return self.artifacts_dir / "images"

    @property
    def faiss_path(self) -> Path:
        return self.artifacts_dir / "faiss.index"

    @property
    def cv_source(self):
        """OpenCV wants an int for webcams, a string for URLs / files."""
        return int(self.source) if str(self.source).isdigit() else self.source

    @property
    def headless(self) -> bool:
        return not (self.show_live or self.show_dashboard)

    def ensure_dirs(self) -> None:
        self.artifacts_dir.mkdir(parents=True, exist_ok=True)
        self.images_dir.mkdir(parents=True, exist_ok=True)
