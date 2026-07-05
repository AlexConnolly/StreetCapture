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

    # --- Storage ----------------------------------------------------------
    data_dir: Path = Path(_env("STREETCAPTURE_DATA_DIR", "data"))
    save_snapshots: bool = _env("STREETCAPTURE_SNAPSHOTS", "1") == "1"

    # --- Display ----------------------------------------------------------
    show_live: bool = _env("STREETCAPTURE_SHOW_LIVE", "1") == "1"
    show_dashboard: bool = _env("STREETCAPTURE_SHOW_DASHBOARD", "1") == "1"

    @property
    def cv_source(self):
        """OpenCV wants an int for webcams, a string for URLs / files."""
        return int(self.source) if str(self.source).isdigit() else self.source

    @property
    def headless(self) -> bool:
        return not (self.show_live or self.show_dashboard)

    def ensure_dirs(self) -> None:
        self.data_dir.mkdir(parents=True, exist_ok=True)
        (self.data_dir / "snapshots").mkdir(parents=True, exist_ok=True)
