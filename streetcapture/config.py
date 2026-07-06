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
    # yolov8s at imgsz 960 runs ~45 FPS on an RTX 2070 — far above the ~18 FPS
    # the camera delivers — while catching smaller/distant objects that the
    # nano model at 640 misses. See the benchmark notes in the README.
    model: str = _env("STREETCAPTURE_MODEL", "yolov8s.pt")
    imgsz: int = int(_env("STREETCAPTURE_IMGSZ", "960"))     # detector input size
    device: str = _env("STREETCAPTURE_DEVICE", "")           # "" auto, "cpu", "0" (gpu)
    conf: float = float(_env("STREETCAPTURE_CONF", "0.35"))

    # --- FPS caps ---------------------------------------------------------
    live_fps: float = float(_env("STREETCAPTURE_LIVE_FPS", "15"))
    artifact_fps: float = float(_env("STREETCAPTURE_ARTIFACT_FPS", "2"))

    # --- Track lifecycle --------------------------------------------------
    stay_seconds: float = float(_env("STREETCAPTURE_STAY_SECONDS", "8"))
    # Finalise a track into an artifact only after it's been gone this long. Must
    # be >= the tracker's revive window (track_buffer/fps ~5-7s), otherwise a
    # brief detection dropout finalises artifact #1 and the SAME object (same
    # track id) revives as artifact #2 — i.e. one car in frame counted twice.
    forget_seconds: float = float(_env("STREETCAPTURE_FORGET_SECONDS", "7"))
    max_positions: int = int(_env("STREETCAPTURE_MAX_POSITIONS", "300"))
    # Two sightings of the SAME entity closer than this are one "visit" (collapses
    # any residual fragmentation from track-id changes at the display level).
    visit_gap_seconds: float = float(_env("STREETCAPTURE_VISIT_GAP", "120"))

    # --- Background / idle-object suppression -----------------------------
    # An object that sits motionless in the same spot for > background_seconds
    # (a potted plant, a permanently-parked car) becomes scene BACKGROUND: its
    # box is hidden and it stops generating artifacts. ANY motion resets the
    # timer, so a car is shown while driving in + for background_seconds after it
    # parks (arrival captured), then goes quiet, and is flagged again when it
    # leaves. A location cleared for background_forget_seconds resets, so a fresh
    # arrival there is new again.
    background_suppress: bool = _env("STREETCAPTURE_BG_SUPPRESS", "1") == "1"
    background_seconds: float = float(_env("STREETCAPTURE_BG_SECONDS", "300"))     # motionless -> background
    background_forget_seconds: float = float(_env("STREETCAPTURE_BG_FORGET", "60"))  # cleared -> reset
    # ByteTrack keeps a lost track alive for this many frames before retiring its
    # ID. Default ultralytics value is 30 (~2s @15fps); we raise it so someone
    # who ducks behind a billboard/pole for a few seconds keeps the same ID.
    track_buffer: int = int(_env("STREETCAPTURE_TRACK_BUFFER", "75"))  # ~5s @15fps
    # Tracker: 'bytetrack' (motion only, fast) or 'botsort' (adds appearance
    # ReID so an object keeps its ID across a jump/gap — the fix for IDs churning
    # when the stream stalls). Camera is static so GMC is disabled either way.
    tracker: str = _env("STREETCAPTURE_TRACKER", "botsort")
    track_reid: bool = _env("STREETCAPTURE_TRACK_REID", "1") == "1"

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

    # --- DVR continuous recording (24h scrub-back) -----------------------
    # A separate ffmpeg process records the RTSP stream into short mp4 segments
    # so the web UI can scrub back through the last N hours. `-c copy` by default
    # (no re-encode: lightest on CPU, exact camera quality). Point record_source
    # at the camera's sub-stream (e.g. .../stream2) to save disk if needed.
    record_enabled: bool = _env("STREETCAPTURE_RECORD", "1") == "1"
    record_source: str = _env("STREETCAPTURE_RECORD_SOURCE", "")   # blank -> use `source`
    record_segment_s: int = int(_env("STREETCAPTURE_RECORD_SEGMENT", "120"))   # mp4 chunk length
    record_retention_h: float = float(_env("STREETCAPTURE_RECORD_RETENTION_H", "24"))
    record_scale: str = _env("STREETCAPTURE_RECORD_SCALE", "")     # e.g. "1280:-2"; blank -> copy
    record_crf: int = int(_env("STREETCAPTURE_RECORD_CRF", "23"))  # only used when re-encoding

    # --- Movement scrobbler ----------------------------------------------
    # Per-frame, a track only counts as "moving" if its centroid shifts more
    # than this fraction of its own bbox diagonal (kills box jitter on parked
    # cars / standing people). Movement energy is summed per minute for the
    # timeline so the spikes reflect motion, not static presence.
    movement_deadzone_frac: float = float(_env("STREETCAPTURE_MOVE_DEADZONE", "0.06"))

    # --- Display ---------------------------------------------------------
    show_live: bool = _env("STREETCAPTURE_SHOW_LIVE", "1") == "1"
    show_dashboard: bool = _env("STREETCAPTURE_SHOW_DASHBOARD", "1") == "1"

    # --- v2: groups / entities / notifications ---------------------------
    cluster_distance: float = float(_env("STREETCAPTURE_CLUSTER_DIST", "0.28"))  # cosine dist, complete linkage
    cluster_min_size: int = int(_env("STREETCAPTURE_CLUSTER_MIN", "5"))

    # --- Live labels: tag tracks on the live video with taught group labels --
    # Rate-limited so it barely touches the GPU: each track's crop is CLIP-matched
    # against your labeled prototypes at most every `interval` seconds, capped at
    # `budget` embeds per frame.
    live_label_enabled: bool = _env("STREETCAPTURE_LIVE_LABELS", "1") == "1"
    live_label_threshold: float = float(_env("STREETCAPTURE_LIVE_LABEL_MATCH", "0.78"))
    live_label_interval_s: float = float(_env("STREETCAPTURE_LIVE_LABEL_INTERVAL", "1.5"))
    live_label_budget: int = int(_env("STREETCAPTURE_LIVE_LABEL_BUDGET", "1"))  # embeds/frame
    group_match_threshold: float = float(_env("STREETCAPTURE_GROUP_MATCH", "0.72"))  # image cosine to auto-tag
    # Once a group has this many HUMAN-confirmed examples it is trusted to
    # auto-classify: any artifact clearing its match threshold is confirmed
    # automatically instead of queued for you to verify. Auto-classified items
    # (source='auto_confirm') do NOT feed centroid training, so the model stays
    # anchored to your hand-labelled examples.
    auto_classify_min_confirmed: int = int(_env("STREETCAPTURE_AUTOCLASSIFY_MIN", "10"))
    # Stricter bar when retro-applying a freshly-taught region label to existing
    # artifacts — a single drawn crop is a weak prototype, so keep it precise
    # (common things like 'a person' look alike in CLIP and would otherwise flood).
    label_match_threshold: float = float(_env("STREETCAPTURE_LABEL_MATCH", "0.82"))
    text_match_threshold: float = float(_env("STREETCAPTURE_TEXT_MATCH", "0.22"))    # text->image cosine
    entity_threshold: float = float(_env("STREETCAPTURE_ENTITY_MATCH", "0.83"))      # same-instance cosine (CLIP, non-person)

    # --- Person re-identification ----------------------------------------
    # CLIP can't tell pedestrians apart (all embed as "a person"), so person
    # ENTITIES use a dedicated ReID model instead — trained to separate people
    # by build/clothing. Runs on the async artifact loop (CPU ~30ms/crop), never
    # the live loop. Different-people cosine ~0.2, so a ~0.5 cut separates them.
    reid_enabled: bool = _env("STREETCAPTURE_REID", "1") == "1"
    reid_model: str = _env("STREETCAPTURE_REID_MODEL", "yolo26s-reid.onnx")
    # 0.65 favours PURITY: different cars/people stay separate (no "one car seen
    # 200×" blobs). Same instance may split into a couple of entities — the
    # acceptable direction. Lower = merges more (risks blobs); higher = fragments.
    reid_entity_threshold: float = float(_env("STREETCAPTURE_REID_ENTITY_MATCH", "0.65"))
    # Entity rebuild uses COMPLETE-LINKAGE agglomerative clustering, not a running
    # centroid — an item joins a cluster only if it's within this cosine distance
    # of EVERY member, killing the "generic average" snowball that merged
    # visibly-different people. 0.45 => members must be >0.55 cosine to each other.
    entity_cluster_distance: float = float(_env("STREETCAPTURE_ENTITY_CLUSTER_DIST", "0.45"))
    notify_cooldown_s: float = float(_env("STREETCAPTURE_NOTIFY_COOLDOWN", "300"))
    ntfy_server: str = _env("STREETCAPTURE_NTFY_SERVER", "https://ntfy.sh")
    ntfy_topic: str = _env("STREETCAPTURE_NTFY_TOPIC", "")   # blank = notifications disabled

    # --- Web server ------------------------------------------------------
    web_host: str = _env("STREETCAPTURE_WEB_HOST", "0.0.0.0")
    web_port: int = int(_env("STREETCAPTURE_WEB_PORT", "8000"))
    web_password: str = _env("STREETCAPTURE_PASSWORD", "streetcapture")
    web_secret: str = _env("STREETCAPTURE_SECRET", "")   # blank -> derived from password
    jpeg_quality: int = int(_env("STREETCAPTURE_JPEG_QUALITY", "70"))

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
    def recordings_dir(self) -> Path:
        return self.artifacts_dir / "recordings"

    @property
    def library_dir(self) -> Path:
        """Saved clips — pulled out of the 24h prune cycle, kept forever."""
        return self.artifacts_dir / "library"

    @property
    def tracker_cfg_path(self) -> Path:
        """Generated ByteTrack yaml (track_buffer applied)."""
        return self.artifacts_dir / "bytetrack.yaml"

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
        if self.record_enabled:
            self.recordings_dir.mkdir(parents=True, exist_ok=True)
            self.library_dir.mkdir(parents=True, exist_ok=True)
