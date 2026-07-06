"""YOLOv8-nano detection + ByteTrack tracking (via ultralytics' built-in tracker).

One ``model.track`` call does detection *and* persistent-ID assignment, which is
exactly the LIVE pipeline's Steps 2 + 3. Output is the flat track dict the rest
of the system consumes.
"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

# Tracker config templates. track_buffer (occlusion tolerance) is parameterised;
# the rest are ultralytics' shipped defaults. GMC is disabled (static camera).
_BYTETRACK_TEMPLATE = """\
tracker_type: bytetrack
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {track_buffer}
match_thresh: 0.8
fuse_score: True
"""

_BOTSORT_TEMPLATE = """\
tracker_type: botsort
track_high_thresh: 0.25
track_low_thresh: 0.1
new_track_thresh: 0.25
track_buffer: {track_buffer}
match_thresh: 0.8
fuse_score: True
gmc_method: None
proximity_thresh: 0.5
appearance_thresh: 0.25
with_reid: {with_reid}
model: auto
"""


def _write_tracker_cfg(cfg) -> str:
    """Materialise the tracker yaml (bytetrack or botsort) and return its path."""
    path = Path(cfg.tracker_cfg_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    if str(cfg.tracker).lower() == "botsort":
        text = _BOTSORT_TEMPLATE.format(
            track_buffer=int(cfg.track_buffer),
            with_reid="True" if cfg.track_reid else "False")
    else:
        text = _BYTETRACK_TEMPLATE.format(track_buffer=int(cfg.track_buffer))
    path.write_text(text)
    return str(path)


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg.model)          # auto-downloads yolov8n.pt on first run
        self.names = self.model.names
        self.device = cfg.device or None
        self.tracker_cfg = _write_tracker_cfg(cfg)

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker=self.tracker_cfg,         # generated (track_buffer applied)
            conf=self.cfg.conf,
            imgsz=self.cfg.imgsz,
            device=self.device,
            verbose=False,
        )
        out = []
        if not results:
            return out
        boxes = results[0].boxes
        # boxes.id is None until ByteTrack has established at least one track.
        if boxes is None or boxes.id is None:
            return out
        ids = boxes.id.int().tolist()
        clss = boxes.cls.int().tolist()
        confs = boxes.conf.tolist()
        xyxy = boxes.xyxy.tolist()
        for tid, c, cf, box in zip(ids, clss, confs, xyxy):
            out.append(
                {
                    "track_id": int(tid),
                    "class": self.names[int(c)],
                    "confidence": round(float(cf), 3),
                    "bbox": [round(float(v), 1) for v in box],  # x1, y1, x2, y2
                }
            )
        return out
