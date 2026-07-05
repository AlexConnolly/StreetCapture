"""YOLOv8-nano detection + ByteTrack tracking (via ultralytics' built-in tracker).

One ``model.track`` call does detection *and* persistent-ID assignment, which is
exactly the LIVE pipeline's Steps 2 + 3. Output is the flat track dict the rest
of the system consumes.
"""

from __future__ import annotations

from ultralytics import YOLO


class Detector:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = YOLO(cfg.model)          # auto-downloads yolov8n.pt on first run
        self.names = self.model.names
        self.device = cfg.device or None

    def track(self, frame):
        results = self.model.track(
            frame,
            persist=True,
            tracker="bytetrack.yaml",         # bundled with ultralytics
            conf=self.cfg.conf,
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
