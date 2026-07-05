"""Local, dependency-free data store: JSONL logs + snapshot crops.

    data/
      tracks.jsonl     one line per completed track record
      events.jsonl     one line per event (entered / left / stay)
      snapshots/       one JPG per track (first sighting)
"""

from __future__ import annotations

import json
import threading
from pathlib import Path

import cv2


class Store:
    def __init__(self, data_dir, save_snapshots: bool = True):
        self.data_dir = Path(data_dir)
        self.tracks_path = self.data_dir / "tracks.jsonl"
        self.events_path = self.data_dir / "events.jsonl"
        self.snap_dir = self.data_dir / "snapshots"
        self.save_snapshots = save_snapshots
        self._lock = threading.Lock()

    def _append(self, path: Path, obj: dict) -> None:
        line = json.dumps(obj)
        with self._lock:
            with open(path, "a", encoding="utf-8") as f:
                f.write(line + "\n")

    def write_track(self, record: dict) -> None:
        self._append(self.tracks_path, record)

    def write_event(self, event: dict) -> None:
        self._append(self.events_path, event)

    def save_snapshot(self, track_id: int, crop):
        if not self.save_snapshots or crop is None or crop.size == 0:
            return None
        path = self.snap_dir / f"track_{track_id:06d}.jpg"
        try:
            cv2.imwrite(str(path), crop)
        except Exception:
            return None
        return str(path)
