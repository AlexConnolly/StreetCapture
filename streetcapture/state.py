"""Thread-safe hand-off between the live loop and the artifact loop.

The live loop publishes the newest frame + its tracks; the (slower) artifact
loop pulls whatever is latest at its own cadence. Detection therefore runs
once, not twice — the two "pipelines" share a single YOLO pass, which matters
on a 6GB GPU.
"""

from __future__ import annotations

import threading


class SharedState:
    def __init__(self):
        self._lock = threading.Lock()
        self._frame = None
        self._tracks = []
        self._frame_id = -1

    def publish(self, frame, tracks, frame_id) -> None:
        with self._lock:
            self._frame = frame
            self._tracks = tracks
            self._frame_id = frame_id

    def latest(self):
        """Return (frame, tracks_copy, frame_id). frame is None before first publish."""
        with self._lock:
            if self._frame is None:
                return None, [], self._frame_id
            return self._frame, list(self._tracks), self._frame_id
