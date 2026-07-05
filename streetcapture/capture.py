"""RTSP / webcam frame grabber.

Runs the blocking ``VideoCapture.read()`` on its own thread and keeps *only the
latest* frame. The live loop reads whatever is freshest, so a slow consumer
never builds up a backlog on the stream — this is the "drop frames if lagging"
requirement from the spec.
"""

from __future__ import annotations

import threading
import time

import cv2


class FrameGrabber:
    def __init__(self, source, reconnect_delay: float = 2.0):
        self.source = source
        self.reconnect_delay = reconnect_delay
        self._cap = None
        self._lock = threading.Lock()
        self._frame = None
        self._frame_id = 0
        self._running = False
        self._thread = None

    def start(self) -> "FrameGrabber":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="FrameGrabber", daemon=True)
        self._thread.start()
        return self

    def _open(self):
        cap = cv2.VideoCapture(self.source)
        # Best-effort: keep the backend buffer tiny so reads return fresh frames.
        try:
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
        except Exception:
            pass
        return cap

    def _loop(self) -> None:
        while self._running:
            if self._cap is None or not self._cap.isOpened():
                self._cap = self._open()
                if not self._cap.isOpened():
                    time.sleep(self.reconnect_delay)
                    continue
            ok, frame = self._cap.read()
            if not ok:
                # Stream hiccup or end of file — drop the capture and retry.
                self._cap.release()
                self._cap = None
                time.sleep(self.reconnect_delay)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self):
        """Return (frame_copy, frame_id). frame is None until the first read."""
        with self._lock:
            if self._frame is None:
                return None, self._frame_id
            return self._frame.copy(), self._frame_id

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2)
        if self._cap:
            self._cap.release()
