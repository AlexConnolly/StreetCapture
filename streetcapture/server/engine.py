"""Headless perception service for the web server.

Runs the same pipeline as the desktop app (FrameGrabber -> Detector -> shared
state -> ArtifactEngine) but instead of OpenCV windows it keeps the latest
annotated JPEG in memory for MJPEG streaming and exposes live stats. The live
loop runs on its own thread so it never blocks the web server / event loop.
"""

from __future__ import annotations

import threading
import time

import cv2

from ..artifact import ArtifactEngine
from ..capture import FrameGrabber
from ..dashboard import draw_live
from ..db import Database
from ..detector import Detector
from ..embeddings import Embedder
from ..state import SharedState
from ..vectorstore import VectorStore


class PerceptionService:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.db = Database(cfg.db_path)
        self.session_id = self.db.start_session(cfg.source, cfg.model)
        self.state = SharedState()
        self.embedder = None
        self.vectorstore = None
        self.artifact = None
        self.grabber = None
        self.detector = None
        self._jpeg = None
        self._jpeg_id = 0
        self._fps = 0.0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._started_at = time.time()

    def start(self) -> "PerceptionService":
        self.grabber = FrameGrabber(self.cfg.cv_source).start()
        self.detector = Detector(self.cfg)
        self.embedder = Embedder(self.cfg)
        self.vectorstore = VectorStore(self.cfg.faiss_path)
        self.artifact = ArtifactEngine(
            self.cfg, self.state, self.db, self.embedder, self.vectorstore, self.session_id
        ).start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="LiveLoop", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        interval = 1.0 / max(self.cfg.live_fps, 0.1)
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality]
        fps_ema = None
        while self._running:
            t0 = time.time()
            frame, fid = self.grabber.read()
            if frame is None:
                time.sleep(0.05)
                continue
            tracks = self.detector.track(frame)
            self.state.publish(frame, tracks, fid)
            vis = draw_live(frame.copy(), tracks, fps_ema, self.artifact.live_meta_snapshot())
            ok, buf = cv2.imencode(".jpg", vis, enc)
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()
                    self._jpeg_id += 1
                    self._fps = fps_ema or 0.0
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)
            inst = 1.0 / max(time.time() - t0, 1e-6)
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

    # -- accessors for the API --------------------------------------------
    def latest_jpeg(self):
        with self._lock:
            return self._jpeg, self._jpeg_id

    def live_stats(self) -> dict:
        snap = self.artifact.dashboard_snapshot()
        with self._lock:
            fps = round(self._fps, 1)
            has_frame = self._jpeg is not None
        snap.update({
            "fps": fps,
            "online": has_frame,
            "uptime_s": int(time.time() - self._started_at),
            "faiss_vectors": self.vectorstore.ntotal if self.vectorstore else 0,
            "embed_model": self.embedder.model_version if self.embedder else "n/a",
        })
        return snap

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.artifact:
            self.artifact.stop()
        if self.vectorstore:
            self.vectorstore.save()
        if self.grabber:
            self.grabber.stop()
        if self.db:
            self.db.close()
