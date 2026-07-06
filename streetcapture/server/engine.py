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
from ..background import BackgroundModel
from ..capture import FrameGrabber
from ..dashboard import draw_live
from ..db import Database
from ..detector import Detector
from ..embeddings import Embedder
from ..reid import ReIDEmbedder
from ..state import SharedState
from ..vectorstore import VectorStore
from .groups import GroupService
from .recorder import Recorder


class PerceptionService:
    def __init__(self, cfg):
        self.cfg = cfg
        cfg.ensure_dirs()
        self.db = Database(cfg.db_path)
        self.session_id = self.db.start_session(cfg.source, cfg.model)
        self.state = SharedState()
        self.embedder = None
        self.reid = None
        self.vectorstore = None
        self.artifact = None
        self.groups = None
        self.recorder = None
        self.grabber = None
        self.detector = None
        self._jpeg = None          # annotated (boxes/labels drawn)
        self._raw_jpeg = None      # clean frame (overlay off)
        self._jpeg_id = 0
        self._fps = 0.0
        self._source_fps = 0.0     # rate of genuinely NEW frames from the camera
        self._stale_pct = 0.0      # % of loops that re-used the previous frame
        self._track_labels = {}    # track_id -> {"label", "score", "t"} taught-label match
        self.bg = BackgroundModel(cfg)   # idle/background-object suppression
        self._idle_count = 0
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        self._started_at = time.time()

    def start(self) -> "PerceptionService":
        self.grabber = FrameGrabber(self.cfg.cv_source).start()
        self.detector = Detector(self.cfg)
        self.embedder = Embedder(self.cfg)
        self.reid = ReIDEmbedder(self.cfg)
        self.vectorstore = VectorStore(self.cfg.faiss_path)
        self.groups = GroupService(self.cfg, self.db, self.embedder, self.vectorstore,
                                   reid=self.reid)
        self.groups.startup()
        self.artifact = ArtifactEngine(
            self.cfg, self.state, self.db, self.embedder, self.vectorstore, self.session_id,
            artifact_hook=self.groups.on_new_artifact, reid=self.reid,
        ).start()
        if self.cfg.record_enabled:
            self.recorder = Recorder(self.cfg).start()
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="LiveLoop", daemon=True)
        self._thread.start()
        return self

    def _loop(self) -> None:
        interval = 1.0 / max(self.cfg.live_fps, 0.1)
        enc = [int(cv2.IMWRITE_JPEG_QUALITY), self.cfg.jpeg_quality]
        fps_ema = None
        last_fid = -1
        # Rolling counts (reset each second) to derive real source fps + stale %.
        win_new = win_total = 0
        win_t0 = time.time()
        while self._running:
            t0 = time.time()
            frame, fid = self.grabber.read()
            if frame is None:
                time.sleep(0.05)
                continue
            # A repeated frame_id means the camera gave us no fresh frame this
            # loop (RTSP stall/packet loss). Don't re-run detection or re-send
            # the same JPEG — that just wastes GPU and pushes duplicate frames
            # down the MJPEG stream, which is what makes it feel choppy.
            is_new = fid != last_fid
            win_total += 1
            now = time.time()
            if now - win_t0 >= 1.0:
                with self._lock:
                    self._source_fps = round(win_new / (now - win_t0), 1)
                    self._stale_pct = round(100.0 * (win_total - win_new) / max(win_total, 1))
                win_new = win_total = 0
                win_t0 = now
            if not is_new:
                # Pace stale loops at ~the target interval (don't busy-poll), so
                # stale% reads as "% of target frames the camera didn't deliver"
                # rather than being inflated by fast re-polling.
                dt = time.time() - t0
                if dt < interval:
                    time.sleep(interval - dt)
                continue
            last_fid = fid
            win_new += 1

            tracks = self.detector.track(frame)
            # Drop idle/background objects (parked-forever car, potted plant) so
            # they neither draw a box nor generate artifacts.
            tracks, idle = self.bg.filter(tracks, now)
            self._idle_count = idle
            self.state.publish(frame, tracks, fid)
            self._label_live_tracks(frame, tracks)
            vis = draw_live(frame.copy(), tracks, fps_ema,
                            self.artifact.live_meta_snapshot(), self._track_labels)
            ok, buf = cv2.imencode(".jpg", vis, enc)
            ok2, raw = cv2.imencode(".jpg", frame, enc)
            if ok:
                with self._lock:
                    self._jpeg = buf.tobytes()
                    if ok2:
                        self._raw_jpeg = raw.tobytes()
                    self._jpeg_id += 1
                    self._fps = fps_ema or 0.0
            dt = time.time() - t0
            if dt < interval:
                time.sleep(interval - dt)
            inst = 1.0 / max(time.time() - t0, 1e-6)
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst

    def _label_live_tracks(self, frame, tracks) -> None:
        """Match live track crops to taught labels, rate-limited so the extra
        CLIP work is a few embeds/sec (each track re-checked every ~interval s)."""
        if not (self.cfg.live_label_enabled and self.groups and self.groups.has_labels()):
            return
        now = time.time()
        seen = set()
        done = 0
        for t in tracks:
            tid = t["track_id"]
            seen.add(tid)
            prev = self._track_labels.get(tid)
            if prev and now - prev["t"] < self.cfg.live_label_interval_s:
                continue
            if done >= self.cfg.live_label_budget:
                continue
            x1, y1, x2, y2 = (int(v) for v in t["bbox"])
            if x2 - x1 < 24 or y2 - y1 < 24:   # too small to embed usefully
                continue
            crop = frame[max(0, y1):y2, max(0, x1):x2]
            if crop.size == 0:
                continue
            m = self.groups.match_live(crop, t["class"])
            self._track_labels[tid] = {
                "label": m["label"] if m else None,
                "score": m["score"] if m else 0.0, "t": now}
            done += 1
        for tid in [k for k in self._track_labels if k not in seen]:
            del self._track_labels[tid]

    # -- accessors for the API --------------------------------------------
    def latest_jpeg(self, overlay: bool = True):
        with self._lock:
            jpeg = self._jpeg if overlay else (self._raw_jpeg or self._jpeg)
            return jpeg, self._jpeg_id

    def live_stats(self) -> dict:
        snap = self.artifact.dashboard_snapshot()
        with self._lock:
            fps = round(self._fps, 1)
            source_fps = self._source_fps
            stale_pct = self._stale_pct
            has_frame = self._jpeg is not None
        snap.update({
            "fps": fps,
            "source_fps": source_fps,   # real fresh-frame rate from the camera
            "stale_pct": stale_pct,     # % of loops that re-used a stale frame
            "idle_objects": self._idle_count,   # background objects being suppressed
            "online": has_frame,
            "uptime_s": int(time.time() - self._started_at),
            "faiss_vectors": self.vectorstore.ntotal if self.vectorstore else 0,
            "embed_model": self.embedder.model_version if self.embedder else "n/a",
            "dvr": self.recorder.stats() if self.recorder else {"recording": False},
        })
        return snap

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=5)
        if self.groups:
            self.groups.stop()
        if self.recorder:
            self.recorder.stop()
        if self.artifact:
            self.artifact.stop()
        if self.vectorstore:
            self.vectorstore.save()
        if self.grabber:
            self.grabber.stop()
        if self.db:
            self.db.close()
