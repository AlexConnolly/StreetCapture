"""ARTIFACT pipeline — the slow loop.

Runs at ~1-2 FPS off the shared state. For every active track it maintains a
persistent record (identity memory), samples a crop + a cheap embedding stub on
first sighting, and emits simple events (entered / stay / left). Completed
records are flushed to the JSONL store. It also keeps the live aggregates the
ARTIFACT VIEW dashboard renders.
"""

from __future__ import annotations

import threading
import time
from collections import Counter, defaultdict, deque

import cv2

VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "train"}


def category(cls_name: str) -> str:
    if cls_name == "person":
        return "person"
    if cls_name in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


def crop_embedding(crop):
    """v0.1 stub: an 8x8 grayscale downsample (64-dim). Cheap, no CLIP/FAISS.

    Real embeddings arrive in v0.2 — this just gives every track *some* visual
    fingerprint to store now.
    """
    if crop is None or crop.size == 0:
        return None
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    small = cv2.resize(gray, (8, 8))
    return [int(v) for v in small.flatten()]


class TrackRecord:
    __slots__ = (
        "track_id", "first_seen", "last_seen", "class_history",
        "positions", "frames_seen", "embedding", "snapshot", "stay_flagged",
    )

    def __init__(self, track_id, now, cls_name):
        self.track_id = track_id
        self.first_seen = now
        self.last_seen = now
        self.class_history = [cls_name]
        self.positions = []
        self.frames_seen = 0
        self.embedding = None
        self.snapshot = None
        self.stay_flagged = False

    @property
    def duration(self):
        return round(self.last_seen - self.first_seen, 2)

    def dominant_class(self):
        return Counter(self.class_history).most_common(1)[0][0]

    def to_dict(self):
        return {
            "track_id": self.track_id,
            "first_seen": round(self.first_seen, 2),
            "last_seen": round(self.last_seen, 2),
            "duration": self.duration,
            "class": self.dominant_class(),
            "class_history": self.class_history,
            "positions": self.positions,
            "frames_seen": self.frames_seen,
            "embedding": self.embedding,
            "snapshot": self.snapshot,
        }


class ArtifactEngine:
    def __init__(self, cfg, state, store):
        self.cfg = cfg
        self.state = state
        self.store = store
        self.records: dict[int, TrackRecord] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        # Dashboard aggregates (guarded by _lock).
        self.daily_counts = defaultdict(int)          # category -> unique tracks today
        self.recent_events = deque(maxlen=8)          # pre-formatted strings
        self._day = time.strftime("%Y-%m-%d")

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "ArtifactEngine":
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="ArtifactEngine", daemon=True)
        self._thread.start()
        return self

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=3)
        # Flush whatever is still live so no record is lost on shutdown.
        with self._lock:
            for rec in self.records.values():
                self.store.write_track(rec.to_dict())
            self.records.clear()

    def _loop(self) -> None:
        interval = 1.0 / max(self.cfg.artifact_fps, 0.1)
        while self._running:
            t0 = time.time()
            try:
                self._process()
            except Exception as e:  # never let the slow loop kill the thread
                print(f"[artifact] error: {e}")
            time.sleep(max(0.0, interval - (time.time() - t0)))

    # -- core --------------------------------------------------------------
    def _process(self) -> None:
        frame, tracks, _ = self.state.latest()
        if frame is None:
            return
        now = time.time()
        with self._lock:
            self._rollover(now)
            seen = set()
            for t in tracks:
                tid = t["track_id"]
                seen.add(tid)
                rec = self.records.get(tid)
                if rec is None:
                    rec = self._register(t, frame, now)
                rec.last_seen = now
                rec.frames_seen += 1
                rec.class_history.append(t["class"])
                x1, y1, x2, y2 = t["bbox"]
                rec.positions.append([round((x1 + x2) / 2, 1), round((y1 + y2) / 2, 1)])
                if len(rec.positions) > self.cfg.max_positions:
                    del rec.positions[: -self.cfg.max_positions]
                if not rec.stay_flagged and rec.duration >= self.cfg.stay_seconds:
                    rec.stay_flagged = True
                    self._emit({"type": "object_stay", "track_id": tid,
                                "class": rec.dominant_class(), "duration": rec.duration, "time": now})
            self._expire(seen, now)

    def _register(self, t, frame, now) -> TrackRecord:
        tid = t["track_id"]
        rec = TrackRecord(tid, now, t["class"])
        crop = self._crop(frame, t["bbox"])
        rec.embedding = crop_embedding(crop)
        rec.snapshot = self.store.save_snapshot(tid, crop)
        self.records[tid] = rec
        self.daily_counts[category(t["class"])] += 1
        self._emit({"type": "object_entered", "track_id": tid, "class": t["class"], "time": now})
        return rec

    def _expire(self, seen, now) -> None:
        gone = [
            tid for tid, rec in self.records.items()
            if tid not in seen and (now - rec.last_seen) > self.cfg.forget_seconds
        ]
        for tid in gone:
            rec = self.records.pop(tid)
            self.store.write_track(rec.to_dict())
            self._emit({"type": "object_left", "track_id": tid,
                        "class": rec.dominant_class(), "duration": rec.duration, "time": now})

    def _rollover(self, now) -> None:
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if day != self._day:
            self._day = day
            self.daily_counts.clear()

    @staticmethod
    def _crop(frame, bbox):
        h, w = frame.shape[:2]
        x1, y1, x2, y2 = bbox
        x1 = max(0, min(int(x1), w - 1))
        y1 = max(0, min(int(y1), h - 1))
        x2 = max(0, min(int(x2), w))
        y2 = max(0, min(int(y2), h))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def _emit(self, event) -> None:
        # Called with _lock held.
        self.store.write_event(event)
        ts = time.strftime("%H:%M:%S", time.localtime(event["time"]))
        et = event["type"]
        if et == "object_entered":
            s = f"{ts}  {event['class']} entered  (#{event['track_id']})"
        elif et == "object_left":
            s = f"{ts}  {event['class']} left {event['duration']:.0f}s  (#{event['track_id']})"
        elif et == "object_stay":
            s = f"{ts}  {event['class']} stayed {event['duration']:.0f}s  (#{event['track_id']})"
        else:
            s = f"{ts}  {et}"
        self.recent_events.appendleft(s)

    # -- dashboard read ----------------------------------------------------
    def dashboard_snapshot(self) -> dict:
        with self._lock:
            by_cat = defaultdict(int)
            for rec in self.records.values():
                by_cat[category(rec.dominant_class())] += 1
            return {
                "day": self._day,
                "active": len(self.records),
                "active_by_cat": dict(by_cat),
                "daily": dict(self.daily_counts),
                "events": list(self.recent_events),
            }
