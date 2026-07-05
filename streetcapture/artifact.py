"""ARTIFACT pipeline — the async "memory" loop (v0.2).

Runs at ~1-2 FPS off the shared state. For every active track it accumulates
candidate crops + metrics. When a track *completes* it is evaluated: meaningful
tracks become **Artifacts** (representative images + quality scores + embedding,
persisted to SQLite); the rest are rejected with a reason. The live loop stays
untouched.

Track ID (a single observation) is kept strictly separate from the future
Entity ID (a persistent identity across tracks) — see db.py.
"""

from __future__ import annotations

import json
import threading
import time
from collections import Counter, defaultdict, deque

import cv2

from . import quality, taxonomy
from .taxonomy import category


class Sample:
    """One candidate crop for a track, with its quality metrics."""
    __slots__ = ("crop", "time", "sharpness", "bbox", "area_frac", "visibility", "conf", "w", "h")

    def __init__(self, crop, t, sharp, bbox, area_frac, vis, conf):
        self.crop = crop
        self.time = t
        self.sharpness = sharp
        self.bbox = bbox
        self.area_frac = area_frac
        self.visibility = vis
        self.conf = conf
        self.h, self.w = crop.shape[:2]

    def score(self) -> float:
        # Prefer sharp + large; visibility acts as a soft gate.
        return self.sharpness * (self.w * self.h) ** 0.25 * (0.4 + 0.6 * self.visibility)


class TrackAccumulator:
    __slots__ = ("track_id", "first_seen", "last_seen", "class_history",
                 "confidences", "positions", "samples", "frames_seen",
                 "started_flagged", "stay_flagged", "max_area_frac")

    def __init__(self, track_id, now):
        self.track_id = track_id
        self.first_seen = now
        self.last_seen = now
        self.class_history = []
        self.confidences = []
        self.positions = []
        self.samples: list[Sample] = []
        self.frames_seen = 0
        self.started_flagged = False
        self.stay_flagged = False
        self.max_area_frac = 0.0

    @property
    def duration(self):
        return round(self.last_seen - self.first_seen, 2)

    def dominant_class(self):
        return Counter(self.class_history).most_common(1)[0][0]


class ArtifactEngine:
    def __init__(self, cfg, state, db, embedder, vectorstore, session_id):
        self.cfg = cfg
        self.state = state
        self.db = db
        self.embedder = embedder
        self.vectorstore = vectorstore
        self.session_id = session_id
        self.tracks: dict[int, TrackAccumulator] = {}
        self._lock = threading.Lock()
        self._running = False
        self._thread = None
        # Live dashboard aggregates + per-track meta for the LIVE overlay.
        self.daily_counts = defaultdict(int)
        self.artifact_counts = defaultdict(int)
        self.recent_events = deque(maxlen=10)
        self.live_meta: dict[int, dict] = {}
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
            self._thread.join(timeout=5)
        # Finalise every still-open track so nothing is lost on shutdown.
        with self._lock:
            for acc in list(self.tracks.values()):
                self._finalise(acc, time.time())
            self.tracks.clear()

    def _loop(self) -> None:
        interval = 1.0 / max(self.cfg.artifact_fps, 0.1)
        while self._running:
            t0 = time.time()
            try:
                self._process()
            except Exception as e:
                print(f"[artifact] error: {e}")
            time.sleep(max(0.0, interval - (time.time() - t0)))

    # -- per-tick sampling -------------------------------------------------
    def _process(self) -> None:
        frame, tracks, _ = self.state.latest()
        if frame is None:
            return
        now = time.time()
        fh, fw = frame.shape[:2]
        with self._lock:
            self._rollover(now)
            seen = set()
            for t in tracks:
                tid = t["track_id"]
                seen.add(tid)
                acc = self.tracks.get(tid)
                if acc is None:
                    acc = TrackAccumulator(tid, now)
                    self.tracks[tid] = acc
                    acc.started_flagged = True
                    self._emit({"type": "track_started", "source_track_id": tid,
                                "class": t["class"], "time": now})
                acc.last_seen = now
                acc.frames_seen += 1
                acc.class_history.append(t["class"])
                acc.confidences.append(t["confidence"])
                bbox = t["bbox"]
                cx, cy = (bbox[0] + bbox[2]) / 2, (bbox[1] + bbox[3]) / 2
                acc.positions.append([round(cx, 1), round(cy, 1)])
                if len(acc.positions) > self.cfg.max_positions:
                    del acc.positions[: -self.cfg.max_positions]
                acc.max_area_frac = max(acc.max_area_frac, quality.area_frac(bbox, fw, fh))
                self._sample(acc, frame, bbox, fw, fh, now, t["confidence"])
            self._expire(seen, now)
            self._refresh_live_meta(now)

    def _sample(self, acc, frame, bbox, fw, fh, now, conf) -> None:
        crop = self._crop(frame, bbox)
        if crop is None:
            return
        crop = self._downscale(crop)
        s = Sample(
            crop=crop, t=now, sharp=quality.sharpness(crop), bbox=bbox,
            area_frac=quality.area_frac(bbox, fw, fh),
            vis=quality.visibility(bbox, fw, fh, self.cfg.artifact_edge_margin), conf=conf,
        )
        acc.samples.append(s)
        if len(acc.samples) > self.cfg.sample_buffer:
            # Keep the best candidates (drop the weakest by score).
            worst = min(range(len(acc.samples)), key=lambda i: acc.samples[i].score())
            acc.samples.pop(worst)

    def _expire(self, seen, now) -> None:
        gone = [
            tid for tid, acc in self.tracks.items()
            if tid not in seen and (now - acc.last_seen) > self.cfg.forget_seconds
        ]
        for tid in gone:
            acc = self.tracks.pop(tid)
            self.live_meta.pop(tid, None)
            self._finalise(acc, now)

    # -- track completion --------------------------------------------------
    def _finalise(self, acc, now) -> None:
        cls = acc.dominant_class() if acc.class_history else "unknown"
        avg_conf = sum(acc.confidences) / len(acc.confidences) if acc.confidences else 0.0
        track_pk = self.db.insert_track({
            "session_id": self.session_id,
            "source_track_id": acc.track_id,
            "primary_class": cls,
            "first_seen": round(acc.first_seen, 2),
            "last_seen": round(acc.last_seen, 2),
            "duration": acc.duration,
            "frames_seen": acc.frames_seen,
            "created_at": now,
        })
        self._emit({"type": "track_ended", "source_track_id": acc.track_id,
                    "class": cls, "duration": acc.duration, "time": now})

        ok, reason = self._evaluate(acc)
        if not ok:
            self._emit({"type": "artifact_rejected", "source_track_id": acc.track_id,
                        "class": cls, "reason": reason, "time": now})
            return
        self._create_artifact(acc, track_pk, cls, avg_conf, now)

    def _evaluate(self, acc) -> tuple[bool, str]:
        if acc.frames_seen < 2:
            return False, "single-frame"
        if acc.duration < self.cfg.artifact_min_duration:
            return False, f"too-brief({acc.duration:.1f}s)"
        if not acc.samples:
            return False, "no-samples"
        peak_conf = max(acc.confidences) if acc.confidences else 0.0
        if peak_conf < self.cfg.artifact_min_confidence:
            return False, f"low-confidence({peak_conf:.2f})"
        if acc.max_area_frac < self.cfg.artifact_min_area_frac:
            return False, "too-small"
        best_sharp = max(s.sharpness for s in acc.samples)
        if best_sharp < self.cfg.artifact_min_sharpness:
            return False, f"blurry({best_sharp:.0f})"
        best_vis = max(s.visibility for s in acc.samples)
        if best_vis < self.cfg.artifact_min_visibility:
            return False, f"edge-cut({best_vis:.2f})"
        return True, "ok"

    def _create_artifact(self, acc, track_pk, cls, avg_conf, now) -> None:
        reps = self._select_representatives(acc.samples)
        rep_bbox = reps[0].bbox if reps else (acc.samples[-1].bbox if acc.samples else [0, 0, 0, 0])
        artifact_id = self.db.insert_artifact({
            "track_pk": track_pk,
            "session_id": self.session_id,
            "source_track_id": acc.track_id,
            "primary_class": cls,
            "start_time": round(acc.first_seen, 2),
            "end_time": round(acc.last_seen, 2),
            "duration": acc.duration,
            "avg_confidence": round(avg_conf, 3),
            "sharpness": round(max((s.sharpness for s in reps), default=0.0), 1),
            "visibility": round(max((s.visibility for s in reps), default=0.0), 3),
            "motion_distance": round(quality.motion_distance(acc.positions), 1),
            "track_length": acc.frames_seen,
            "bbox_json": json.dumps([round(v, 1) for v in rep_bbox]),
            "motion_path_json": json.dumps(acc.positions[-self.cfg.max_positions:]),
            "entity_id": None,  # reserved for v0.3+
            "created_at": now,
        })

        # Save representative crops.
        for rank, s in enumerate(reps):
            path = self.cfg.images_dir / f"{artifact_id:06d}_{rank}.jpg"
            try:
                cv2.imwrite(str(path), s.crop)
            except Exception:
                continue
            self.db.insert_image({
                "artifact_id": artifact_id,
                "path": str(path),
                "frame_time": round(s.time, 2),
                "sharpness": round(s.sharpness, 1),
                "width": s.w,
                "height": s.h,
                "rank": rank,
            })

        # Multi-label taxonomy (object/subtype/function; company/energy are v2).
        for lab in taxonomy.labels_for(cls):
            self.db.insert_label(artifact_id, lab["type"], lab["value"])

        # Embedding from the single best representative -> DB + FAISS index.
        if reps and self.cfg.embed_enabled:
            vec = self.embedder.embed(reps[0].crop)
            if vec:
                self.db.insert_embedding(artifact_id, vec, self.embedder.model_version)
                if self.vectorstore is not None:
                    self.vectorstore.add(artifact_id, vec)

        self.artifact_counts[category(cls)] += 1
        self.daily_counts[category(cls)] += 1
        self._emit({"type": "artifact_created", "source_track_id": acc.track_id,
                    "artifact_id": artifact_id, "class": cls,
                    "duration": acc.duration, "time": now})
        self._product_events(acc, artifact_id, cls, now)

    def _product_events(self, acc, artifact_id, cls, now) -> None:
        """Structured events derived from a completed artifact (section 7)."""
        self._emit({"type": "object_entered", "artifact_id": artifact_id,
                    "source_track_id": acc.track_id, "class": cls, "time": acc.first_seen})
        self._emit({"type": "object_left", "artifact_id": artifact_id,
                    "source_track_id": acc.track_id, "class": cls,
                    "duration": acc.duration, "time": acc.last_seen})
        if acc.duration >= self.cfg.stay_seconds:
            self._emit({"type": "object_stayed", "artifact_id": artifact_id,
                        "source_track_id": acc.track_id, "class": cls,
                        "duration": acc.duration, "time": now})
        if category(cls) == "vehicle":
            self._emit({"type": "vehicle_passed", "artifact_id": artifact_id,
                        "source_track_id": acc.track_id, "class": cls,
                        "duration": acc.duration, "time": now})

    def _select_representatives(self, samples):
        """Pick rep_min..rep_max crops: prefer visible ones, spread over the
        track's lifetime, best-scoring within each time bucket."""
        if not samples:
            return []
        good = [s for s in samples if s.visibility >= self.cfg.artifact_min_visibility]
        pool = good if len(good) >= self.cfg.rep_images_min else samples
        pool = sorted(pool, key=lambda s: s.time)
        k = max(self.cfg.rep_images_min, min(self.cfg.rep_images_max, len(pool)))
        if len(pool) <= k:
            return pool
        # Evenly spaced time buckets; best-scoring sample per bucket.
        reps = []
        n = len(pool)
        for i in range(k):
            lo = i * n // k
            hi = max(lo + 1, (i + 1) * n // k)
            reps.append(max(pool[lo:hi], key=lambda s: s.score()))
        return reps

    # -- helpers -----------------------------------------------------------
    def _downscale(self, crop):
        h, w = crop.shape[:2]
        m = max(h, w)
        if m <= self.cfg.crop_max_dim:
            return crop
        scale = self.cfg.crop_max_dim / m
        return cv2.resize(crop, (int(w * scale), int(h * scale)))

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

    def _rollover(self, now) -> None:
        day = time.strftime("%Y-%m-%d", time.localtime(now))
        if day != self._day:
            self._day = day
            self.daily_counts.clear()
            self.artifact_counts.clear()

    def _emit(self, event) -> None:
        # Called with _lock held.
        self.db.insert_event({
            "session_id": self.session_id,
            "type": event["type"],
            "source_track_id": event.get("source_track_id"),
            "artifact_id": event.get("artifact_id"),
            "class": event.get("class"),
            "duration": event.get("duration"),
            "reason": event.get("reason"),
            "time": event["time"],
        })
        # object_entered/left duplicate the track lifecycle — keep them in the DB
        # but out of the small on-screen feed to avoid flooding it.
        if event["type"] in ("object_entered", "object_left"):
            return
        ts = time.strftime("%H:%M:%S", time.localtime(event["time"]))
        et = event["type"]
        cls = event.get("class", "")
        tid = event.get("source_track_id")
        if et == "artifact_created":
            s = f"{ts}  ARTIFACT #{event['artifact_id']}  {cls} {event.get('duration', 0):.0f}s"
        elif et == "artifact_rejected":
            s = f"{ts}  rejected {cls} (#{tid}) — {event.get('reason', '')}"
        elif et == "vehicle_passed":
            s = f"{ts}  vehicle passed — {cls} (#{tid})"
        elif et == "object_stayed":
            s = f"{ts}  {cls} stayed {event.get('duration', 0):.0f}s (#{tid})"
        elif et == "track_started":
            s = f"{ts}  track #{tid} started ({cls})"
        elif et == "track_ended":
            s = f"{ts}  track #{tid} ended {event.get('duration', 0):.0f}s"
        else:
            s = f"{ts}  {et}"
        self.recent_events.appendleft(s)

    def _refresh_live_meta(self, now) -> None:
        meta = {}
        for tid, acc in self.tracks.items():
            meta[tid] = {
                "age": round(now - acc.first_seen, 1),
                "pending": acc.duration >= self.cfg.artifact_min_duration,
                "samples": len(acc.samples),
            }
        self.live_meta = meta

    # -- reads for the UI --------------------------------------------------
    def live_meta_snapshot(self) -> dict:
        with self._lock:
            return dict(self.live_meta)

    def dashboard_snapshot(self) -> dict:
        with self._lock:
            by_cat = defaultdict(int)
            for acc in self.tracks.values():
                if acc.class_history:
                    by_cat[category(acc.dominant_class())] += 1
            return {
                "day": self._day,
                "active": len(self.tracks),
                "active_by_cat": dict(by_cat),
                "daily": dict(self.daily_counts),
                "artifacts": dict(self.artifact_counts),
                "events": list(self.recent_events),
            }
