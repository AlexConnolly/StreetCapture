"""FastAPI application: auth + live MJPEG + data/query API + SPA hosting.

One process serves everything: the perception pipeline runs in a background
thread (engine.py), the REST API and MJPEG stream sit on top, and the built
React SPA (web/dist) is served from the same origin. nginx (deploy/nginx.conf)
is only needed for TLS / production; for a POC point ngrok straight at this port.
"""

from __future__ import annotations

import time
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import Depends, FastAPI, HTTPException, Request, Query, Response, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

import cv2
import numpy as np
from ..db import unpack_vector

def _norm(v):
    v = np.asarray(v, dtype="float32")
    n = np.linalg.norm(v)
    return v / n if n else v

from ..config import Config
from ..query import QueryEngine
from . import auth
from .engine import PerceptionService
from .recorder import parse_segment_start

DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"


class LoginBody(BaseModel):
    password: str


class TextGroupBody(BaseModel):
    name: str
    prompt: str


class ArtifactGroupBody(BaseModel):
    name: str
    artifact_id: int


class NameBody(BaseModel):
    name: str


class NotifyBody(BaseModel):
    notify: bool


class MemberBody(BaseModel):
    status: str   # 'confirmed' | 'rejected' | 'removed'


class BatchMemberFeedbackBody(BaseModel):
    artifact_ids: list[int]
    status: str   # 'confirmed' | 'rejected'


class SaveClipBody(BaseModel):
    start: float
    end: float
    name: str


class LabelRegionBody(BaseModel):
    box: list[float]     # normalised [x1, y1, x2, y2]
    label: str
    rank: int = 0


class TagInfo(BaseModel):
    key: str
    value: str


class TagArtifactsBody(BaseModel):
    artifact_ids: list[int]
    tags: list[TagInfo]
    source_group_id: int | None = None


def create_app(cfg: Config | None = None) -> FastAPI:
    cfg = cfg or Config()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.service = PerceptionService(cfg).start()
        print(f"[server] pipeline started (source={cfg.source})")
        try:
            yield
        finally:
            app.state.service.stop()
            print("[server] pipeline stopped")

    app = FastAPI(title="StreetCapture", lifespan=lifespan)
    app.add_middleware(
        CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"],
    )
    guard = Depends(auth.require_auth(cfg))

    def _artifact_json(a: dict) -> dict:
        return {
            "id": a["id"],
            "class": a["primary_class"],
            "track_id": a["source_track_id"],
            "entity_id": a["entity_id"],
            "start": a["start_time"],
            "end": a["end_time"],
            "duration": a["duration"],
            "confidence": a["avg_confidence"],
            "sharpness": a["sharpness"],
            "visibility": a["visibility"],
            "motion": a["motion_distance"],
            "direction": a.get("direction"),
            "dir_x": a.get("dir_x"),
            "dir_y": a.get("dir_y"),
            "frames": a["track_length"],
            "labels": a.get("labels", []),
            "embedding": a.get("embedding"),
            "images": [
                {"url": f"/api/media/{a['id']}/{im['rank']}", "rank": im["rank"],
                 "w": im["width"], "h": im["height"], "sharpness": im["sharpness"]}
                for im in a.get("images", [])
            ],
        }

    # -- auth --------------------------------------------------------------
    @app.post("/api/login")
    def login(body: LoginBody):
        if not auth.verify_password(cfg, body.password):
            raise HTTPException(status_code=401, detail="wrong password")
        return {"token": auth.issue_token(cfg)}

    def _today_counts() -> dict:
        """Per-category artifact counts for today, from the DB (survives restarts)."""
        import sqlite3
        from ..taxonomy import category
        d = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
        lo, hi = d.timestamp(), (d + timedelta(days=1)).timestamp()
        counts = {"person": 0, "vehicle": 0, "other": 0}
        conn = sqlite3.connect(str(cfg.db_path))
        for (cls,) in conn.execute(
            "SELECT primary_class FROM artifacts WHERE start_time>=? AND start_time<?", (lo, hi)
        ):
            counts[category(cls)] = counts.get(category(cls), 0) + 1
        conn.close()
        return counts

    # -- live --------------------------------------------------------------
    @app.get("/api/stats", dependencies=[guard])
    def stats():
        s = app.state.service.live_stats()
        # "today" figures come from the DB, not the in-memory session counter,
        # so they reflect the full day and persist across restarts.
        today = _today_counts()
        s["daily"] = today
        s["artifacts"] = today
        return s

    @app.get("/api/stream", dependencies=[guard])
    def stream(overlay: int = 1):
        service = app.state.service
        boundary = "frame"
        want_overlay = bool(overlay)

        def gen():
            last_id = -1
            deadline = time.time() + 3600  # cap a single connection at 1h
            while time.time() < deadline:
                jpeg, jid = service.latest_jpeg(overlay=want_overlay)
                if jpeg is None or jid == last_id:
                    time.sleep(0.03)
                    continue
                last_id = jid
                yield (b"--" + boundary.encode() + b"\r\n"
                       b"Content-Type: image/jpeg\r\n"
                       b"Content-Length: " + str(len(jpeg)).encode() + b"\r\n\r\n"
                       + jpeg + b"\r\n")
                time.sleep(0.02)

        return StreamingResponse(
            gen(), media_type=f"multipart/x-mixed-replace; boundary={boundary}")

    # -- artifacts / events ------------------------------------------------
    @app.get("/api/artifacts", dependencies=[guard])
    def artifacts(limit: int = 60, cls: str | None = None):
        rows = app.state.service.db.recent_artifacts(limit=min(limit, 200), cls=cls)
        return [_artifact_json(a) for a in rows]

    @app.get("/api/artifacts/{artifact_id}", dependencies=[guard])
    def artifact(artifact_id: int):
        a = app.state.service.db.get_artifact(artifact_id)
        if not a:
            raise HTTPException(status_code=404, detail="not found")
        return _artifact_json(a)

    @app.get("/api/events", dependencies=[guard])
    def events(limit: int = 80):
        return app.state.service.db.recent_events(limit=min(limit, 300))

    @app.get("/api/media/{artifact_id}/{rank}", dependencies=[guard])
    def media(artifact_id: int, rank: int, group_id: int | None = None):
        path = app.state.service.db.image_path(artifact_id, rank)
        if not path:
            raise HTTPException(status_code=404, detail="no image")
        ap = Path(path).resolve()
        if not str(ap).startswith(str(cfg.images_dir.resolve())) or not ap.is_file():
            raise HTTPException(status_code=404, detail="no image")
            
        return FileResponse(str(ap), media_type="image/jpeg")

    @app.get("/api/media/{artifact_id}/{rank}/full", dependencies=[guard])
    def media_full(artifact_id: int, rank: int, draw_box: int = 0):
        with app.state.service.db._lock:
            row = app.state.service.db._conn.execute(
                "SELECT frame_time FROM artifact_images WHERE artifact_id=? AND rank=?",
                (artifact_id, rank)
            ).fetchone()
            
        if row and row[0] is not None:
            frame_time = row[0]
            rec = app.state.service.recorder
            if rec:
                segs, offset = rec._covering(frame_time, frame_time + 1.0)
                if segs:
                    seg_file = rec.dir / segs[0]["name"]
                    cmd = [
                        "ffmpeg", "-y", "-ss", f"{offset:.2f}",
                        "-i", str(seg_file),
                        "-vframes", "1",
                        "-f", "image2",
                        "-q:v", "2",
                        "-"
                    ]
                    try:
                        import subprocess
                        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        stdout, stderr = proc.communicate(timeout=4)
                        if proc.returncode == 0 and stdout:
                            if draw_box:
                                with app.state.service.db._lock:
                                    bbox_row = app.state.service.db._conn.execute(
                                        "SELECT bbox_json FROM artifacts WHERE id=?", (artifact_id,)
                                    ).fetchone()
                                if bbox_row and bbox_row[0]:
                                    import json
                                    import cv2
                                    import numpy as np
                                    bbox = json.loads(bbox_row[0])
                                    nparr = np.frombuffer(stdout, np.uint8)
                                    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                                    if img is not None:
                                        x1, y1, x2, y2 = int(bbox[0]), int(bbox[1]), int(bbox[2]), int(bbox[3])
                                        # Draw thick red bounding box
                                        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 0, 255), 3)
                                        _, encoded = cv2.imencode(".jpg", img)
                                        stdout = encoded.tobytes()
                            return Response(content=stdout, media_type="image/jpeg")
                    except Exception as e:
                        print(f"[app] ffmpeg full frame extraction failed ({e})")

        # Fallback to the cropped image
        path = app.state.service.db.image_path(artifact_id, rank)
        if not path:
            raise HTTPException(status_code=404, detail="no image")
        ap = Path(path).resolve()
        if not str(ap).startswith(str(cfg.images_dir.resolve())) or not ap.is_file():
            raise HTTPException(status_code=404, detail="no image")
        return FileResponse(str(ap), media_type="image/jpeg")

    # -- query -------------------------------------------------------------
    @app.get("/api/query", dependencies=[guard])
    def query(q: str):
        eng = QueryEngine(cfg.db_path)
        try:
            return {"question": q, "answer": eng.answer(q)}
        finally:
            eng.close()

    @app.get("/api/hourly", dependencies=[guard])
    def hourly(cls: str = "person", rng: str = "today"):
        import sqlite3
        conn = sqlite3.connect(str(cfg.db_path))
        lo = None
        if rng == "today":
            d = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            lo, hi = d.timestamp(), (d.timestamp() + 86400)
        buckets = [0] * 24
        sql = "SELECT start_time FROM artifacts WHERE primary_class=?"
        params = [cls]
        if lo is not None:
            sql += " AND start_time>=? AND start_time<?"; params += [lo, hi]
        for (ts,) in conn.execute(sql, params):
            buckets[datetime.fromtimestamp(ts).hour] += 1
        conn.close()
        return {"cls": cls, "range": rng, "buckets": buckets}

    @app.get("/api/timeseries", dependencies=[guard])
    def timeseries(bucket: int = 15, rng: str = "today"):
        """People & vehicles over time in `bucket`-minute slots (default 15).

        Category-aggregated (vehicle = car/truck/bus/… combined), with per-series
        totals and the busiest slot — this is the "what have we actually seen"
        view, sourced from the artifact DB.
        """
        import sqlite3
        from ..taxonomy import category

        now = datetime.now()
        if rng == "24h":
            base = now - timedelta(hours=24)
            span_min = 24 * 60
        else:  # today, midnight -> midnight
            base = now.replace(hour=0, minute=0, second=0, microsecond=0)
            span_min = 24 * 60
        bucket = max(1, min(bucket, 120))
        lo = base.timestamp()
        nb = span_min // bucket
        hi = lo + nb * bucket * 60
        person, vehicle = [0] * nb, [0] * nb

        conn = sqlite3.connect(str(cfg.db_path))
        for cls, ts in conn.execute(
            "SELECT primary_class, start_time FROM artifacts WHERE start_time>=? AND start_time<?",
            (lo, hi),
        ):
            idx = int((ts - lo) // (bucket * 60))
            if 0 <= idx < nb:
                c = category(cls)
                if c == "person":
                    person[idx] += 1
                elif c == "vehicle":
                    vehicle[idx] += 1
        conn.close()

        labels = [(base + timedelta(minutes=bucket * i)).strftime("%H:%M") for i in range(nb)]

        def busiest(arr):
            if not any(arr):
                return None
            i = max(range(nb), key=lambda k: arr[k])
            return {"label": labels[i], "count": arr[i]}

        return {
            "bucket": bucket, "range": rng, "labels": labels,
            "person": person, "vehicle": vehicle,
            "totals": {"person": sum(person), "vehicle": sum(vehicle)},
            "busiest": {"person": busiest(person), "vehicle": busiest(vehicle)},
        }

    # -- DVR (continuous recording + scrub-back timeline) ------------------
    @app.get("/api/dvr/index", dependencies=[guard])
    def dvr_index():
        rec = app.state.service.recorder
        segs = rec.index() if rec else []
        return {"now": time.time(), "segments": segs,
                "retention_h": cfg.record_retention_h}

    @app.get("/api/dvr/timeline", dependencies=[guard])
    def dvr_timeline(hours: float = 24, bucket_min: int = 10):
        """Activity spikes for the scrobble bar: artifact counts per bucket over
        the recent window, split person/vehicle."""
        import sqlite3
        from ..taxonomy import category

        bucket_min = max(1, min(bucket_min, 60))
        span_s = max(1.0, hours) * 3600
        now = time.time()
        lo = now - span_s
        nb = int(span_s // (bucket_min * 60)) or 1
        step = span_s / nb
        person, vehicle, motion = [0] * nb, [0] * nb, [0.0] * nb
        conn = sqlite3.connect(str(cfg.db_path))
        # Movement intensity (what the scrobbler shows) = summed per-artifact
        # motion_distance per bucket. A parked car / standing person barely moves
        # so it barely registers — motion, not mere presence. person/vehicle are
        # kept only for the Stats page, not for the bar.
        for cls, ts, dist in conn.execute(
            "SELECT primary_class, start_time, motion_distance FROM artifacts WHERE start_time>=?",
            (lo,),
        ):
            idx = int((ts - lo) // step)
            if 0 <= idx < nb:
                c = category(cls)
                if c == "person":
                    person[idx] += 1
                elif c == "vehicle":
                    vehicle[idx] += 1
                motion[idx] += (dist or 0.0)
        conn.close()
        return {"start": lo, "end": now, "bucket_s": step,
                "person": person, "vehicle": vehicle,
                "motion": [round(m, 1) for m in motion]}

    @app.get("/api/dvr/segment/{name}", dependencies=[guard])
    def dvr_segment(name: str, request: Request):
        # Only allow the exact seg-*.mp4 files we produced (no path traversal).
        if "/" in name or "\\" in name or parse_segment_start(name) is None:
            raise HTTPException(status_code=404, detail="bad segment")
        path = (cfg.recordings_dir / name).resolve()
        if not str(path).startswith(str(cfg.recordings_dir.resolve())) or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        # Starlette's FileResponse honours the Range header, which the browser
        # <video> element needs to seek within a segment.
        return FileResponse(str(path), media_type="video/mp4")

    @app.get("/api/dvr/play", dependencies=[guard])
    def dvr_play(start: float):
        """One continuous fragmented-mp4 stream concatenated from `start` -> now,
        so scrub-back plays seamlessly instead of in per-segment chunks."""
        rec = app.state.service.recorder
        proc, lst = (rec.play_stream(start) if rec else (None, None))
        if proc is None:
            raise HTTPException(status_code=404, detail="no footage from there")

        def gen():
            try:
                while True:
                    chunk = proc.stdout.read(65536)
                    if not chunk:
                        break
                    yield chunk
            finally:
                try:
                    proc.kill()
                except Exception:  # noqa: BLE001
                    pass
                try:
                    Path(lst).unlink()
                except OSError:
                    pass

        return StreamingResponse(gen(), media_type="video/mp4")

    @app.post("/api/dvr/save", dependencies=[guard])
    def dvr_save(body: SaveClipBody):
        rec = app.state.service.recorder
        if not rec:
            raise HTTPException(status_code=400, detail="recording disabled")
        r = rec.save_clip(body.start, body.end, body.name)
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        return r

    @app.get("/api/dvr/library", dependencies=[guard])
    def dvr_library():
        rec = app.state.service.recorder
        return rec.library_index() if rec else []

    @app.get("/api/dvr/library/{name}", dependencies=[guard])
    def dvr_library_clip(name: str):
        if "/" in name or "\\" in name or not name.endswith(".mp4"):
            raise HTTPException(status_code=404, detail="bad name")
        path = (cfg.library_dir / name).resolve()
        if not str(path).startswith(str(cfg.library_dir.resolve())) or not path.is_file():
            raise HTTPException(status_code=404, detail="not found")
        return FileResponse(str(path), media_type="video/mp4")

    @app.delete("/api/dvr/library/{name}", dependencies=[guard])
    def dvr_library_delete(name: str):
        rec = app.state.service.recorder
        if not rec or not rec.delete_clip(name):
            raise HTTPException(status_code=404, detail="not found")
        return {"ok": True}

    def _artifacts_by_ids(ids: list[int]) -> list[dict]:
        db = app.state.service.db
        out = []
        for aid in ids:
            a = db.get_artifact(aid)
            if a:
                out.append(_artifact_json(a))
        return out

    def _samples(items):
        return [f"/api/media/{s['artifact_id']}/{s['rank']}" for s in items]

    # -- groups ------------------------------------------------------------
    @app.get("/api/groups", dependencies=[guard])
    def groups():
        rows = app.state.service.db.list_groups()
        return [{
            "id": g["id"], "name": g["name"], "kind": g["kind"],
            "notify": bool(g["notify"]), "count": g["count"],
            "hint": g.get("hint"), "pending": g.get("pending", 0),
            "last_seen": g.get("last_seen"),
            "samples": _samples(g["samples"]),
            "tag_key": g.get("tag_key"),
            "tag_value": g.get("tag_value"),
        } for g in rows]

    @app.post("/api/groups/recluster", dependencies=[guard])
    def recluster():
        return app.state.service.groups.recluster()

    @app.post("/api/groups/from-text", dependencies=[guard])
    def group_from_text(body: TextGroupBody):
        r = app.state.service.groups.create_from_text(body.name, body.prompt)
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        return r

    @app.post("/api/groups/from-artifact", dependencies=[guard])
    def group_from_artifact(body: ArtifactGroupBody):
        r = app.state.service.groups.create_from_artifact(body.name, body.artifact_id)
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        return r

    @app.post("/api/groups/{group_id}/name", dependencies=[guard])
    def name_group(group_id: int, body: NameBody):
        gs = app.state.service.groups
        # if it's a cluster suggestion, promote it; otherwise just rename.
        kinds = {g["id"]: g["kind"] for g in app.state.service.db.list_groups()}
        if kinds.get(group_id) == "cluster":
            gs.name_cluster(group_id, body.name)
        else:
            gs.rename(group_id, body.name)
        return {"ok": True}

    @app.post("/api/groups/{group_id}/notify", dependencies=[guard])
    def group_notify(group_id: int, body: NotifyBody):
        app.state.service.groups.set_notify(group_id, body.notify)
        return {"ok": True}

    @app.delete("/api/groups/{group_id}", dependencies=[guard])
    def delete_group(group_id: int):
        app.state.service.groups.delete(group_id)
        return {"ok": True}

    @app.post("/api/groups/tag", dependencies=[guard])
    def tag_artifacts(body: TagArtifactsBody):
        tags_dict = [{"key": t.key, "value": t.value} for t in body.tags]
        r = app.state.service.groups.tag_artifacts(
            body.artifact_ids, tags_dict, body.source_group_id
        )
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        return r

    @app.get("/api/tags/autocomplete", dependencies=[guard])
    def tags_autocomplete():
        return app.state.service.groups.get_tags_autocomplete()

    @app.get("/api/groups/{group_id}", dependencies=[guard])
    def group_members(group_id: int, limit: int = 120, offset: int = 0):
        db = app.state.service.db
        with db._lock:
            rows = db._conn.execute(
                "SELECT gm.artifact_id, gm.status, gm.source FROM group_members gm "
                "WHERE gm.group_id=? AND (gm.status IS NULL OR (gm.status!='rejected' AND gm.status!='removed')) "
                "ORDER BY gm.score DESC LIMIT ? OFFSET ?",
                (group_id, limit, offset)
            ).fetchall()
        out = []
        for r in rows:
            aid, status, source = r[0], r[1], r[2]
            a = db.get_artifact(aid)
            if a:
                aj = _artifact_json(a)
                aj["member_status"] = status
                aj["member_source"] = source
                out.append(aj)
        return out

    @app.post("/api/groups/{group_id}/members/batch", dependencies=[guard])
    def group_members_batch_feedback(group_id: int, body: BatchMemberFeedbackBody, background_tasks: BackgroundTasks):
        if body.status not in ("confirmed", "rejected", "removed"):
            raise HTTPException(status_code=400, detail="bad status")
        gs = app.state.service.groups
        gs.set_members_feedback(group_id, body.artifact_ids, body.status)
        # A rejection is loud: relearn and sweep out other now-failing auto-tags.
        task = gs.reclassify_group if body.status == "rejected" else gs.train_and_backfill
        background_tasks.add_task(task, group_id)
        return {"ok": True}

    @app.post("/api/groups/{group_id}/members/{artifact_id}", dependencies=[guard])
    def group_member_feedback(group_id: int, artifact_id: int, body: MemberBody, background_tasks: BackgroundTasks):
        if body.status not in ("confirmed", "rejected", "removed"):
            raise HTTPException(status_code=400, detail="bad status")
        gs = app.state.service.groups
        gs.set_member_feedback(group_id, artifact_id, body.status)
        task = gs.reclassify_group if body.status == "rejected" else gs.train_and_backfill
        background_tasks.add_task(task, group_id)
        return {"ok": True}

    @app.post("/api/groups/{group_id}/auto-classify-remaining", dependencies=[guard])
    def group_auto_classify_remaining(group_id: int):
        """'I've done enough training — let the model decide the rest.' Re-scores
        the pending suggestions: matches get the tag (auto-classified), the rest
        are dropped. Nothing here retrains the model."""
        r = app.state.service.groups.auto_classify_pending(group_id)
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        return {"ok": True, **r}

    @app.post("/api/groups/{group_id}/backfill", dependencies=[guard])
    def group_backfill(group_id: int, threshold: float | None = Query(None)):
        gs = app.state.service.groups
        # Always recompute centroid first so backfill uses the latest model
        gs._recompute_centroid(group_id)
        matched = gs._backfill_group(group_id, threshold)
        return {"ok": True, "matched": matched}

    # -- search / similar --------------------------------------------------
    @app.get("/api/search", dependencies=[guard])
    def search(q: str):
        hits = app.state.service.groups.search(q)
        arts = {a["id"]: a for a in _artifacts_by_ids([h["artifact_id"] for h in hits])}
        return [{**arts[h["artifact_id"]], "score": h["score"]}
                for h in hits if h["artifact_id"] in arts]

    @app.post("/api/artifacts/{artifact_id}/label", dependencies=[guard])
    def label_region(artifact_id: int, body: LabelRegionBody):
        if len(body.box) != 4:
            raise HTTPException(status_code=400, detail="box must be [x1,y1,x2,y2]")
        r = app.state.service.groups.label_region(
            artifact_id, body.rank, body.box, body.label)
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        return r

    @app.get("/api/artifacts/{artifact_id}/similar", dependencies=[guard])
    def similar(artifact_id: int):
        hits = app.state.service.groups.similar(artifact_id)
        arts = {a["id"]: a for a in _artifacts_by_ids([h["artifact_id"] for h in hits])}
        return [{**arts[h["artifact_id"]], "score": h["score"]}
                for h in hits if h["artifact_id"] in arts]

    # -- entities ----------------------------------------------------------
    @app.get("/api/entities", dependencies=[guard])
    def entities():
        rows = app.state.service.db.list_entities(gap=cfg.visit_gap_seconds)
        return [{
            "id": e["id"], "label": e["label"], "class": e["class"],
            # "occurrences" now = VISITS (continuous presences), not raw detections
            "occurrences": e["visits"], "sightings": e["sightings"],
            "first_seen": e["first_seen"],
            "last_seen": e["last_seen"], "samples": _samples(e["samples"]),
        } for e in rows]

    @app.get("/api/entities/{entity_id}", dependencies=[guard])
    def entity_members(entity_id: int):
        ids = app.state.service.db.entity_members(entity_id)
        return _artifacts_by_ids(ids)

    @app.post("/api/entities/split/{artifact_id}", dependencies=[guard])
    def entity_split(artifact_id: int):
        """Mark an artifact as 'not the same' as the rest of its entity — splits
        it out and remembers the constraint so it won't merge back."""
        r = app.state.service.groups.reject_entity_member(artifact_id)
        if "error" in r:
            raise HTTPException(status_code=400, detail=r["error"])
        return r

    @app.post("/api/entities/rebuild", dependencies=[guard])
    def entities_rebuild():
        """Re-resolve all entities from ReID at the current threshold (use after
        tuning STREETCAPTURE_REID_ENTITY_MATCH). Runs in the background."""
        import threading
        gs = app.state.service.groups
        threading.Thread(target=gs._safe_rebuild, name="ReIDRebuild", daemon=True).start()
        return {"ok": True, "note": "rebuilding in background"}

    @app.get("/api/entities/{entity_id}/timeline", dependencies=[guard])
    def entity_timeline(entity_id: int):
        """When this entity was here, as VISITS (detection dropouts within a
        continuous presence are merged into one visit, not counted separately)."""
        from ..db import merge_visits
        s = app.state.service.db.entity_sightings(entity_id)
        visits = merge_visits(s, cfg.visit_gap_seconds)
        return [{
            "start": v["start"], "end": v["end"], "class": v["class"],
            "artifact_id": v["artifact_ids"][0],
            "detections": len(v["artifact_ids"]),
            "url": f"/api/media/{v['artifact_ids'][0]}/0",
        } for v in visits]

    @app.get("/api/stats/summary", dependencies=[guard])
    def stats_summary(rng: str = "today"):
        """Aggregated dashboard: the object MIX (car/person/van/… counts),
        UNIQUE things (distinct entities) vs total sightings, by category."""
        import sqlite3
        from ..taxonomy import category

        if rng == "24h":
            lo, hi = time.time() - 86400, time.time()
        else:
            d = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            lo, hi = d.timestamp(), (d + timedelta(days=1)).timestamp()

        conn = sqlite3.connect(str(cfg.db_path))
        mix: dict[str, int] = {}
        sightings = 0
        cat_unique = {"person": set(), "vehicle": set(), "other": set()}
        cat_sightings = {"person": 0, "vehicle": 0, "other": 0}
        for cls, eid in conn.execute(
            "SELECT primary_class, entity_id FROM artifacts WHERE start_time>=? AND start_time<?",
            (lo, hi),
        ):
            sightings += 1
            mix[cls] = mix.get(cls, 0) + 1
            c = category(cls)
            cat_sightings[c] += 1
            # distinct physical things = distinct entities (fall back to a unique
            # marker when an artifact never got resolved to an entity)
            cat_unique[c].add(eid if eid is not None else f"a{sightings}")
        conn.close()

        mix_sorted = sorted(mix.items(), key=lambda kv: -kv[1])
        return {
            "range": rng,
            "sightings": sightings,
            "mix": [{"class": k, "count": v} for k, v in mix_sorted],
            "unique": {k: len(v) for k, v in cat_unique.items()},
            "cat_sightings": cat_sightings,
        }

    @app.get("/api/stats/recurring", dependencies=[guard])
    def stats_recurring(rng: str = "today"):
        """The dynamic 'what keeps coming back today' breakdown for the Stats page."""
        if rng == "24h":
            lo, hi = time.time() - 86400, time.time()
        else:
            d = datetime.now().replace(hour=0, minute=0, second=0, microsecond=0)
            lo, hi = d.timestamp(), (d + timedelta(days=1)).timestamp()
        rows = app.state.service.db.recurring_entities(lo, hi)
        return [{
            "entity_id": r["eid"], "label": r["label"], "class": r["cls"],
            "count": r["n"], "first": r["first"], "last": r["last"],
            "samples": _samples(r["samples"]),
        } for r in rows]

    @app.post("/api/entities/{entity_id}/name", dependencies=[guard])
    def name_entity(entity_id: int, body: NameBody):
        app.state.service.db.update_entity_label(entity_id, body.name)
        return {"ok": True}

    # -- notifications -----------------------------------------------------
    @app.get("/api/notify/status", dependencies=[guard])
    def notify_status():
        return {"enabled": bool(cfg.ntfy_topic), "server": cfg.ntfy_server,
                "topic": cfg.ntfy_topic}

    @app.post("/api/notify/test", dependencies=[guard])
    def notify_test():
        if not cfg.ntfy_topic:
            raise HTTPException(status_code=400, detail="No ntfy topic configured (set STREETCAPTURE_NTFY_TOPIC).")
        ok = app.state.service.groups.notify_test()
        if not ok:
            raise HTTPException(status_code=502, detail="ntfy send failed")
        return {"ok": True}

    # -- SPA hosting (must be registered last) -----------------------------
    if (DIST / "assets").is_dir():
        app.mount("/assets", StaticFiles(directory=str(DIST / "assets")), name="assets")

    @app.get("/{full_path:path}")
    def spa(full_path: str):
        if full_path.startswith("api/"):
            raise HTTPException(status_code=404)
        candidate = (DIST / full_path)
        if full_path and candidate.is_file():
            return FileResponse(str(candidate))
        index = DIST / "index.html"
        if index.is_file():
            return FileResponse(str(index))
        return JSONResponse(
            {"detail": "Web UI not built yet. Run: cd web && npm install && npm run build"},
            status_code=200,
        )

    return app
