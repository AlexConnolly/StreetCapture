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

from fastapi import Depends, FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

from ..config import Config
from ..query import QueryEngine
from . import auth
from .engine import PerceptionService

DIST = Path(__file__).resolve().parent.parent.parent / "web" / "dist"


class LoginBody(BaseModel):
    password: str


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
    def stream():
        service = app.state.service
        boundary = "frame"

        def gen():
            last_id = -1
            deadline = time.time() + 3600  # cap a single connection at 1h
            while time.time() < deadline:
                jpeg, jid = service.latest_jpeg()
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
    def media(artifact_id: int, rank: int):
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
