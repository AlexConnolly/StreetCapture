"""SQLite persistence — the permanent memory layer.

Schema (artifacts/artifact.db):

    sessions          one row per process run
    tracks            one row per completed observation (ByteTrack track)
    artifacts         the subset of tracks judged "meaningful"
    artifact_images   3-10 representative crops per artifact
    embeddings        one vector per artifact (model-versioned)
    events            full lifecycle trace

Track ID vs Entity ID
---------------------
ByteTrack IDs restart at 1 every run, so ``tracks.source_track_id`` is only
unique *within a session*. The permanent, never-reused identifier is the
autoincrement primary key. ``artifacts.entity_id`` is reserved for v0.3+
cross-track identity matching and is always NULL for now.

Only the artifact thread writes; the viewer opens its own read-only connection
in a separate process, so a single write lock here is sufficient.
"""

from __future__ import annotations

import sqlite3
import struct
import threading
import time
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    id         INTEGER PRIMARY KEY AUTOINCREMENT,
    started_at REAL,
    source     TEXT,
    model      TEXT
);
CREATE TABLE IF NOT EXISTS tracks (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER,
    source_track_id INTEGER,
    primary_class   TEXT,
    first_seen      REAL,
    last_seen       REAL,
    duration        REAL,
    frames_seen     INTEGER,
    created_at      REAL
);
CREATE TABLE IF NOT EXISTS artifacts (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    track_pk        INTEGER REFERENCES tracks(id),
    session_id      INTEGER,
    source_track_id INTEGER,
    primary_class   TEXT,
    start_time      REAL,
    end_time        REAL,
    duration        REAL,
    avg_confidence  REAL,
    sharpness       REAL,
    visibility      REAL,
    motion_distance REAL,
    track_length    INTEGER,
    bbox_json       TEXT,
    motion_path_json TEXT,
    entity_id       INTEGER,     -- reserved for v0.3+, always NULL for now
    created_at      REAL
);
CREATE TABLE IF NOT EXISTS artifact_images (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id INTEGER REFERENCES artifacts(id),
    path        TEXT,
    frame_time  REAL,
    sharpness   REAL,
    width       INTEGER,
    height      INTEGER,
    rank        INTEGER
);
CREATE TABLE IF NOT EXISTS embeddings (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id   INTEGER REFERENCES artifacts(id),
    dim           INTEGER,
    vector        BLOB,        -- packed float32
    model_version TEXT,
    created_at    REAL
);
CREATE TABLE IF NOT EXISTS labels (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    artifact_id INTEGER REFERENCES artifacts(id),
    type        TEXT,          -- object | subtype | function | company | energy
    value       TEXT
);
CREATE TABLE IF NOT EXISTS events (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      INTEGER,
    type            TEXT,
    source_track_id INTEGER,
    artifact_id     INTEGER,
    class           TEXT,
    duration        REAL,
    reason          TEXT,
    time            REAL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_start ON artifacts(start_time);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(time);
CREATE INDEX IF NOT EXISTS idx_labels_artifact ON labels(artifact_id);
CREATE INDEX IF NOT EXISTS idx_labels_value ON labels(type, value);
"""


def pack_vector(vec) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(SCHEMA)
        self._conn.commit()

    # -- sessions ----------------------------------------------------------
    def start_session(self, source: str, model: str) -> int:
        with self._lock:
            cur = self._conn.execute(
                "INSERT INTO sessions(started_at, source, model) VALUES (?,?,?)",
                (time.time(), str(source), model),
            )
            self._conn.commit()
            return cur.lastrowid

    # -- writes ------------------------------------------------------------
    def insert_track(self, row: dict) -> int:
        return self._insert("tracks", row)

    def insert_artifact(self, row: dict) -> int:
        return self._insert("artifacts", row)

    def insert_image(self, row: dict) -> int:
        return self._insert("artifact_images", row)

    def insert_embedding(self, artifact_id: int, vector, model_version: str) -> int:
        return self._insert("embeddings", {
            "artifact_id": artifact_id,
            "dim": len(vector),
            "vector": pack_vector(vector),
            "model_version": model_version,
            "created_at": time.time(),
        })

    def insert_event(self, row: dict) -> int:
        row.setdefault("time", time.time())
        return self._insert("events", row)

    def insert_label(self, artifact_id: int, ltype: str, value: str) -> int:
        return self._insert("labels", {"artifact_id": artifact_id, "type": ltype, "value": value})

    def _insert(self, table: str, row: dict) -> int:
        cols = ", ".join(row)
        ph = ", ".join("?" for _ in row)
        with self._lock:
            cur = self._conn.execute(
                f"INSERT INTO {table} ({cols}) VALUES ({ph})", tuple(row.values())
            )
            self._conn.commit()
            return cur.lastrowid

    # -- reads (used by the viewer / report) ------------------------------
    def counts(self) -> dict:
        with self._lock:
            def one(q):
                return self._conn.execute(q).fetchone()[0]
            return {
                "tracks": one("SELECT COUNT(*) FROM tracks"),
                "artifacts": one("SELECT COUNT(*) FROM artifacts"),
                "embeddings": one("SELECT COUNT(*) FROM embeddings"),
                "events": one("SELECT COUNT(*) FROM events"),
            }

    def recent_artifacts(self, limit: int = 200) -> list[dict]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                "SELECT * FROM artifacts ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                imgs = self._conn.execute(
                    "SELECT path, sharpness, width, height, rank FROM artifact_images "
                    "WHERE artifact_id=? ORDER BY rank", (r["id"],)
                ).fetchall()
                d["images"] = [dict(i) for i in imgs]
                emb = self._conn.execute(
                    "SELECT model_version, dim FROM embeddings WHERE artifact_id=?", (r["id"],)
                ).fetchone()
                d["embedding"] = dict(emb) if emb else None
                labs = self._conn.execute(
                    "SELECT type, value FROM labels WHERE artifact_id=? ORDER BY id", (r["id"],)
                ).fetchall()
                d["labels"] = [dict(l) for l in labs]
                out.append(d)
            self._conn.row_factory = None
            return out

    def close(self) -> None:
        with self._lock:
            self._conn.close()
