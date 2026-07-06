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
CREATE TABLE IF NOT EXISTS groups (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    name          TEXT,          -- NULL = unnamed cluster suggestion
    kind          TEXT,          -- 'cluster' (ephemeral) | 'labeled'
    centroid      BLOB,          -- image-space mean, packed float32
    dim           INTEGER,
    notify        INTEGER DEFAULT 0,
    last_notified REAL DEFAULT 0,
    size          INTEGER DEFAULT 0,
    created_at    REAL,
    updated_at    REAL
);
CREATE TABLE IF NOT EXISTS group_members (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id    INTEGER REFERENCES groups(id),
    artifact_id INTEGER REFERENCES artifacts(id),
    score       REAL,
    source      TEXT,            -- 'cluster' | 'text' | 'manual' | 'auto'
    status      TEXT,            -- NULL=suggested | 'confirmed' | 'rejected'
    created_at  REAL,
    UNIQUE(group_id, artifact_id)
);
CREATE TABLE IF NOT EXISTS label_seeds (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    group_id    INTEGER REFERENCES groups(id),
    artifact_id INTEGER,
    box         TEXT,          -- normalised [x1,y1,x2,y2] the user drew
    vector      BLOB,          -- CLIP embedding of the cropped region
    dim         INTEGER,
    created_at  REAL
);
CREATE TABLE IF NOT EXISTS entity_dislinks (
    a          INTEGER,        -- two artifacts the user said are NOT the same
    b          INTEGER,        -- (normalised a<b); rebuild won't co-locate them
    created_at REAL,
    UNIQUE(a, b)
);
CREATE TABLE IF NOT EXISTS reid_embeddings (
    artifact_id INTEGER PRIMARY KEY REFERENCES artifacts(id),
    dim         INTEGER,
    vector      BLOB
);
CREATE TABLE IF NOT EXISTS entities (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    label       TEXT,
    centroid    BLOB,
    dim         INTEGER,
    space       TEXT DEFAULT 'clip',   -- 'clip' (objects) | 'reid' (people)
    occurrences INTEGER DEFAULT 0,
    first_seen  REAL,
    last_seen   REAL,
    created_at  REAL
);
CREATE INDEX IF NOT EXISTS idx_artifacts_created ON artifacts(created_at);
CREATE INDEX IF NOT EXISTS idx_artifacts_start ON artifacts(start_time);
CREATE INDEX IF NOT EXISTS idx_artifacts_entity ON artifacts(entity_id);
CREATE INDEX IF NOT EXISTS idx_events_time ON events(time);
CREATE INDEX IF NOT EXISTS idx_labels_artifact ON labels(artifact_id);
CREATE INDEX IF NOT EXISTS idx_labels_value ON labels(type, value);
CREATE INDEX IF NOT EXISTS idx_gm_group ON group_members(group_id);
CREATE INDEX IF NOT EXISTS idx_gm_artifact ON group_members(artifact_id);
"""


def pack_vector(vec) -> bytes:
    return struct.pack(f"{len(vec)}f", *vec)


def unpack_vector(blob: bytes) -> list[float]:
    return list(struct.unpack(f"{len(blob) // 4}f", blob))


def merge_visits(sightings: list[dict], gap: float) -> list[dict]:
    """Collapse time-ordered sightings ({start,end,artifact_id,...}) into visits:
    consecutive sightings less than `gap` seconds apart become one visit. This is
    how "seen N times" should count — one continuous presence is one visit, even
    if detection blinked and split it into several artifacts."""
    visits: list[dict] = []
    for s in sorted(sightings, key=lambda x: x["start"]):
        if visits and s["start"] - visits[-1]["end"] <= gap:
            v = visits[-1]
            v["end"] = max(v["end"], s["end"])
            v["artifact_ids"].append(s["artifact_id"])
        else:
            visits.append({"start": s["start"], "end": s["end"],
                           "class": s.get("class"), "artifact_ids": [s["artifact_id"]]})
    return visits


class Database:
    def __init__(self, path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(str(self.path), check_same_thread=False)
        self._conn.execute("PRAGMA journal_mode=WAL;")
        self._conn.executescript(SCHEMA)
        self._migrate()
        self._conn.commit()

    def _migrate(self) -> None:
        """Additive migrations for DBs created by an older schema."""
        cols = {r[1] for r in self._conn.execute("PRAGMA table_info(group_members)")}
        if "status" not in cols:
            self._conn.execute("ALTER TABLE group_members ADD COLUMN status TEXT")
        gcols = {r[1] for r in self._conn.execute("PRAGMA table_info(groups)")}
        if "hint" not in gcols:
            self._conn.execute("ALTER TABLE groups ADD COLUMN hint TEXT")
        if "tag_key" not in gcols:
            self._conn.execute("ALTER TABLE groups ADD COLUMN tag_key TEXT")
        if "tag_value" not in gcols:
            self._conn.execute("ALTER TABLE groups ADD COLUMN tag_value TEXT")
        if "match_threshold" not in gcols:
            self._conn.execute("ALTER TABLE groups ADD COLUMN match_threshold REAL")
        ecols = {r[1] for r in self._conn.execute("PRAGMA table_info(entities)")}
        if "space" not in ecols:
            self._conn.execute("ALTER TABLE entities ADD COLUMN space TEXT DEFAULT 'clip'")

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

    def _update(self, table: str, row_id: int, fields: dict) -> None:
        sets = ", ".join(f"{k}=?" for k in fields)
        with self._lock:
            self._conn.execute(f"UPDATE {table} SET {sets} WHERE id=?",
                               (*fields.values(), row_id))
            self._conn.commit()

    # -- embeddings (for clustering / matching) ---------------------------
    def all_embeddings(self) -> list[tuple]:
        """[(artifact_id, [float...], primary_class), ...] for every embedded artifact."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.artifact_id, e.vector, a.primary_class "
                "FROM embeddings e JOIN artifacts a ON a.id=e.artifact_id "
                "ORDER BY e.artifact_id"
            ).fetchall()
        return [(r[0], unpack_vector(r[1]), r[2]) for r in rows]

    def embeddings_for_clustering(self) -> list[tuple]:
        """Like all_embeddings, but excludes artifacts already confirmed into a
        labeled group. Those are categorised, so re-proposing them as fresh
        cluster suggestions just nags about things the user has handled."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.artifact_id, e.vector, a.primary_class "
                "FROM embeddings e JOIN artifacts a ON a.id=e.artifact_id "
                "WHERE e.artifact_id NOT IN ("
                "  SELECT gm.artifact_id FROM group_members gm "
                "  JOIN groups g ON g.id=gm.group_id "
                "  WHERE g.kind='labeled' AND gm.status='confirmed') "
                "ORDER BY e.artifact_id"
            ).fetchall()
        return [(r[0], unpack_vector(r[1]), r[2]) for r in rows]

    def embedding_for(self, artifact_id: int):
        with self._lock:
            r = self._conn.execute(
                "SELECT vector FROM embeddings WHERE artifact_id=?", (artifact_id,)
            ).fetchone()
        return unpack_vector(r[0]) if r else None

    def embeddings_missing_entity(self) -> list[tuple]:
        """[(artifact_id, [float...]), ...] for embedded artifacts with no entity yet."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.artifact_id, e.vector FROM embeddings e "
                "JOIN artifacts a ON a.id=e.artifact_id "
                "WHERE a.entity_id IS NULL ORDER BY e.artifact_id"
            ).fetchall()
        return [(r[0], unpack_vector(r[1])) for r in rows]

    def embeddings_missing_entity_with_class(self) -> list[tuple]:
        """[(artifact_id, [float...], primary_class), ...] with no entity yet."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT e.artifact_id, e.vector, a.primary_class FROM embeddings e "
                "JOIN artifacts a ON a.id=e.artifact_id "
                "WHERE a.entity_id IS NULL ORDER BY e.artifact_id"
            ).fetchall()
        return [(r[0], unpack_vector(r[1]), r[2]) for r in rows]

    # -- groups ------------------------------------------------------------
    def insert_group(self, name, kind, centroid, size, notify=0, tag_key=None, tag_value=None) -> int:
        now = time.time()
        return self._insert("groups", {
            "name": name, "kind": kind,
            "centroid": pack_vector(centroid) if centroid is not None else None,
            "dim": len(centroid) if centroid is not None else 0,
            "notify": notify, "size": size, "created_at": now, "updated_at": now,
            "tag_key": tag_key, "tag_value": tag_value
        })

    def update_group(self, group_id: int, **fields) -> None:
        if "centroid" in fields and fields["centroid"] is not None:
            vec = fields["centroid"]
            fields["centroid"] = pack_vector(vec)
            fields["dim"] = len(vec)
        fields["updated_at"] = time.time()
        self._update("groups", group_id, fields)

    def delete_group(self, group_id: int) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM group_members WHERE group_id=?", (group_id,))
            self._conn.execute("DELETE FROM groups WHERE id=?", (group_id,))
            self._conn.commit()

    def delete_groups_by_kind(self, kind: str) -> None:
        with self._lock:
            ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM groups WHERE kind=?", (kind,)).fetchall()]
            for gid in ids:
                self._conn.execute("DELETE FROM group_members WHERE group_id=?", (gid,))
            self._conn.execute("DELETE FROM groups WHERE kind=?", (kind,))
            self._conn.commit()

    def recluster_save(self, clusters_data: list[dict]) -> None:
        """Save reclustered groups in a single transaction/commit to avoid locking sqlite."""
        now = time.time()
        with self._lock:
            # 1. Delete all existing clusters
            ids = [r[0] for r in self._conn.execute(
                "SELECT id FROM groups WHERE kind='cluster'").fetchall()]
            for gid in ids:
                self._conn.execute("DELETE FROM group_members WHERE group_id=?", (gid,))
            self._conn.execute("DELETE FROM groups WHERE kind='cluster'")
            
            # 2. Bulk insert new clusters
            for cluster in clusters_data:
                centroid = cluster["centroid"]
                size = cluster["size"]
                hint = cluster.get("hint")
                deviation = cluster.get("deviation", 0.0)
                members = cluster["members"] # list of (artifact_id, score)
                
                # Insert group
                cur = self._conn.execute(
                    "INSERT INTO groups (name, kind, centroid, dim, notify, size, hint, deviation, created_at, updated_at) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (None, "cluster", pack_vector(centroid), len(centroid), 0, size, hint, deviation, now, now)
                )
                gid = cur.lastrowid
                
                # Insert group members in bulk
                member_data = [
                    (gid, aid, round(sc, 4), "cluster", now)
                    for aid, sc in members
                ]
                self._conn.executemany(
                    "INSERT OR IGNORE INTO group_members (group_id, artifact_id, score, source, created_at) "
                    "VALUES (?, ?, ?, ?, ?)",
                    member_data
                )
                
            self._conn.commit()

    def add_member(self, group_id: int, artifact_id: int, score: float, source: str, status: str | None = None) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO group_members(group_id, artifact_id, score, source, status, created_at) "
                "VALUES (?,?,?,?,?,?)", (group_id, artifact_id, round(score, 4), source, status, time.time()))
            self._conn.commit()

    def add_members(self, group_id: int, member_data: list[tuple]) -> None:
        if not member_data:
            return
        now = time.time()
        with self._lock:
            rows = []
            for item in member_data:
                aid, sc, src = item[0], item[1], item[2]
                status = item[3] if len(item) > 3 else None
                rows.append((group_id, aid, round(sc, 4), src, status, now))
            self._conn.executemany(
                "INSERT OR IGNORE INTO group_members(group_id, artifact_id, score, source, status, created_at) "
                "VALUES (?,?,?,?,?,?)", rows
            )
            self._conn.commit()

    def labeled_group_centroids(self) -> list[tuple]:
        """[(group_id, name, notify, last_notified, [centroid...], match_threshold), ...]
        for labeled groups. match_threshold is NULL for mean centroids (callers fall
        back to the global cosine defaults) and a calibrated cut for LR centroids."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, notify, last_notified, centroid, match_threshold FROM groups "
                "WHERE kind='labeled' AND centroid IS NOT NULL").fetchall()
        return [(r[0], r[1], r[2], r[3], unpack_vector(r[4]), r[5]) for r in rows]

    def group_member_vectors(self, group_id: int, status: str | None = None,
                             exclude_rejected: bool = False,
                             exclude_auto_confirm: bool = False) -> list:
        """Member embedding vectors, optionally filtered by feedback status.

        status='confirmed'   -> only thumbs-up members.
        exclude_rejected     -> everything except thumbs-down members.
        exclude_auto_confirm -> drop machine auto-classified members, so only
                                human-vouched examples train the centroid.
        """
        sql = ("SELECT e.vector FROM group_members gm "
               "JOIN embeddings e ON e.artifact_id=gm.artifact_id WHERE gm.group_id=?")
        params = [group_id]
        if status is not None:
            sql += " AND gm.status=?"; params.append(status)
        elif exclude_rejected:
            sql += " AND (gm.status IS NULL OR (gm.status!='rejected' AND gm.status!='removed'))"
        if exclude_auto_confirm:
            sql += " AND (gm.source IS NULL OR gm.source!='auto_confirm')"
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [unpack_vector(r[0]) for r in rows]

    def human_confirmed_count(self, group_id: int) -> int:
        """Members the user actually vouched for (confirmed, not machine
        auto-classified). This is the signal that a group is trustworthy."""
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM group_members WHERE group_id=? AND status='confirmed' "
                "AND (source IS NULL OR source!='auto_confirm')", (group_id,)).fetchone()[0]

    def pending_member_ids(self, group_id: int) -> list[int]:
        """Un-reviewed auto-suggestions (status NULL) awaiting a yes/no."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT artifact_id FROM group_members WHERE group_id=? AND status IS NULL",
                (group_id,)).fetchall()
        return [r[0] for r in rows]

    def mark_auto_classified(self, group_id: int, artifact_ids: list[int]) -> None:
        """Confirm members as machine-classified: they carry the tag but
        source='auto_confirm' keeps them out of centroid training."""
        if not artifact_ids:
            return
        with self._lock:
            self._conn.executemany(
                "UPDATE group_members SET status='confirmed', source='auto_confirm' "
                "WHERE group_id=? AND artifact_id=?",
                [(group_id, aid) for aid in artifact_ids])
            self._conn.commit()

    def remove_members(self, group_id: int, artifact_ids: list[int]) -> None:
        if not artifact_ids:
            return
        with self._lock:
            self._conn.executemany(
                "DELETE FROM group_members WHERE group_id=? AND artifact_id=?",
                [(group_id, aid) for aid in artifact_ids])
            self._conn.commit()

    def background_negative_vectors(self, group_id: int, classes: list[str], limit: int = 200) -> list:
        """Fetch other vectors of the same classes that are not in the specified group."""
        if not classes:
            return []
        placeholders = ",".join("?" for _ in classes)
        sql = (
            f"SELECT e.vector FROM embeddings e JOIN artifacts a ON a.id=e.artifact_id "
            f"WHERE a.primary_class IN ({placeholders}) AND a.id NOT IN "
            f"(SELECT artifact_id FROM group_members WHERE group_id=?) "
            f"LIMIT ?"
        )
        params = list(classes) + [group_id, limit]
        with self._lock:
            rows = self._conn.execute(sql, params).fetchall()
        return [unpack_vector(r[0]) for r in rows]

    def set_member_status(self, group_id: int, artifact_id: int, status: str | None) -> None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id FROM group_members WHERE group_id=? AND artifact_id=?",
                (group_id, artifact_id)
            ).fetchone()
            if row:
                self._conn.execute(
                    "UPDATE group_members SET status=? WHERE id=?",
                    (status, row[0])
                )
            else:
                self._conn.execute(
                    "INSERT INTO group_members (group_id, artifact_id, score, source, created_at, status) "
                    "VALUES (?, ?, 1.0, 'user', ?, ?)",
                    (group_id, artifact_id, time.time(), status)
                )
            self._conn.commit()

    def set_members_status(self, group_id: int, artifact_ids: list[int], status: str | None) -> None:
        with self._lock:
            for aid in artifact_ids:
                row = self._conn.execute(
                    "SELECT id FROM group_members WHERE group_id=? AND artifact_id=?",
                    (group_id, aid)
                ).fetchone()
                if row:
                    self._conn.execute(
                        "UPDATE group_members SET status=? WHERE id=?",
                        (status, row[0])
                    )
                else:
                    self._conn.execute(
                        "INSERT INTO group_members (group_id, artifact_id, score, source, created_at, status) "
                        "VALUES (?, ?, 1.0, 'user', ?, ?)",
                        (group_id, aid, time.time(), status)
                    )
            self._conn.commit()

    def remove_member(self, group_id: int, artifact_id: int) -> None:
        with self._lock:
            self._conn.execute(
                "DELETE FROM group_members WHERE group_id=? AND artifact_id=?",
                (group_id, artifact_id))
            self._conn.commit()

    def rejected_member_ids(self, group_id: int) -> set:
        with self._lock:
            rows = self._conn.execute(
                "SELECT artifact_id FROM group_members "
                "WHERE group_id=? AND status='rejected'", (group_id,)).fetchall()
        return {r[0] for r in rows}

    def member_status_map(self, group_id: int) -> dict:
        """artifact_id -> {'status', 'source', 'score'} for a group's members."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT artifact_id, status, source, score FROM group_members "
                "WHERE group_id=?", (group_id,)).fetchall()
        return {r[0]: {"status": r[1], "source": r[2], "score": r[3]} for r in rows}

    def list_groups(self, sample: int = 4) -> list[dict]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            grows = self._conn.execute(
                "SELECT id, name, kind, notify, size, hint, deviation, tag_key, tag_value FROM groups ORDER BY "
                "(name IS NULL), deviation DESC, size DESC, id DESC").fetchall()
            out = []
            for g in grows:
                d = dict(g)
                # exclude rejected members from the card count + thumbnails so it
                # matches what you see inside the group.
                n = self._conn.execute(
                    "SELECT COUNT(*) FROM group_members WHERE group_id=? "
                    "AND (status IS NULL OR status!='rejected')", (g["id"],)).fetchone()[0]
                d["count"] = n
                # unreviewed members (auto-added, not yet confirmed/rejected) — the
                # "N new to approve/decline" prompt.
                d["pending"] = self._conn.execute(
                    "SELECT COUNT(*) FROM group_members WHERE group_id=? AND status IS NULL "
                    "AND source='auto'", (g["id"],)).fetchone()[0]
                # when this group's thing was last seen (latest member artifact)
                d["last_seen"] = self._conn.execute(
                    "SELECT MAX(a.start_time) FROM group_members gm "
                    "JOIN artifacts a ON a.id=gm.artifact_id WHERE gm.group_id=? "
                    "AND (gm.status IS NULL OR gm.status!='rejected')", (g["id"],)).fetchone()[0]
                imgs = self._conn.execute(
                    "SELECT ai.artifact_id, ai.path FROM group_members gm "
                    "JOIN artifact_images ai ON ai.artifact_id=gm.artifact_id AND ai.rank=0 "
                    "WHERE gm.group_id=? AND (gm.status IS NULL OR gm.status!='rejected') "
                    "ORDER BY gm.score DESC LIMIT ?", (g["id"], sample)).fetchall()
                d["samples"] = [{"artifact_id": r["artifact_id"], "rank": 0} for r in imgs]
                out.append(d)
            self._conn.row_factory = None
            return out

    def group_members(self, group_id: int) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT artifact_id FROM group_members WHERE group_id=? ORDER BY score DESC",
                (group_id,)).fetchall()
        return [r[0] for r in rows]

    def touch_group_notified(self, group_id: int) -> None:
        self._update("groups", group_id, {"last_notified": time.time()})

    def set_group_hint(self, group_id: int, hint: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE groups SET hint=? WHERE id=?", (hint, group_id))
            self._conn.commit()

    # -- custom region labels (few-shot prototypes) -----------------------
    def group_id_by_name(self, name: str) -> int | None:
        with self._lock:
            r = self._conn.execute(
                "SELECT id FROM groups WHERE kind='labeled' AND name=? COLLATE NOCASE "
                "ORDER BY id LIMIT 1", (name,)).fetchone()
        return r[0] if r else None

    def add_label_seed(self, group_id: int, artifact_id: int, box, vector) -> int:
        import json
        return self._insert("label_seeds", {
            "group_id": group_id, "artifact_id": artifact_id,
            "box": json.dumps(box), "vector": pack_vector(vector),
            "dim": len(vector), "created_at": time.time()})

    def seed_vectors(self, group_id: int) -> list:
        with self._lock:
            rows = self._conn.execute(
                "SELECT vector FROM label_seeds WHERE group_id=?", (group_id,)).fetchall()
        return [unpack_vector(r[0]) for r in rows]

    def seed_count(self, group_id: int) -> int:
        with self._lock:
            return self._conn.execute(
                "SELECT COUNT(*) FROM label_seeds WHERE group_id=?", (group_id,)).fetchone()[0]

    def seeded_group_ids(self) -> set:
        with self._lock:
            rows = self._conn.execute("SELECT DISTINCT group_id FROM label_seeds").fetchall()
        return {r[0] for r in rows}

    # -- recurring things (Stats page) ------------------------------------
    def recurring_entities(self, lo: float, hi: float, limit: int = 12,
                           sample: int = 3) -> list[dict]:
        """NAMED regulars seen in [lo, hi), ranked by sighting count. Only labelled
        entities appear — anonymous same-class entities (many distinct 'person's)
        are aggregated in the object-mix/unique counts instead, so this list reads
        like a dashboard of the things you actually track."""
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                "SELECT a.entity_id AS eid, e.label AS label, a.primary_class AS cls, "
                "       COUNT(*) AS n, MIN(a.start_time) AS first, MAX(a.start_time) AS last "
                "FROM artifacts a JOIN entities e ON e.id=a.entity_id "
                "WHERE a.entity_id IS NOT NULL AND e.label IS NOT NULL "
                "  AND a.start_time>=? AND a.start_time<? "
                "GROUP BY a.entity_id "
                "ORDER BY n DESC, last DESC LIMIT ?", (lo, hi, limit)).fetchall()
            out = []
            for r in rows:
                d = dict(r)
                imgs = self._conn.execute(
                    "SELECT ai.artifact_id FROM artifacts a "
                    "JOIN artifact_images ai ON ai.artifact_id=a.id AND ai.rank=0 "
                    "WHERE a.entity_id=? ORDER BY a.id DESC LIMIT ?", (r["eid"], sample)).fetchall()
                d["samples"] = [{"artifact_id": x["artifact_id"], "rank": 0} for x in imgs]
                out.append(d)
            self._conn.row_factory = None
            return out

    def entity_sightings(self, entity_id: int) -> list[dict]:
        """Every sighting of an entity as [{artifact_id, start, end, class}]."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, start_time, end_time, primary_class FROM artifacts "
                "WHERE entity_id=? ORDER BY start_time", (entity_id,)).fetchall()
        return [{"artifact_id": r[0], "start": r[1], "end": r[2], "class": r[3]} for r in rows]

    # -- entities ----------------------------------------------------------
    def insert_entity(self, centroid, now, space: str = "clip") -> int:
        return self._insert("entities", {
            "label": None, "centroid": pack_vector(centroid), "dim": len(centroid),
            "space": space, "occurrences": 1,
            "first_seen": now, "last_seen": now, "created_at": now,
        })

    def entity_centroids(self) -> list[tuple]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, occurrences, centroid, space FROM entities").fetchall()
        return [(r[0], r[1], unpack_vector(r[2]), r[3] or "clip") for r in rows]

    # -- person ReID embeddings (identity) --------------------------------
    def insert_reid(self, artifact_id: int, vector) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT OR REPLACE INTO reid_embeddings(artifact_id, dim, vector) VALUES (?,?,?)",
                (artifact_id, len(vector), pack_vector(vector)))
            self._conn.commit()

    def reid_for(self, artifact_id: int):
        with self._lock:
            r = self._conn.execute(
                "SELECT vector FROM reid_embeddings WHERE artifact_id=?", (artifact_id,)).fetchone()
        return unpack_vector(r[0]) if r else None

    def artifacts_missing_reid(self) -> list[tuple]:
        """[(artifact_id, image_path), ...] for any artifact with no ReID yet."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT a.id, ai.path FROM artifacts a "
                "JOIN artifact_images ai ON ai.artifact_id=a.id AND ai.rank=0 "
                "LEFT JOIN reid_embeddings r ON r.artifact_id=a.id "
                "WHERE r.artifact_id IS NULL ORDER BY a.id").fetchall()
        return [(r[0], r[1]) for r in rows]

    def all_reid_with_times(self) -> list[tuple]:
        """[(artifact_id, [float...], start_time, end_time), ...] for the rebuild."""
        with self._lock:
            rows = self._conn.execute(
                "SELECT r.artifact_id, r.vector, a.start_time, a.end_time "
                "FROM reid_embeddings r JOIN artifacts a ON a.id=r.artifact_id "
                "ORDER BY r.artifact_id").fetchall()
        return [(r[0], unpack_vector(r[1]), r[2], r[3]) for r in rows]

    def rebuild_entities_write(self, entities: list[dict], assignments: list[tuple],
                               now: float) -> None:
        """Batch-create entities and point artifacts at them in ONE transaction
        (a per-row commit made the rebuild take minutes)."""
        with self._lock:
            ids = []
            for e in entities:
                cur = self._conn.execute(
                    "INSERT INTO entities(label,centroid,dim,space,occurrences,"
                    "first_seen,last_seen,created_at) VALUES (NULL,?,?,?,?,?,?,?)",
                    (pack_vector(e["centroid"]), len(e["centroid"]), e["space"],
                     e["occ"], e["first_seen"], e["last_seen"], now))
                ids.append(cur.lastrowid)
            for aid, idx in assignments:
                self._conn.execute("UPDATE artifacts SET entity_id=? WHERE id=?", (ids[idx], aid))
            self._conn.commit()

    def reset_entities(self) -> None:
        """Wipe all entities and unlink artifacts — for a clean identity rebuild."""
        with self._lock:
            self._conn.execute("DELETE FROM entities")
            self._conn.execute("UPDATE artifacts SET entity_id=NULL")
            self._conn.commit()

    def update_entity(self, entity_id, centroid, occurrences, last_seen) -> None:
        self._update("entities", entity_id, {
            "centroid": pack_vector(centroid), "dim": len(centroid),
            "occurrences": occurrences, "last_seen": last_seen})

    def set_artifact_entity(self, artifact_id, entity_id) -> None:
        self._update("artifacts", artifact_id, {"entity_id": entity_id})

    def list_entities(self, min_visits: int = 2, sample: int = 4,
                      gap: float = 120.0) -> list[dict]:
        """Entities that genuinely recur — ranked by VISIT count (a continuous
        presence split by detection dropouts counts as one visit, not many)."""
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            # occurrences>=2 is a cheap prefilter; the real gate is visits below.
            rows = self._conn.execute(
                "SELECT id, label, occurrences, first_seen, last_seen FROM entities "
                "WHERE occurrences>=2 ORDER BY occurrences DESC, last_seen DESC").fetchall()
            out = []
            for e in rows:
                times = self._conn.execute(
                    "SELECT id, start_time, end_time FROM artifacts WHERE entity_id=?",
                    (e["id"],)).fetchall()
                visits = merge_visits(
                    [{"artifact_id": r["id"], "start": r["start_time"], "end": r["end_time"]}
                     for r in times], gap)
                if len(visits) < min_visits:
                    continue
                d = dict(e)
                d["visits"] = len(visits)
                d["sightings"] = e["occurrences"]
                imgs = self._conn.execute(
                    "SELECT ai.artifact_id, a.primary_class FROM artifacts a "
                    "JOIN artifact_images ai ON ai.artifact_id=a.id AND ai.rank=0 "
                    "WHERE a.entity_id=? ORDER BY a.id DESC LIMIT ?", (e["id"], sample)).fetchall()
                d["samples"] = [{"artifact_id": r["artifact_id"], "rank": 0} for r in imgs]
                d["class"] = imgs[0]["primary_class"] if imgs else "?"
                out.append(d)
            self._conn.row_factory = None
            out.sort(key=lambda d: (d["visits"], d["last_seen"]), reverse=True)
            return out

    def entity_members(self, entity_id: int) -> list[int]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id FROM artifacts WHERE entity_id=? ORDER BY id DESC", (entity_id,)).fetchall()
        return [r[0] for r in rows]

    def entity_of(self, artifact_id: int):
        with self._lock:
            r = self._conn.execute(
                "SELECT entity_id FROM artifacts WHERE id=?", (artifact_id,)).fetchone()
        return r[0] if r else None

    # -- "not the same" constraints (survive rebuilds) --------------------
    def add_dislink(self, a: int, b: int) -> None:
        a, b = (a, b) if a < b else (b, a)
        if a == b:
            return
        with self._lock:
            self._conn.execute(
                "INSERT OR IGNORE INTO entity_dislinks(a, b, created_at) VALUES (?,?,?)",
                (a, b, time.time()))
            self._conn.commit()

    def all_dislinks(self) -> list[tuple]:
        with self._lock:
            rows = self._conn.execute("SELECT a, b FROM entity_dislinks").fetchall()
        return [(r[0], r[1]) for r in rows]

    def update_entity_label(self, entity_id: int, label: str) -> None:
        self._update("entities", entity_id, {"label": label})

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

    def recent_events(self, limit: int = 100) -> list[dict]:
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                "SELECT type, source_track_id, artifact_id, class, duration, reason, time "
                "FROM events ORDER BY id DESC LIMIT ?", (limit,)
            ).fetchall()
            self._conn.row_factory = None
            return [dict(r) for r in rows]

    def image_path(self, artifact_id: int, rank: int):
        with self._lock:
            row = self._conn.execute(
                "SELECT path FROM artifact_images WHERE artifact_id=? AND rank=?",
                (artifact_id, rank),
            ).fetchone()
            return row[0] if row else None

    def get_artifact(self, artifact_id: int):
        arts = self._recent_artifacts_where("WHERE id=?", (artifact_id,), 1)
        return arts[0] if arts else None

    def recent_artifacts(self, limit: int = 200, cls: str | None = None) -> list[dict]:
        if cls:
            return self._recent_artifacts_where("WHERE primary_class=?", (cls,), limit)
        return self._recent_artifacts_where("", (), limit)

    def _recent_artifacts_where(self, where: str, params: tuple, limit: int) -> list[dict]:
        import json
        with self._lock:
            self._conn.row_factory = sqlite3.Row
            rows = self._conn.execute(
                f"SELECT * FROM artifacts {where} ORDER BY id DESC LIMIT ?", (*params, limit)
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
                
                group_tags = self._conn.execute(
                    "SELECT g.tag_key AS type, g.tag_value AS value FROM group_members gm "
                    "JOIN groups g ON g.id=gm.group_id "
                    "WHERE gm.artifact_id=? AND g.kind='labeled' AND g.tag_key IS NOT NULL "
                    "AND gm.status='confirmed'", (r["id"],)
                ).fetchall()
                merged_labs = { (l["type"], l["value"]): l for l in d["labels"] }
                for gt in group_tags:
                    merged_labs[(gt["type"], gt["value"])] = {"type": gt["type"], "value": gt["value"]}
                d["labels"] = list(merged_labs.values())
                
                # Fetch label seeds
                seeds = self._conn.execute(
                    "SELECT ls.group_id, g.name AS label, ls.box FROM label_seeds ls "
                    "JOIN groups g ON g.id=ls.group_id "
                    "WHERE ls.artifact_id=? ORDER BY ls.id", (r["id"],)
                ).fetchall()
                d["seeds"] = [
                    {"group_id": s["group_id"], "label": s["label"], "box": json.loads(s["box"])}
                    for s in seeds
                ]
                
                out.append(d)
            self._conn.row_factory = None
            return out

    def group_allowed_classes(self, group_id: int) -> set[str]:
        """Return the set of primary_classes of seeds or confirmed members of a group.
        If none exist, falls back to manual/text/seed/unrejected members."""
        with self._lock:
            # Classes of seeds
            seed_rows = self._conn.execute(
                "SELECT DISTINCT a.primary_class FROM label_seeds s "
                "JOIN artifacts a ON s.artifact_id=a.id WHERE s.group_id=?",
                (group_id,)
            ).fetchall()
            
            # Classes of manual/confirmed/text members
            member_rows = self._conn.execute(
                "SELECT DISTINCT a.primary_class FROM group_members m "
                "JOIN artifacts a ON m.artifact_id=a.id "
                "WHERE m.group_id=? AND (m.status='confirmed' OR m.source IN ('seed', 'manual', 'text'))",
                (group_id,)
            ).fetchall()
            
            classes = {r[0] for r in seed_rows if r[0]} | {r[0] for r in member_rows if r[0]}
            
            # Fallback to any member (excluding rejected ones) if empty
            if not classes:
                fallback_rows = self._conn.execute(
                    "SELECT DISTINCT a.primary_class FROM group_members m "
                    "JOIN artifacts a ON m.artifact_id=a.id "
                    "WHERE m.group_id=? AND (m.status IS NULL OR m.status != 'rejected')",
                    (group_id,)
                ).fetchall()
                classes = {r[0] for r in fallback_rows if r[0]}
                
        return classes

    def close(self) -> None:
        with self._lock:
            self._conn.close()

