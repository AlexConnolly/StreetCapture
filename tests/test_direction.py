"""Travel-direction feature: compute_direction, migration backfill, and the
query-engine direction filter/summary. Pure-stdlib (no cv2/torch)."""

from __future__ import annotations

import json

import pytest

from streetcapture.db import Database, compute_direction
from streetcapture.query import QueryEngine, parse_direction, detect_intent


@pytest.mark.parametrize("path,label", [
    ([[10, 100], [300, 100]], "right"),
    ([[300, 100], [10, 100]], "left"),
    ([[100, 10], [100, 300]], "down"),      # screen y increases downward
    ([[100, 300], [100, 10]], "up"),
    ([[0, 0], [200, 200]], "down-right"),
    ([[200, 200], [0, 0]], "up-left"),
    ([[100, 100], [104, 98]], "stationary"),
])
def test_compute_direction_labels(path, label):
    assert compute_direction(path)[2] == label


def test_compute_direction_needs_two_points():
    assert compute_direction([[1, 1]]) == (None, None, None)
    assert compute_direction([]) == (None, None, None)


def test_migration_backfills_direction(tmp_path):
    db = Database(tmp_path / "t.db")
    aid = db.insert_artifact({"primary_class": "person", "start_time": 1.0, "created_at": 0.0,
                              "motion_path_json": json.dumps([[10, 50], [400, 60]])})
    db._conn.close()
    # Re-open: _migrate backfills direction from the stored motion path.
    db2 = Database(tmp_path / "t.db")
    row = db2._conn.execute("SELECT direction FROM artifacts WHERE id=?", (aid,)).fetchone()
    assert row[0] == "right"


def test_parse_direction_requires_motion_cue():
    assert parse_direction("people moving left")[2] == "moving left"
    assert parse_direction("cars going right")[2] == "moving right"
    assert parse_direction("right now the street is busy") == ("", [], None)


def test_direction_intent_and_summary(tmp_path):
    db = Database(tmp_path / "t.db")
    for path in ([[0, 50], [400, 50]], [[0, 50], [400, 60]], [[400, 50], [0, 50]]):
        db.insert_artifact({"primary_class": "person", "start_time": 1.0, "created_at": 0.0,
                            "motion_path_json": json.dumps(path)})
    # populate direction (fresh reopen backfills)
    db._conn.close()
    Database(tmp_path / "t.db")._conn.close()

    assert detect_intent("which way were people going") == "direction"
    qe = QueryEngine(tmp_path / "t.db")
    ans = qe.answer("which way were people going")
    assert "right" in ans and "2" in ans          # 2 of 3 went right
    assert "1 people moving left" in qe.answer("how many people moving left")
