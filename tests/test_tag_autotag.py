"""Regression tests for the tag auto-tagging bugs.

Two distinct defects made key/value tags (e.g. "gender: male") stop tagging NEW
artifacts:

1. GroupService.tag_artifacts created the labeled group but never recomputed its
   centroid, so it kept a NULL centroid. labeled_group_centroids() filters
   `WHERE centroid IS NOT NULL`, so the group never entered `_labeled` and the
   `_auto_tag` loop never considered it. (test_mean_centroid_path)

2. Once a group had >=5 members, _recompute_centroid trained a LogisticRegression
   and stored centroid = _norm(w) — a discriminative hyperplane NORMAL. But the
   matchers gated on `v . centroid >= 0.72`, a threshold calibrated for a
   cosine-to-mean centroid. Dot products against the LR normal are ~0..0.3, so
   0.72 was never met and nothing got tagged. The fix calibrates and stores a
   per-group decision threshold. (test_lr_discriminative_path)

These exercise the real DB layer with a stub embedder (the tag / auto-tag paths
operate on stored vectors, not the embedder).
"""

from __future__ import annotations

import time

import numpy as np
import pytest

from streetcapture.config import Config
from streetcapture.db import Database
from streetcapture.server.groups import GroupService


class _StubEmbedder:
    def embed(self, *_a, **_k):
        return None

    def embed_text(self, *_a, **_k):
        return None


def _vec(*components, dim=16):
    v = np.zeros(dim, dtype="float32")
    for i, c in enumerate(components):
        v[i] = c
    return v


def _add_artifact(db: Database, cls: str, vec) -> int:
    now = time.time()
    aid = db.insert_artifact({
        "primary_class": cls,
        "start_time": now,
        "end_time": now,
        "created_at": now,
    })
    db.insert_embedding(aid, list(map(float, vec)), model_version="test")
    return aid


def _tag_group_id(db: Database, key: str, value: str):
    row = db._conn.execute(
        "SELECT id, centroid, match_threshold FROM groups WHERE tag_key=? AND tag_value=?",
        (key, value),
    ).fetchone()
    return row


@pytest.fixture()
def service(tmp_path):
    db = Database(tmp_path / "artifact.db")
    gs = GroupService(Config(), db, _StubEmbedder(), vectorstore=None, reid=None)
    return gs, db


def test_mean_centroid_path(service):
    """Few members + no negatives -> mean centroid, NULL match_threshold, and the
    group must still enter _labeled and auto-tag a matching new artifact."""
    gs, db = service
    males = [_add_artifact(db, "person", _vec(1.0, 0.0)),
             _add_artifact(db, "person", _vec(0.98, 0.10)),
             _add_artifact(db, "person", _vec(0.98, -0.10))]

    gs.tag_artifacts(males, [{"key": "gender", "value": "male"}])

    row = _tag_group_id(db, "gender", "male")
    assert row is not None, "tag group was not created"
    gid, centroid, mthr = row
    assert centroid is not None, "tag group centroid is NULL -> auto-tagger skips it"
    assert gid in [g[0] for g in gs._labeled], "tag group missing from _labeled"

    new_vec = _vec(0.97, 0.05, 0.02)
    new_id = _add_artifact(db, "person", new_vec)
    gs.on_new_artifact(new_id, new_vec, "person")
    assert new_id in db.group_members(gid), "new matching artifact was not auto-tagged"


def test_lr_discriminative_path(service):
    """>=5 members + same-class background -> discriminative LR centroid with a
    calibrated threshold. A matching new artifact must be tagged; a clearly
    different same-class one must not."""
    gs, db = service
    # Six "male" crops near axis-0.
    males = [_add_artifact(db, "person", _vec(1.0, 0.0)),
             _add_artifact(db, "person", _vec(0.97, 0.12)),
             _add_artifact(db, "person", _vec(0.97, -0.12)),
             _add_artifact(db, "person", _vec(0.95, 0.20)),
             _add_artifact(db, "person", _vec(0.95, -0.20)),
             _add_artifact(db, "person", _vec(0.99, 0.05))]
    # Six "female" crops near axis-1 as same-class background (never tagged).
    for y in (1.0, 0.97, 0.95, 0.99, 0.96, 0.98):
        _add_artifact(db, "person", _vec(0.0, y))

    gs.tag_artifacts(males, [{"key": "gender", "value": "male"}])

    gid, centroid, mthr = _tag_group_id(db, "gender", "male")
    assert centroid is not None
    assert mthr is not None, "LR centroid must store a calibrated match_threshold"

    # A new male-like artifact should be tagged.
    male_new = _vec(0.98, 0.08)
    mid = _add_artifact(db, "person", male_new)
    gs.on_new_artifact(mid, male_new, "person")
    assert mid in db.group_members(gid), "new male artifact was not auto-tagged"

    # A new female-like artifact must NOT be tagged as male.
    female_new = _vec(0.05, 0.98)
    fid = _add_artifact(db, "person", female_new)
    gs.on_new_artifact(fid, female_new, "person")
    assert fid not in db.group_members(gid), "female artifact wrongly tagged as male"


def _member(db, gid, aid):
    return db._conn.execute(
        "SELECT status, source FROM group_members WHERE group_id=? AND artifact_id=?",
        (gid, aid)).fetchone()


def test_below_confidence_bar_queues_pending(service):
    """A group under the confidence bar leaves new matches as pending 'auto'
    suggestions for the user to verify."""
    gs, db = service
    males = [_add_artifact(db, "person", _vec(1.0, 0.0)),
             _add_artifact(db, "person", _vec(0.98, 0.10)),
             _add_artifact(db, "person", _vec(0.98, -0.10))]  # 3 confirmed < bar(10)
    gs.tag_artifacts(males, [{"key": "gender", "value": "male"}])
    gid = _tag_group_id(db, "gender", "male")[0]

    new_vec = _vec(0.97, 0.05)
    nid = _add_artifact(db, "person", new_vec)
    gs.on_new_artifact(nid, new_vec, "person")

    status, source = _member(db, gid, nid)
    assert status is None and source == "auto", \
        f"below the bar a match should be pending 'auto', got ({status!r},{source!r})"


def test_confident_group_auto_classifies(service):
    """>=10 human confirmations -> matches are auto-classified (confirmed,
    source auto_confirm) with no verify queue; non-matches are not added."""
    gs, db = service
    # 12 confirmed males (>= bar of 10), plus same-class background.
    males = [_add_artifact(db, "person", _vec(1.0, 0.02 * i - 0.1)) for i in range(12)]
    for y in (1.0, 0.97, 0.95, 0.99, 0.96, 0.98):
        _add_artifact(db, "person", _vec(0.0, y))
    gs.tag_artifacts(males, [{"key": "gender", "value": "male"}])
    gid = _tag_group_id(db, "gender", "male")[0]
    assert db.human_confirmed_count(gid) >= gs.cfg.auto_classify_min_confirmed

    # New male: auto-classified, not queued.
    male_new = _vec(0.98, 0.06)
    mid = _add_artifact(db, "person", male_new)
    gs.on_new_artifact(mid, male_new, "person")
    status, source = _member(db, gid, mid)
    assert status == "confirmed" and source == "auto_confirm", \
        f"confident group should auto-classify, got ({status!r},{source!r})"

    # Auto-classified members must NOT inflate the human confidence count.
    assert db.human_confirmed_count(gid) == len(males)

    # New female: below threshold -> not added at all (not queued).
    female_new = _vec(0.05, 0.98)
    fid = _add_artifact(db, "person", female_new)
    gs.on_new_artifact(fid, female_new, "person")
    assert fid not in db.group_members(gid)


def test_recluster_excludes_labeled_artifacts(service):
    """Artifacts confirmed into a labeled group are excluded from the suggestion
    clustering pool, so they aren't re-proposed."""
    gs, db = service
    males = [_add_artifact(db, "person", _vec(1.0, 0.0)),
             _add_artifact(db, "person", _vec(0.98, 0.1))]
    ungrouped = _add_artifact(db, "person", _vec(0.0, 1.0))
    gs.tag_artifacts(males, [{"key": "gender", "value": "male"}])

    pool = {r[0] for r in db.embeddings_for_clustering()}
    assert ungrouped in pool
    assert not (set(males) & pool), "labeled-confirmed artifacts leaked into clustering"


def test_auto_classify_pending(service):
    """'Auto-classify remaining' lets the model decide the backlog: a match keeps
    the tag (confirmed, source auto_confirm -> not training); a non-match is
    dropped and does NOT get the tag. Neither retrains the centroid."""
    gs, db = service
    males = [_add_artifact(db, "person", _vec(1.0, 0.0)),
             _add_artifact(db, "person", _vec(0.98, 0.10)),
             _add_artifact(db, "person", _vec(0.98, -0.10))]
    gs.tag_artifacts(males, [{"key": "gender", "value": "male"}])
    gid = _tag_group_id(db, "gender", "male")[0]

    # Two un-reviewed suggestions: one clearly matches, one clearly doesn't.
    match = _add_artifact(db, "person", _vec(0.97, 0.05))
    nonmatch = _add_artifact(db, "person", _vec(0.0, 1.0))
    db.add_member(gid, match, 0.5, "auto")     # status NULL = pending
    db.add_member(gid, nonmatch, 0.5, "auto")

    r = gs.auto_classify_pending(gid)
    assert r["classified"] == 1 and r["dropped"] == 1, r

    # Match: tagged, but as auto_confirm so it doesn't train the model.
    assert _member(db, gid, match) == ("confirmed", "auto_confirm")
    # Non-match: dropped entirely, does not carry the tag.
    assert _member(db, gid, nonmatch) is None
    assert nonmatch not in db.group_members(gid)

    # Only the human-tagged males train the centroid.
    human = db.group_member_vectors(gid, status="confirmed", exclude_auto_confirm=True)
    assert len(human) == len(males)
    assert db.human_confirmed_count(gid) == len(males)


def test_reject_sweep_untags_machine_lookalikes(service):
    """The reject-sweep re-scores MACHINE-classified members and untags ones that
    no longer clear the threshold — never touching human confirmations."""
    gs, db = service
    males = [_add_artifact(db, "person", _vec(1.0, 0.0)),
             _add_artifact(db, "person", _vec(0.98, 0.10)),
             _add_artifact(db, "person", _vec(0.98, -0.10))]
    gs.tag_artifacts(males, [{"key": "gender", "value": "male"}])
    gid = _tag_group_id(db, "gender", "male")[0]

    good = _add_artifact(db, "person", _vec(0.95, 0.05))   # on-model
    bad = _add_artifact(db, "person", _vec(0.0, 1.0))       # off-model (a woman)
    db.add_member(gid, good, 0.9, "auto_confirm", "confirmed")
    db.add_member(gid, bad, 0.9, "auto_confirm", "confirmed")
    gs._refresh_labeled()

    gs.reclassify_group(gid, retrain=False)

    assert bad not in db.group_members(gid), "off-model auto-tag was not swept"
    assert good in db.group_members(gid), "on-model auto-tag wrongly swept"
    assert set(males) <= set(db.group_members(gid)), "human confirmations were touched"


def test_resync_adds_and_removes_in_one_pass(service):
    """A manual edit re-derives the whole group: pull in a candidate that now
    matches AND drop a machine-tag that no longer does — both in one resync."""
    gs, db = service
    males = [_add_artifact(db, "person", _vec(1.0, 0.0)),
             _add_artifact(db, "person", _vec(0.98, 0.10)),
             _add_artifact(db, "person", _vec(0.98, -0.10))]
    gs.tag_artifacts(males, [{"key": "gender", "value": "male"}])
    gid = _tag_group_id(db, "gender", "male")[0]

    newmatch = _add_artifact(db, "person", _vec(0.96, 0.04))   # on-model, NOT yet a member
    stale = _add_artifact(db, "person", _vec(0.0, 1.0))         # off-model machine tag
    db.add_member(gid, stale, 0.9, "auto_confirm", "confirmed")
    gs._refresh_labeled()

    res = gs.resync_group(gid)

    assert newmatch in db.group_members(gid), "matching candidate was not pulled in"
    assert stale not in db.group_members(gid), "stale auto-tag was not swept"
    assert set(males) <= set(db.group_members(gid))
    assert res["added"] >= 1 and res["removed"] >= 1
