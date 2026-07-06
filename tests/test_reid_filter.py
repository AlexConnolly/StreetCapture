"""Dominant-person frame filter: keeps one artifact == one person after a track
ID-switch scatters multiple people across its frames. Pure (numpy only, no cv2)."""

from __future__ import annotations

import numpy as np

from streetcapture.reid import dominant_indices


def _norm(v):
    v = np.asarray(v, "float32")
    return (v / np.linalg.norm(v)).tolist()


# person A near axis-0, person B near axis-1 (different people ~0.2 cosine)
_A = [_norm([1.0, 0.05]), _norm([0.98, 0.10]), _norm([1.0, -0.08])]
_B = [_norm([0.10, 1.0]), _norm([0.05, 0.98])]


def test_drops_the_other_person():
    vecs = _A + _B                       # 0,1,2 = A ; 3,4 = B
    scores = [5, 9, 4, 8, 3]             # top score is an A crop
    assert dominant_indices(scores, vecs, 0.65) == [0, 1, 2]


def test_anchor_follows_the_highest_score():
    vecs = _A + _B
    scores = [4, 5, 3, 9, 6]             # top score is a B crop -> keep B cluster
    assert dominant_indices(scores, vecs, 0.65) == [3, 4]


def test_consistent_track_keeps_all():
    assert dominant_indices([1, 2, 3], _A, 0.65) == [0, 1, 2]


def test_empty():
    assert dominant_indices([], [], 0.65) == []
