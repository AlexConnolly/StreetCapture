"""Rendering for both windows: LIVE VIEW overlay and ARTIFACT VIEW dashboard.

Both are drawn with OpenCV so the whole system needs no extra GUI dependency.
"""

from __future__ import annotations

import cv2
import numpy as np

# Stable per-category colours (BGR).
CAT_COLOR = {"person": (0, 200, 0), "vehicle": (0, 160, 255), "other": (200, 200, 200)}
VEHICLE_CLASSES = {"car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "train"}


def _cat(cls_name: str) -> str:
    if cls_name == "person":
        return "person"
    if cls_name in VEHICLE_CLASSES:
        return "vehicle"
    return "other"


def draw_live(frame, tracks, fps, live_meta=None, track_labels=None):
    """Boxes + class + track ID + per-track analysis state (age / size / artifact-pending).
    If a track matches a taught label ('Santander Bike'), that's shown on top."""
    live_meta = live_meta or {}
    track_labels = track_labels or {}
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        lab = track_labels.get(t["track_id"], {})
        taught = lab.get("label")
        # A recognised taught label gets a bright cyan box; otherwise category colour.
        color = (255, 220, 0) if taught else CAT_COLOR[_cat(t["class"])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

        meta = live_meta.get(t["track_id"], {})
        pending = meta.get("pending", False)
        # Line 1: the taught label if we have one, else class + id + confidence.
        if taught:
            l1 = f"{taught} {lab.get('score', 0):.0%}"
        else:
            l1 = f"{t['class']} #{t['track_id']} {t['confidence']:.2f}"
        size = f"{x2 - x1}x{y2 - y1}"
        age = meta.get("age")
        state = "ARTIFACT PENDING" if pending else "analysing"
        l2 = f"age {age:.0f}s  {size}  {state}" if age is not None else f"{size}"

        (tw, th), _ = cv2.getTextSize(l1, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        (tw2, _), _ = cv2.getTextSize(l2, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
        box_w = max(tw, tw2) + 6
        cv2.rectangle(frame, (x1, y1 - th - 22), (x1 + box_w, y1), color, -1)
        cv2.putText(frame, l1, (x1 + 3, y1 - 12),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
        l2_col = (0, 0, 160) if pending else (40, 40, 40)
        cv2.putText(frame, l2, (x1 + 3, y1 - 2),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, l2_col, 1, cv2.LINE_AA)

    header = f"LIVE VIEW   {fps:4.1f} FPS   tracks: {len(tracks)}" if fps else "LIVE VIEW"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(frame, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_dashboard(snap, w=420, h=560):
    """Render the ARTIFACT VIEW panel from a dashboard snapshot dict."""
    img = np.full((h, w, 3), 26, np.uint8)
    x = 18
    y = [36]  # mutable cursor

    def line(text, dy=26, color=(230, 230, 230), scale=0.55, thick=1):
        cv2.putText(img, text, (x, y[0]), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)
        y[0] += dy

    line("ARTIFACT VIEW", dy=34, color=(0, 200, 255), scale=0.75, thick=2)
    line(f"Active Tracks: {snap['active']}", dy=22, color=(255, 255, 255))
    by = snap["active_by_cat"]
    line(f"  person {by.get('person', 0)}   vehicle {by.get('vehicle', 0)}   other {by.get('other', 0)}",
         dy=30, color=(160, 160, 160), scale=0.5)

    a = snap.get("artifacts", {})
    total_art = sum(a.values())
    line(f"Artifacts today: {total_art}", dy=22, color=(255, 255, 255))
    line(f"  person {a.get('person', 0)}   vehicle {a.get('vehicle', 0)}   other {a.get('other', 0)}",
         dy=30, color=(0, 200, 255), scale=0.5)

    line(f"Seen today ({snap['day']}):", dy=24, color=(255, 255, 255))
    d = snap["daily"]
    line(f"  persons {d.get('person', 0)}   vehicles {d.get('vehicle', 0)}   other {d.get('other', 0)}",
         dy=30, color=(200, 200, 200), scale=0.5)

    line("Recent Events:", dy=24, color=(255, 255, 255))
    if not snap["events"]:
        line("  (none yet)", dy=22, color=(120, 120, 120), scale=0.5)
    for ev in snap["events"]:
        col = (0, 200, 255) if "ARTIFACT #" in ev else (200, 200, 200)
        line("  " + ev, dy=21, color=col, scale=0.46)

    cv2.putText(img, "press  q  to quit    (browse: python -m streetcapture.viewer)",
                (x, h - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (110, 110, 110), 1, cv2.LINE_AA)
    return img
