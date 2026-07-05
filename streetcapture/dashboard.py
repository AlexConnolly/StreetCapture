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


def draw_live(frame, tracks, fps):
    """Boxes + class + track ID + FPS counter on top of the live frame."""
    for t in tracks:
        x1, y1, x2, y2 = (int(v) for v in t["bbox"])
        color = CAT_COLOR[_cat(t["class"])]
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        label = f"{t['class']} #{t['track_id']} {t['confidence']:.2f}"
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        cv2.rectangle(frame, (x1, y1 - th - 6), (x1 + tw + 4, y1), color, -1)
        cv2.putText(frame, label, (x1 + 2, y1 - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1, cv2.LINE_AA)

    header = f"LIVE VIEW   {fps:4.1f} FPS   tracks: {len(tracks)}" if fps else "LIVE VIEW"
    cv2.rectangle(frame, (0, 0), (frame.shape[1], 26), (0, 0, 0), -1)
    cv2.putText(frame, header, (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    return frame


def draw_dashboard(snap, w=420, h=520):
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
         dy=32, color=(160, 160, 160), scale=0.5)

    line(f"Today ({snap['day']}):", dy=26, color=(255, 255, 255))
    d = snap["daily"]
    line(f"  persons:  {d.get('person', 0)}", dy=22, color=(0, 200, 0))
    line(f"  vehicles: {d.get('vehicle', 0)}", dy=22, color=(0, 160, 255))
    line(f"  other:    {d.get('other', 0)}", dy=32, color=(200, 200, 200))

    line("Recent Events:", dy=26, color=(255, 255, 255))
    if not snap["events"]:
        line("  (none yet)", dy=22, color=(120, 120, 120), scale=0.5)
    for ev in snap["events"]:
        line("  " + ev, dy=22, color=(200, 200, 200), scale=0.48)

    cv2.putText(img, "press  q  to quit", (x, h - 16),
                cv2.FONT_HERSHEY_SIMPLEX, 0.45, (110, 110, 110), 1, cv2.LINE_AA)
    return img
