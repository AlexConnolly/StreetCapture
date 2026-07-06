"""Background / idle-object suppression.

A fixed camera sees permanent fixtures (a potted plant, street furniture YOLO
misreads as an "oven", a car that's always parked there) constantly — they flood
the system with boxes and artifacts that carry no information.

This models the scene at the detection level: an object that stays in the same
place (high bbox overlap) with sub-jitter motion for longer than
``background_seconds`` is declared BACKGROUND and filtered out (no box, no
artifact). Any real motion resets its timer, so:

* potted plant  -> motionless forever -> background (suppressed).
* car drives in  -> moving -> foreground; parks -> after background_seconds it
  goes quiet; drives off -> motion -> foreground again (departure flagged).

Keyed by location + category (not track id) so it survives the detection flicker
that splits a static object across many track ids.
"""

from __future__ import annotations

from .taxonomy import category


def _iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    ix1, iy1 = max(ax1, bx1), max(ay1, by1)
    ix2, iy2 = min(ax2, bx2), min(ay2, by2)
    iw, ih = max(0.0, ix2 - ix1), max(0.0, iy2 - iy1)
    inter = iw * ih
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


class BackgroundModel:
    def __init__(self, cfg):
        self.cfg = cfg
        self.entries = []   # [{bbox,cx,cy,cat,static_since,last,is_bg}, ...]

    def filter(self, tracks, now):
        """Return (foreground_tracks, idle_count). Idle/background tracks are
        dropped so they neither draw a box nor become artifacts."""
        if not self.cfg.background_suppress:
            return tracks, 0
        dz = self.cfg.movement_deadzone_frac
        used = set()
        keep = []
        idle = 0
        for t in tracks:
            x1, y1, x2, y2 = t["bbox"]
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            diag = max(1.0, ((x2 - x1) ** 2 + (y2 - y1) ** 2) ** 0.5)
            cat = category(t["class"])
            best, best_iou = None, 0.4
            for e in self.entries:
                if id(e) in used or e["cat"] != cat:
                    continue
                iou = _iou((x1, y1, x2, y2), e["bbox"])
                if iou >= best_iou:
                    best, best_iou = e, iou
            if best is None:
                best = {"bbox": (x1, y1, x2, y2), "cx": cx, "cy": cy, "cat": cat,
                        "static_since": now, "last": now, "is_bg": False}
                self.entries.append(best)
            else:
                disp = ((cx - best["cx"]) ** 2 + (cy - best["cy"]) ** 2) ** 0.5 / diag
                if disp > dz:                       # it moved -> not background
                    best["static_since"] = now
                    best["is_bg"] = False
                best["bbox"] = (x1, y1, x2, y2)
                best["cx"], best["cy"] = cx, cy
                best["last"] = now
                if now - best["static_since"] >= self.cfg.background_seconds:
                    best["is_bg"] = True
            used.add(id(best))
            if best["is_bg"]:
                idle += 1
            else:
                keep.append(t)
        # forget locations that have been clear for a while (so re-arrivals are new)
        self.entries = [e for e in self.entries
                        if now - e["last"] <= self.cfg.background_forget_seconds]
        return keep, idle
