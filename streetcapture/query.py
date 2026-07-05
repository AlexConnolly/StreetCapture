"""Query engine — turns questions into answers over the artifact database.

This is the v1 product layer: a *rule-based* parser (time range + label filter +
intent), not an LLM. The "natural language querying layer (LLM over DB)" is
explicitly a v2 item; this engine gives real answers now and is the interface an
LLM would later target.

    python -m streetcapture.query "how many vehicles passed yesterday?"
    python -m streetcapture.query            # interactive

Handles: counts, "when did X happen", quietest/busiest time (density histogram),
"how often", and list/show — filtered by time and by taxonomy labels. Filters on
emergent labels (company/function like "DPD"/"delivery") are accepted but return
a note that identity clustering lands in v2.
"""

from __future__ import annotations

import argparse
import re
import sqlite3
import time
from datetime import datetime, timedelta

from .config import Config
from .taxonomy import VEHICLE_CLASSES

WEEKDAYS = ["monday", "tuesday", "wednesday", "thursday", "friday", "saturday", "sunday"]
PERSON_WORDS = ("foot traffic", "pedestrian", "people", "person", "walker", "on foot")
VEHICLE_WORDS = ("vehicle", "traffic", "car", "van", "truck", "lorry", "bus")
BIKE_WORDS = ("bike", "bicycle", "cyclist", "motorbike", "motorcycle")
# Emergent (v2) label references we recognise but can't yet resolve from data.
EMERGENT = {
    "dpd": ("company", "DPD"), "amazon": ("company", "Amazon"),
    "royal mail": ("company", "Royal Mail"), "delivery": ("function", "delivery"),
    "bin lorry": ("function", "waste collection"), "bin lorries": ("function", "waste collection"),
    "electric": ("energy", "electric"),
}


def _day_bounds(dt: datetime):
    start = dt.replace(hour=0, minute=0, second=0, microsecond=0)
    return start.timestamp(), (start + timedelta(days=1)).timestamp()


def parse_time_range(q: str, now: float):
    """Return (start_ts, end_ts, human_label). Falls back to all-time."""
    ndt = datetime.fromtimestamp(now)
    lo = hi = None
    label = "all time"

    if "yesterday" in q:
        lo, hi = _day_bounds(ndt - timedelta(days=1)); label = "yesterday"
    elif "today" in q:
        lo, hi = _day_bounds(ndt); label = "today"
    elif "last week" in q:
        this_mon = (ndt - timedelta(days=ndt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        last_mon = this_mon - timedelta(days=7)
        lo, hi, label = last_mon.timestamp(), this_mon.timestamp(), "last week"
    elif "this week" in q:
        mon = (ndt - timedelta(days=ndt.weekday())).replace(hour=0, minute=0, second=0, microsecond=0)
        lo, hi, label = mon.timestamp(), (mon + timedelta(days=7)).timestamp(), "this week"
    else:
        for i, wd in enumerate(WEEKDAYS):
            if wd in q:
                # most recent past occurrence of that weekday
                delta = (ndt.weekday() - i) % 7 or 7
                day = ndt - timedelta(days=delta)
                lo, hi = _day_bounds(day)
                label = f"last {wd.capitalize()}"
                break

    # Optional hour window e.g. "between 8-10am", "8am to 10am", "8 and 10"
    m = re.search(r"(\d{1,2})\s*(am|pm)?\s*(?:-|to|and|–|until)\s*(\d{1,2})\s*(am|pm)?", q)
    if m:
        h1 = int(m.group(1)) % 12 + (12 if m.group(2) == "pm" else 0)
        h2 = int(m.group(3)) % 12 + (12 if (m.group(4) or m.group(2)) == "pm" else 0)
        base = datetime.fromtimestamp(lo) if lo else ndt.replace(hour=0, minute=0, second=0, microsecond=0)
        lo = base.replace(hour=h1).timestamp()
        hi = base.replace(hour=h2).timestamp()
        label = f"{label} {h1:02d}:00–{h2:02d}:00" if label != "all time" else f"{h1:02d}:00–{h2:02d}:00 today"
    return lo, hi, label


def parse_filter(q: str):
    """Return (sql_where_fragment, params, description, emergent_note)."""
    where, params, desc, note = [], [], [], None

    for key, (ltype, lval) in EMERGENT.items():
        if key in q:
            note = (f"'{lval}' is a {ltype} label — identity/clustering resolution is a v2 "
                    f"feature, so this isn't populated yet; showing the closest physical match.")
            if ltype == "function" and lval == "waste collection":
                where.append("a.primary_class = 'truck'"); desc.append("bin lorry (≈truck)")
            elif ltype == "company":
                where.append("a.primary_class = 'truck'"); desc.append(f"{lval} (≈van/truck)")
            elif ltype == "function" and lval == "delivery":
                where.append("a.primary_class IN ('truck','car')"); desc.append("delivery (≈van/truck)")
            break

    if not where:
        if any(w in q for w in PERSON_WORDS):
            where.append("a.primary_class = 'person'"); desc.append("people")
        elif any(w in q for w in BIKE_WORDS):
            where.append("a.primary_class IN ('bicycle','motorcycle')"); desc.append("bikes")
        elif any(w in q for w in VEHICLE_WORDS):
            vlist = ",".join(f"'{c}'" for c in sorted(VEHICLE_CLASSES))
            where.append(f"a.primary_class IN ({vlist})"); desc.append("vehicles")

    frag = (" AND " + " AND ".join(where)) if where else ""
    return frag, params, " ".join(desc) or "objects", note


def detect_intent(q: str) -> str:
    if any(w in q for w in ("quietest", "least busy", "quiet")):
        return "quietest"
    if any(w in q for w in ("busiest", "peak", "most busy")):
        return "busiest"
    if any(w in q for w in ("how many", "how much", "count", "number of")):
        return "count"
    if any(w in q for w in ("how often", "how frequently", "frequency")):
        return "howoften"
    if any(w in q for w in ("show", "list", "which", "all ")):
        return "list"
    if q.strip().startswith("when") or "what time" in q:
        return "when"
    return "count"


class QueryEngine:
    def __init__(self, db_path):
        self.conn = sqlite3.connect(str(db_path))
        self.conn.row_factory = sqlite3.Row

    def _rows(self, frag, lo, hi):
        where = "1=1" + frag
        params = []
        if lo is not None:
            where += " AND a.start_time >= ? AND a.start_time < ?"; params += [lo, hi]
        return self.conn.execute(
            f"SELECT a.* FROM artifacts a WHERE {where} ORDER BY a.start_time", params
        ).fetchall()

    def answer(self, q: str) -> str:
        q = q.lower().strip()
        if not q:
            return "Ask me something like: 'how many vehicles passed yesterday?'"
        now = time.time()
        lo, hi, tlabel = parse_time_range(q, now)
        frag, _, desc, note = parse_filter(q)
        intent = detect_intent(q)
        rows = self._rows(frag, lo, hi)

        if intent in ("quietest", "busiest"):
            ans = self._density(rows, desc, tlabel, intent)
        elif intent == "when":
            ans = self._when(rows, desc, tlabel)
        elif intent == "howoften":
            ans = self._howoften(rows, desc, tlabel, lo, hi)
        elif intent == "list":
            ans = self._list(rows, desc, tlabel)
        else:
            ans = f"{len(rows)} {desc} recorded ({tlabel})."
        return ans + (f"\n  note: {note}" if note else "")

    def _density(self, rows, desc, tlabel, which):
        if not rows:
            return f"No {desc} recorded ({tlabel}) — not enough data to judge."
        hours = [0] * 24
        for r in rows:
            hours[datetime.fromtimestamp(r["start_time"]).hour] += 1
        active = [(h, c) for h, c in enumerate(hours) if c > 0]
        pick = min(active, key=lambda x: x[1]) if which == "quietest" else max(active, key=lambda x: x[1])
        word = "quietest" if which == "quietest" else "busiest"
        return (f"The {word} hour for {desc} ({tlabel}) is "
                f"{pick[0]:02d}:00–{pick[0] + 1:02d}:00 with {pick[1]} recorded.")

    def _when(self, rows, desc, tlabel):
        if not rows:
            return f"No {desc} found ({tlabel})."
        times = [datetime.fromtimestamp(r["start_time"]).strftime("%a %H:%M:%S") for r in rows[:12]]
        more = f" (+{len(rows) - 12} more)" if len(rows) > 12 else ""
        return f"{desc.capitalize()} seen ({tlabel}) at: " + ", ".join(times) + more

    def _howoften(self, rows, desc, tlabel, lo, hi):
        n = len(rows)
        if lo is not None and hi is not None:
            days = max(1, (hi - lo) / 86400)
            return f"{desc.capitalize()} appeared {n} times ({tlabel}) ≈ {n / days:.1f}/day."
        return f"{desc.capitalize()} appeared {n} times (all recorded time)."

    def _list(self, rows, desc, tlabel):
        if not rows:
            return f"No {desc} found ({tlabel})."
        lines = [f"{len(rows)} {desc} ({tlabel}):"]
        for r in rows[:20]:
            t = datetime.fromtimestamp(r["start_time"]).strftime("%a %H:%M:%S")
            lines.append(f"  #{r['id']:<5} {r['primary_class']:<8} {t}  {r['duration']:.1f}s")
        if len(rows) > 20:
            lines.append(f"  … +{len(rows) - 20} more")
        return "\n".join(lines)

    def close(self):
        self.conn.close()


def main(argv=None):
    p = argparse.ArgumentParser(prog="streetcapture.query")
    p.add_argument("question", nargs="*", help="natural-language question")
    p.add_argument("--db", default=str(Config().db_path))
    a = p.parse_args(argv)
    from pathlib import Path
    if not Path(a.db).exists():
        print(f"No database at {a.db}. Run the live system first.")
        return
    eng = QueryEngine(a.db)
    try:
        if a.question:
            print(eng.answer(" ".join(a.question)))
        else:
            print("StreetCapture query — ask a question (blank line to quit).")
            print("e.g. 'how many vehicles passed yesterday?'  /  'quietest time for foot traffic?'")
            while True:
                try:
                    q = input("\n> ").strip()
                except EOFError:
                    break
                if not q:
                    break
                print(eng.answer(q))
    finally:
        eng.close()


if __name__ == "__main__":
    main()
