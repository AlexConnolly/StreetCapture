"""DATA VIEW — offline summary of the local store.

Reads data/tracks.jsonl + data/events.jsonl and prints counts, durations, and
recent events. Pure stdlib so it runs anywhere; uses pandas only if available.

    python scripts/report.py [--data-dir data]
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path


def read_jsonl(path: Path):
    if not path.exists():
        return []
    with open(path, encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--data-dir", default="data")
    a = p.parse_args()
    data = Path(a.data_dir)

    tracks = read_jsonl(data / "tracks.jsonl")
    events = read_jsonl(data / "events.jsonl")

    print("=" * 48)
    print(f"STREETCAPTURE DATA VIEW  ({data})")
    print("=" * 48)
    print(f"Completed tracks : {len(tracks)}")
    print(f"Events logged    : {len(events)}")

    if tracks:
        by_class = Counter(t["class"] for t in tracks)
        print("\nTracks by class:")
        for cls, n in by_class.most_common():
            durs = [t["duration"] for t in tracks if t["class"] == cls]
            print(f"  {cls:<12} {n:>4}   avg {sum(durs) / len(durs):5.1f}s   max {max(durs):5.1f}s")

    if events:
        by_type = Counter(e["type"] for e in events)
        print("\nEvents by type:")
        for et, n in by_type.most_common():
            print(f"  {et:<16} {n:>4}")

        print("\nLast 10 events:")
        for e in events[-10:]:
            extra = f" {e['duration']:.0f}s" if "duration" in e else ""
            print(f"  {e['type']:<16} #{e['track_id']:<4} {e.get('class', '')}{extra}")


if __name__ == "__main__":
    main()
