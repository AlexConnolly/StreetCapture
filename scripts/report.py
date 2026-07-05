"""DATA VIEW — offline summary of the artifact database.

    python scripts/report.py [--db artifacts/artifact.db]
"""

from __future__ import annotations

import argparse
import sqlite3
from collections import Counter
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--db", default="artifacts/artifact.db")
    a = p.parse_args()
    path = Path(a.db)
    if not path.exists():
        print(f"No database at {path}. Run the live system first.")
        return

    conn = sqlite3.connect(str(path))
    conn.row_factory = sqlite3.Row

    def scalar(q):
        return conn.execute(q).fetchone()[0]

    print("=" * 52)
    print(f"STREETCAPTURE DATA VIEW  ({path})")
    print("=" * 52)
    print(f"Sessions    : {scalar('SELECT COUNT(*) FROM sessions')}")
    print(f"Tracks      : {scalar('SELECT COUNT(*) FROM tracks')}")
    print(f"Artifacts   : {scalar('SELECT COUNT(*) FROM artifacts')}")
    print(f"Embeddings  : {scalar('SELECT COUNT(*) FROM embeddings')}")
    print(f"Events      : {scalar('SELECT COUNT(*) FROM events')}")

    arts = conn.execute("SELECT primary_class, duration FROM artifacts").fetchall()
    if arts:
        by = Counter(r["primary_class"] for r in arts)
        print("\nArtifacts by class:")
        for cls, n in by.most_common():
            durs = [r["duration"] for r in arts if r["primary_class"] == cls]
            print(f"  {cls:<12} {n:>4}   avg {sum(durs) / len(durs):5.1f}s   max {max(durs):5.1f}s")

    ev = conn.execute("SELECT type, COUNT(*) c FROM events GROUP BY type ORDER BY c DESC").fetchall()
    if ev:
        print("\nEvents by type:")
        for r in ev:
            print(f"  {r['type']:<18} {r['c']:>4}")

    models = conn.execute(
        "SELECT model_version, COUNT(*) c FROM embeddings GROUP BY model_version").fetchall()
    if models:
        print("\nEmbedding models:")
        for r in models:
            print(f"  {r['model_version']:<32} {r['c']:>4}")

    labs = conn.execute(
        "SELECT type, value, COUNT(*) c FROM labels GROUP BY type, value ORDER BY type, c DESC").fetchall()
    if labs:
        print("\nLabels:")
        for r in labs:
            print(f"  {r['type']:<9} {r['value']:<20} {r['c']:>4}")

    recent = conn.execute(
        "SELECT id, primary_class, duration, avg_confidence, sharpness "
        "FROM artifacts ORDER BY id DESC LIMIT 10").fetchall()
    if recent:
        print("\nMost recent artifacts:")
        for r in recent:
            print(f"  #{r['id']:<5} {r['primary_class']:<10} {r['duration']:5.1f}s "
                  f"conf {r['avg_confidence']:.2f}  sharp {r['sharpness']:.0f}")

    conn.close()


if __name__ == "__main__":
    main()
