# StreetCapture — v1.0

Turn a live RTSP camera into a **searchable, structured memory of the physical world at one location** — not surveillance, not just detection: a visual memory + query system.

- **Live perception (~5 FPS)** — RTSP → YOLOv8-nano → ByteTrack → live overlay.
- **Artifact pipeline (async, ~1–2 FPS)** — completed tracks become **Artifacts**: keyframes, quality stats, multi-label taxonomy, a CLIP embedding (SQLite + files + FAISS index).
- **Event + query engine** — artifacts become structured events, answerable in plain-ish English.

```
RTSP ─▶ FrameGrabber ─▶ [LIVE: YOLO+ByteTrack → overlay]          (main thread, ~5 FPS)
                    └──▶ shared state ──▶ [ARTIFACT loop]          (thread, ~1–2 FPS)
                                             ├─ keyframes + quality scores
                                             ├─ multi-label taxonomy
                                             ├─ OpenCLIP embedding ─▶ FAISS index
                                             └─ SQLite  ─▶  EVENT + QUERY ENGINE  ─▶  answers
```

Detection runs **once** per frame, shared to the artifact loop. Embeddings run only when a track ends — the live loop is never blocked.

## Track vs Artifact vs Entity

- **Track** — one continuous observation (ByteTrack). Ephemeral, never reused; ByteTrack IDs restart each run so the permanent id is the DB primary key.
- **Artifact** — a *meaningful* completed track, promoted through quality gating and stored as persistent memory.
- **Entity** — a persistent identity across many tracks ("the DPD van"). **Not implemented in v1**; the schema reserves a nullable `entity_id`. Emergent identity/clustering is v2.

## Install & run

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt

python -m streetcapture                                             # webcam (default)
python -m streetcapture --source "rtsp://user:pass@192.168.1.50:554/stream1"   # Tapo
python -m streetcapture --source clip.mp4 --headless               # data only
```

Model + CLIP weights download on first use; CUDA torch is used if present. Press **`q`** to quit.

## Ask it questions

```bash
python -m streetcapture.query "how many vehicles passed yesterday?"
python -m streetcapture.query "quietest time for foot traffic?"
python -m streetcapture.query                       # interactive
python -m streetcapture.viewer                      # http://127.0.0.1:8000 — gallery + query box
python scripts/report.py                            # text summary
```

The query engine parses a **time range** (today / yesterday / last <weekday> / last week / "between 8–10am"), a **label filter** (people / vehicles / bikes …), and an **intent** (count / when / quietest / busiest / how-often / list). It is deliberately rule-based — the **LLM-over-DB** layer is a v2 item. Queries that reference emergent labels ("DPD", "delivery", "bin lorry") are accepted but return a note that identity clustering lands in v2, plus the closest physical match.

## Data / memory layer

```
artifacts/
  artifact.db     SQLite: sessions, tracks, artifacts, artifact_images, embeddings, labels, events
  images/         representative keyframes, NNNNNN_<rank>.jpg
  faiss.index     FAISS inner-product index of artifact embeddings (built now; similarity search is v2)
```

**Multi-label taxonomy** (per artifact): `object` / `subtype` / `function` are populated deterministically from the class now; `company` / `energy` / other attributes are reserved for v2 (emergent via clustering).

**Events:** `track_started`, `track_ended`, `artifact_created`, `artifact_rejected` (+reason), plus derived `object_entered`, `object_left`, `object_stayed`, `vehicle_passed`.

## Scope

**In v1.0:** live perception, Track→Artifact pipeline with quality gating, keyframes, multi-label taxonomy, OpenCLIP embeddings, SQLite memory DB, FAISS embedding index, structured event engine, rule-based query engine (CLI + web), artifact viewer.

**Deferred to v2+:** embedding similarity search & auto-clustering (DPD / bin-lorry grouping), entity persistence, anomaly detection, the LLM natural-language query layer, multi-camera.
