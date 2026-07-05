# StreetCapture — v0.2 (Identity & Artifact Foundation)

A local **dual-speed perception engine** for an IP camera.

- **Fast loop (~5 FPS)** — what is happening *now*: RTSP → YOLOv8-nano detection → ByteTrack IDs → live overlay.
- **Slow loop (~1–2 FPS)** — what it *means over time*: it accumulates candidate crops per track, and when a track **completes** it decides whether the track is *meaningful*. Meaningful tracks become **Artifacts** — representative images + quality scores + a CLIP embedding, persisted to a local SQLite "memory" database.

```
RTSP ─▶ FrameGrabber ─▶ [LIVE: YOLO+ByteTrack → overlay]        (main thread, ~5 FPS)
                    └──▶ shared state ──▶ [ARTIFACT loop]        (thread, ~1–2 FPS)
                                             ├─ representative images
                                             ├─ quality scores
                                             ├─ OpenCLIP embedding
                                             └─ SQLite artifact DB
```

Detection runs **once** per frame and is shared with the artifact loop — one YOLO pass, easy on a 6–8 GB GPU. Embeddings run only when a track *ends*, off the live path.

## Track ID vs Entity ID

A **Track ID** is one continuous observation (ByteTrack). It is never reused, and ByteTrack IDs restart each run, so the permanent identifier is the DB primary key. An **Entity ID** (persistent identity across many tracks — "probably the same van") is **not implemented yet** — the schema reserves a nullable `entity_id` for v0.3+. Keeping these separate now is the whole point of v0.2.

## Install

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

`yolov8n.pt` and the OpenCLIP weights download automatically on first use. CUDA torch is used if present (CPU works, slower).

## Run

```bash
python -m streetcapture                                             # webcam (default)
python -m streetcapture --source "rtsp://user:pass@192.168.1.50:554/stream1"   # Tapo
python -m streetcapture --source clip.mp4 --headless                # data only, no windows
```

Windows: **LIVE VIEW** (boxes, class, track ID, plus per-track age / size / *ARTIFACT PENDING*) and **ARTIFACT VIEW** (active tracks, artifacts today, recent events). Press **`q`** to quit.

Browse what it remembered:

```bash
python -m streetcapture.viewer      # http://127.0.0.1:8000  — scrollable artifact gallery
python scripts/report.py            # text summary of the database
```

Flags: `--source --model --device {"",cpu,0} --conf --live-fps --artifact-fps --no-embed --no-live --no-dashboard --headless`. Every setting also has a `STREETCAPTURE_*` env var (thresholds, rep-image counts, embedding model — see `streetcapture/config.py`).

## Data / memory layer

```
artifacts/
  artifact.db        SQLite: sessions, tracks, artifacts, artifact_images, embeddings, events
  images/            representative crops, NNNNNN_<rank>.jpg
```

Each **Artifact** stores: class, start/end/duration, avg confidence, sharpness, visibility, motion distance, track length, motion path, 3–10 representative images, a 512-d OpenCLIP embedding (model-versioned), and a reserved `entity_id`.

**Events:** `track_started`, `track_ended`, `artifact_created`, `artifact_rejected` (with reason) — full traceability of every decision.

## Scope

**In v0.2:** Track→Artifact pipeline with quality gating, Track-ID/Entity-ID separation, representative-image selection, OpenCLIP embedding per artifact, SQLite memory DB, web artifact browser, quality scoring, richer events.

**Explicitly deferred (v0.3+):** similarity / nearest-neighbour search, clustering (HDBSCAN), entity matching, semantic labels (DPD van, bin lorry…), temporal prediction, natural-language search, multi-camera.

> v0.1 → *a system that observes*. v0.2 → *a system that remembers*. Reliable memory first; similarity, clustering and semantics build on top of it.
