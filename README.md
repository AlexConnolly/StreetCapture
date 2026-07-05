# StreetCapture — v0.1

A local **dual-speed perception engine** for an IP camera.

- **Fast loop (~5 FPS)** — what is happening *now*: RTSP → YOLOv8-nano detection → ByteTrack IDs → live overlay.
- **Slow loop (~1–2 FPS)** — what it *means over time*: per-track identity records, snapshots, and simple events, written to a local store.

```
RTSP ─▶ FrameGrabber ─▶ [LIVE: YOLO+ByteTrack → overlay]  (main thread, ~5 FPS)
                    └──▶ shared state ──▶ [ARTIFACT: records+events → JSONL]  (thread, ~1–2 FPS)
```

Detection runs **once** per frame and is shared with the artifact loop — one YOLO pass, easy on a 6 GB GPU.

## Install

```bash
python -m venv .venv && .venv\Scripts\activate      # Windows
pip install -r requirements.txt
```

`yolov8n.pt` downloads automatically on first run. Torch with CUDA is optional; CPU works (slower).

## Run

```bash
# Webcam (default) — good for a first smoke test
python -m streetcapture

# Tapo C100 / any RTSP camera
python -m streetcapture --source "rtsp://user:pass@192.168.1.50:554/stream1"

# A recorded clip, no windows (data only)
python -m streetcapture --source clip.mp4 --headless
```

Two windows open: **LIVE VIEW** (boxes, class, track ID, FPS) and **ARTIFACT VIEW**
(active tracks, per-class daily counts, recent events). Press **`q`** to quit.

Common flags: `--source --model --device {"",cpu,0} --conf --live-fps --artifact-fps --no-live --no-dashboard --headless`.
Every flag also has a `STREETCAPTURE_*` env var (see `streetcapture/config.py`).

## Data

```
data/
  tracks.jsonl     one completed track record per line (duration, class_history, positions, embedding stub, snapshot path)
  events.jsonl     object_entered / object_stay / object_left
  snapshots/       one JPG per track (first sighting)
```

Offline summary:

```bash
python scripts/report.py
```

## Scope

**In v0.1:** capture, YOLOv8-nano, ByteTrack, live overlay, track records, 8×8 grayscale
embedding *stub*, entered/stay/left events, JSONL store, dashboard, FPS caps + frame dropping.

**Not yet (v0.2+):** CLIP embeddings, FAISS similarity, clustering (HDBSCAN),
semantic labels (DPD, bin lorry…), prediction, multi-camera.
