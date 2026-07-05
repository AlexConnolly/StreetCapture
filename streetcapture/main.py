"""Orchestrator — wires the fast loop and the slow loop together.

Threads:
  * FrameGrabber   (capture)          — background thread
  * ArtifactEngine (slow loop)        — background thread (records + embeddings)
  * live loop      (detect + display) — main thread (OpenCV GUI must run here)
"""

from __future__ import annotations

import argparse
import time

import cv2

from .artifact import ArtifactEngine
from .capture import FrameGrabber
from .config import Config
from .dashboard import draw_dashboard, draw_live
from .db import Database
from .detector import Detector
from .embeddings import Embedder
from .state import SharedState


def run(cfg: Config) -> None:
    cfg.ensure_dirs()
    db = Database(cfg.db_path)
    session_id = db.start_session(cfg.source, cfg.model)
    state = SharedState()

    print(f"[streetcapture] source={cfg.source} model={cfg.model} "
          f"live={cfg.live_fps}fps artifact={cfg.artifact_fps}fps db={cfg.db_path}")
    grabber = FrameGrabber(cfg.cv_source).start()
    detector = Detector(cfg)
    embedder = Embedder(cfg)
    print(f"[streetcapture] embeddings: {embedder.model_version}")
    artifact = ArtifactEngine(cfg, state, db, embedder, session_id).start()

    live_interval = 1.0 / max(cfg.live_fps, 0.1)
    fps_ema = None

    try:
        while True:
            t0 = time.time()
            frame, fid = grabber.read()
            if frame is None:
                time.sleep(0.05)
                continue

            tracks = detector.track(frame)
            state.publish(frame, tracks, fid)

            if cfg.show_live:
                cv2.imshow("LIVE VIEW",
                           draw_live(frame.copy(), tracks, fps_ema, artifact.live_meta_snapshot()))
            if cfg.show_dashboard:
                cv2.imshow("ARTIFACT VIEW", draw_dashboard(artifact.dashboard_snapshot()))
            if not cfg.headless:
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break

            # Cap the live loop to the target FPS (frame skipping / drop when lagging).
            dt = time.time() - t0
            if dt < live_interval:
                time.sleep(live_interval - dt)
            inst = 1.0 / max(time.time() - t0, 1e-6)
            fps_ema = inst if fps_ema is None else 0.9 * fps_ema + 0.1 * inst
    except KeyboardInterrupt:
        print("\n[streetcapture] interrupted")
    finally:
        artifact.stop()
        grabber.stop()
        db.close()
        cv2.destroyAllWindows()
        print("[streetcapture] stopped")


def _parse_args(argv=None) -> Config:
    cfg = Config()
    p = argparse.ArgumentParser(prog="streetcapture", description="Dual-speed perception engine (v0.2)")
    p.add_argument("--source", default=cfg.source, help="RTSP URL, webcam index, or video file")
    p.add_argument("--model", default=cfg.model, help="YOLO model (nano recommended)")
    p.add_argument("--device", default=cfg.device, help='"", "cpu", or GPU index e.g. "0"')
    p.add_argument("--conf", type=float, default=cfg.conf)
    p.add_argument("--live-fps", type=float, default=cfg.live_fps)
    p.add_argument("--artifact-fps", type=float, default=cfg.artifact_fps)
    p.add_argument("--no-embed", action="store_true", help="skip embedding generation")
    p.add_argument("--no-live", action="store_true", help="hide the LIVE VIEW window")
    p.add_argument("--no-dashboard", action="store_true", help="hide the ARTIFACT VIEW window")
    p.add_argument("--headless", action="store_true", help="no windows (data only)")
    a = p.parse_args(argv)

    cfg.source = a.source
    cfg.model = a.model
    cfg.device = a.device
    cfg.conf = a.conf
    cfg.live_fps = a.live_fps
    cfg.artifact_fps = a.artifact_fps
    cfg.embed_enabled = cfg.embed_enabled and not a.no_embed
    if a.headless:
        cfg.show_live = cfg.show_dashboard = False
    else:
        cfg.show_live = cfg.show_live and not a.no_live
        cfg.show_dashboard = cfg.show_dashboard and not a.no_dashboard
    return cfg


def main(argv=None) -> None:
    run(_parse_args(argv))


if __name__ == "__main__":
    main()
