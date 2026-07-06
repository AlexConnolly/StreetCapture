"""Continuous DVR recording via ffmpeg.

A separate ffmpeg process pulls the RTSP stream and writes it into short,
independently-playable mp4 segments named by wall-clock start time
(``seg-YYYYmmdd-HHMMSS.mp4``). The web UI reads the segment index to build a
24h scrub-back timeline: to seek to time *T* it loads the segment whose start
<= T and sets ``video.currentTime = T - segment_start``.

* ``-c copy`` by default -> no re-encode, so this is cheap on CPU and keeps the
  camera's exact quality. Seeking granularity is one GOP (~1-4s on a Tapo).
* A retention thread prunes segments older than ``record_retention_h``.
* This opens a *second* RTSP session to the camera (independent of the OpenCV
  grabber used for detection). Tapo cameras allow several; point
  ``record_source`` at the sub-stream if you need to lighten the load.
"""

from __future__ import annotations

import re
import subprocess
import tempfile
import threading
import time
from datetime import datetime
from pathlib import Path

SEG_PREFIX = "seg-"
SEG_GLOB = "seg-*.mp4"
SEG_TIME_FMT = "%Y%m%d-%H%M%S"

FRAG_FLAGS = "movflags=+frag_keyframe+empty_moov+default_base_moof"


def safe_name(name: str) -> str:
    """Filesystem-safe slug for a user-supplied clip name."""
    s = re.sub(r"[^A-Za-z0-9 _-]", "", name).strip().replace(" ", "_")
    return (s or "clip")[:60]


def parse_segment_start(name: str) -> float | None:
    """'seg-20260705-141530.mp4' -> epoch seconds (local time), or None."""
    stem = Path(name).stem
    if not stem.startswith(SEG_PREFIX):
        return None
    try:
        dt = datetime.strptime(stem[len(SEG_PREFIX):], SEG_TIME_FMT)
        return time.mktime(dt.timetuple())
    except ValueError:
        return None


class Recorder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.dir = Path(cfg.recordings_dir)
        self.source = cfg.record_source or cfg.source
        self._proc = None
        self._running = False
        self._rec_thread = None
        self._retention_thread = None

    # -- lifecycle ---------------------------------------------------------
    def start(self) -> "Recorder":
        self.dir.mkdir(parents=True, exist_ok=True)
        self._running = True
        self._rec_thread = threading.Thread(target=self._run, name="Recorder", daemon=True)
        self._rec_thread.start()
        self._retention_thread = threading.Thread(
            target=self._retain, name="RecorderRetention", daemon=True)
        self._retention_thread.start()
        return self

    def _ffmpeg_cmd(self) -> list[str]:
        # ffmpeg -strftime expands these % tokens to the segment's start time.
        out = str(self.dir / f"{SEG_PREFIX}%Y%m%d-%H%M%S.mp4")
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "warning"]
        if str(self.source).lower().startswith("rtsp"):
            cmd += ["-rtsp_transport", "tcp"]
        cmd += ["-i", str(self.source)]
        if self.cfg.record_scale:
            cmd += ["-vf", f"scale={self.cfg.record_scale}",
                    "-c:v", "libx264", "-preset", "veryfast",
                    "-crf", str(self.cfg.record_crf)]
        else:
            cmd += ["-c:v", "copy"]      # no re-encode
        cmd += [
            "-an",                        # drop audio
            "-f", "segment",
            "-segment_time", str(self.cfg.record_segment_s),
            "-segment_format", "mp4",
            # Fragmented mp4: each segment is playable in the browser <video>
            # even while it's still being written (a plain-mp4 segment has no
            # moov atom until it's finalised, so it can't be scrubbed live).
            "-segment_format_options",
            "movflags=+frag_keyframe+empty_moov+default_base_moof",
            "-reset_timestamps", "1",
            "-strftime", "1",
            out,
        ]
        return cmd

    def _run(self) -> None:
        while self._running:
            try:
                self._proc = subprocess.Popen(
                    self._ffmpeg_cmd(),
                    stdin=subprocess.DEVNULL,
                    stdout=subprocess.DEVNULL,
                    stderr=subprocess.DEVNULL,
                )
                print(f"[recorder] ffmpeg started (source={self.source}, "
                      f"segment={self.cfg.record_segment_s}s)")
                self._proc.wait()
            except FileNotFoundError:
                print("[recorder] ffmpeg not found on PATH — DVR disabled")
                return
            except Exception as e:  # noqa: BLE001
                print(f"[recorder] ffmpeg error: {e}")
            if self._running:
                time.sleep(3.0)  # camera hiccup / process died — retry

    def _retain(self) -> None:
        while self._running:
            try:
                cutoff = time.time() - self.cfg.record_retention_h * 3600
                for f in self.dir.glob(SEG_GLOB):
                    start = parse_segment_start(f.name)
                    ref = start if start is not None else f.stat().st_mtime
                    # never delete the newest segment (may be recording into it)
                    if ref < cutoff:
                        try:
                            f.unlink()
                        except OSError:
                            pass
            except Exception as e:  # noqa: BLE001
                print(f"[recorder] retention error: {e}")
            for _ in range(60):  # sweep ~every minute, but stay responsive to stop()
                if not self._running:
                    return
                time.sleep(1.0)

    # -- index for the timeline API ---------------------------------------
    def index(self) -> list[dict]:
        """[{name, start, duration}, ...] oldest->newest. Duration is inferred
        from the next segment's start (now - start for the newest)."""
        segs = []
        for f in self.dir.glob(SEG_GLOB):
            start = parse_segment_start(f.name)
            if start is None:
                continue
            segs.append({"name": f.name, "start": start, "size": f.stat().st_size})
        segs.sort(key=lambda s: s["start"])
        now = time.time()
        for i, s in enumerate(segs):
            nxt = segs[i + 1]["start"] if i + 1 < len(segs) else now
            s["duration"] = round(max(0.0, nxt - s["start"]), 2)
        return segs

    def stats(self) -> dict:
        segs = self.index()
        return {
            "recording": bool(self._proc and self._proc.poll() is None),
            "segments": len(segs),
            "bytes": sum(s["size"] for s in segs),
            "earliest": segs[0]["start"] if segs else None,
        }

    # -- continuous playback / clip export --------------------------------
    def _covering(self, start: float, end: float | None = None) -> tuple[list[dict], float]:
        """Segments overlapping [start, end); offset = seconds into the first."""
        end = end or time.time()
        segs = [s for s in self.index()
                if s["start"] + s["duration"] > start and s["start"] < end]
        offset = max(0.0, start - segs[0]["start"]) if segs else 0.0
        return segs, offset

    def _write_concat_list(self, segs: list[dict]) -> str:
        fd, path = tempfile.mkstemp(suffix=".txt", prefix="sc_concat_")
        with open(fd, "w") as f:
            for s in segs:
                p = str((self.dir / s["name"]).resolve()).replace("\\", "/")
                f.write(f"file '{p}'\n")
        return path

    def play_stream(self, start: float):
        """Popen an ffmpeg that concatenates segments from `start` -> now and
        streams a single continuous fragmented-mp4 to stdout (seamless playback,
        no per-segment reload). Returns (proc, concat_list_path) or (None, None)."""
        segs, offset = self._covering(start)
        if not segs:
            return None, None
        lst = self._write_concat_list(segs)
        # -re streams at real-time pace so the browser buffers a few seconds,
        # not the entire rest of the day when you scrub hours back. -ss is an
        # INPUT option (before -i) so the seek is fast (keyframe demux) rather
        # than reading-and-discarding at 1x.
        cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-re"]
        if offset > 0.5:
            cmd += ["-ss", f"{offset:.2f}"]
        cmd += ["-f", "concat", "-safe", "0", "-i", lst,
                "-c", "copy", "-movflags",
                "+frag_keyframe+empty_moov+default_base_moof", "-f", "mp4", "pipe:1"]
        proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        return proc, lst

    def save_clip(self, start: float, end: float, name: str) -> dict:
        """Export [start, end] into the permanent library (survives the prune)."""
        segs, offset = self._covering(start, end)
        if not segs:
            return {"error": "no footage in that range"}
        duration = max(1.0, end - start)
        base = safe_name(name)
        stamp = datetime.fromtimestamp(start).strftime("%Y%m%d-%H%M%S")
        fname = f"{base}__{stamp}.mp4"
        out = self.cfg.library_dir / fname
        self.cfg.library_dir.mkdir(parents=True, exist_ok=True)
        lst = self._write_concat_list(segs)
        try:
            cmd = ["ffmpeg", "-nostdin", "-loglevel", "error", "-y",
                   "-f", "concat", "-safe", "0", "-i", lst]
            if offset > 0.5:
                cmd += ["-ss", f"{offset:.2f}"]
            cmd += ["-t", f"{duration:.2f}", "-c", "copy",
                    "-movflags", "+faststart", str(out)]
            r = subprocess.run(cmd, stderr=subprocess.PIPE, timeout=120)
        finally:
            try:
                Path(lst).unlink()
            except OSError:
                pass
        if r.returncode != 0 or not out.is_file():
            return {"error": "export failed"}
        return {"name": fname, "start": start, "duration": duration,
                "size": out.stat().st_size}

    def library_index(self) -> list[dict]:
        d = self.cfg.library_dir
        if not d.is_dir():
            return []
        out = []
        for f in d.glob("*.mp4"):
            out.append({"name": f.name, "size": f.stat().st_size,
                        "saved_at": f.stat().st_mtime})
        out.sort(key=lambda s: s["saved_at"], reverse=True)
        return out

    def delete_clip(self, name: str) -> bool:
        if "/" in name or "\\" in name or not name.endswith(".mp4"):
            return False
        p = (self.cfg.library_dir / name).resolve()
        if not str(p).startswith(str(self.cfg.library_dir.resolve())) or not p.is_file():
            return False
        try:
            p.unlink()
            return True
        except OSError:
            return False

    def stop(self) -> None:
        self._running = False
        if self._proc and self._proc.poll() is None:
            try:
                self._proc.terminate()
                self._proc.wait(timeout=5)
            except Exception:  # noqa: BLE001
                try:
                    self._proc.kill()
                except Exception:  # noqa: BLE001
                    pass
