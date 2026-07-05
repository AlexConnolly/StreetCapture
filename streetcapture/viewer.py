"""Artifact Viewer — a local web gallery over the artifact database.

    python -m streetcapture.viewer            # serves http://127.0.0.1:8000
    python -m streetcapture.viewer --port 8080

Stdlib only (no Flask). Reads artifacts/artifact.db and shows a scrollable list
of artifacts with their representative images, class, timing, quality scores and
embedding status. Read-only; safe to run alongside the live system.
"""

from __future__ import annotations

import argparse
import html
import os
import time
import urllib.parse
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from .config import Config
from .db import Database
from .query import QueryEngine

PAGE_CSS = """
* { box-sizing: border-box; }
body { margin:0; font-family: system-ui, Arial, sans-serif; background:#111; color:#eee; }
header { padding:16px 22px; background:#181818; border-bottom:1px solid #2a2a2a; position:sticky; top:0; }
h1 { margin:0; font-size:18px; letter-spacing:.5px; color:#00c8ff; }
.sub { color:#888; font-size:13px; margin-top:4px; }
.grid { display:grid; grid-template-columns:repeat(auto-fill,minmax(260px,1fr)); gap:14px; padding:18px 22px; }
.card { background:#1b1b1b; border:1px solid #2a2a2a; border-radius:10px; overflow:hidden; }
.imgs { display:flex; gap:2px; background:#000; overflow-x:auto; }
.imgs img { height:150px; object-fit:cover; }
.body { padding:10px 12px; }
.cls { font-size:15px; font-weight:600; }
.cls.person{color:#37d67a;} .cls.vehicle{color:#f0a028;} .cls.other{color:#ccc;}
.meta { color:#9a9a9a; font-size:12px; margin-top:4px; line-height:1.5; }
.tag { display:inline-block; padding:1px 6px; border-radius:6px; background:#242424; margin-right:4px; font-size:11px; }
.lab { display:inline-block; padding:1px 6px; border-radius:6px; background:#1e2d33; color:#7fd0e8; margin:2px 4px 0 0; font-size:11px; }
.emb { color:#00c8ff; } .noemb { color:#c85; }
a { color:inherit; text-decoration:none; }
form.q { margin-top:10px; display:flex; gap:8px; }
form.q input[type=text] { flex:1; max-width:640px; padding:8px 10px; border-radius:8px; border:1px solid #333; background:#111; color:#eee; font-size:14px; }
form.q button { padding:8px 16px; border-radius:8px; border:0; background:#00c8ff; color:#001; font-weight:600; cursor:pointer; }
.answer { margin:14px 22px 0; padding:12px 16px; background:#12232a; border:1px solid #1e3a44; border-radius:10px; color:#cfeaf3; white-space:pre-wrap; font-size:14px; }
.ex { color:#666; font-size:12px; margin-top:6px; }
"""


class Handler(BaseHTTPRequestHandler):
    db: Database = None
    db_path: str = ""
    images_root: str = ""

    def log_message(self, *a):  # quiet
        pass

    def do_GET(self):
        parsed = urllib.parse.urlparse(self.path)
        if parsed.path == "/":
            q = urllib.parse.parse_qs(parsed.query).get("q", [""])[0]
            self._index(q)
        elif parsed.path == "/img":
            self._image(urllib.parse.parse_qs(parsed.query).get("p", [""])[0])
        else:
            self.send_error(404)

    def _image(self, path):
        # Only serve files that live under the images directory.
        ap = os.path.abspath(path)
        if not ap.startswith(self.images_root) or not os.path.isfile(ap):
            self.send_error(404)
            return
        with open(ap, "rb") as f:
            data = f.read()
        self.send_response(200)
        self.send_header("Content-Type", "image/jpeg")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def _index(self, q=""):
        arts = self.db.recent_artifacts(limit=300)
        counts = self.db.counts()

        answer_html = ""
        if q.strip():
            eng = QueryEngine(self.db_path)
            try:
                ans = eng.answer(q)
            finally:
                eng.close()
            answer_html = f'<div class="answer">{html.escape(ans)}</div>'

        cards = []
        for a in arts:
            imgs = "".join(
                f'<img src="/img?p={urllib.parse.quote(i["path"])}" loading="lazy">'
                for i in a["images"]
            ) or '<div style="color:#666;padding:20px">no image</div>'
            cls = html.escape(a["primary_class"] or "unknown")
            catcls = cls if cls in ("person",) else ("vehicle" if cls in
                     ("car", "truck", "bus", "motorbike", "motorcycle", "bicycle", "train") else "other")
            when = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(a["start_time"] or 0))
            emb = a["embedding"]
            emb_html = (f'<span class="emb">emb: {html.escape(emb["model_version"])} ({emb["dim"]}d)</span>'
                        if emb else '<span class="noemb">emb: none</span>')
            labs = "".join(
                f'<span class="lab">{html.escape(l["type"])}: {html.escape(l["value"])}</span>'
                for l in a.get("labels", [])
            )
            cards.append(f"""
            <div class="card">
              <div class="imgs">{imgs}</div>
              <div class="body">
                <div class="cls {catcls}">{cls} <span style="color:#666;font-weight:400">· Artifact #{a['id']}</span></div>
                <div>{labs}</div>
                <div class="meta">
                  <span class="tag">track #{a['source_track_id']}</span>
                  <span class="tag">entity {a['entity_id'] if a['entity_id'] is not None else '—'}</span><br>
                  {when} · {a['duration']:.1f}s · {a['track_length']} frames<br>
                  conf {a['avg_confidence']:.2f} · sharp {a['sharpness']:.0f} · vis {a['visibility']:.2f} · motion {a['motion_distance']:.0f}px<br>
                  {emb_html}
                </div>
              </div>
            </div>""")

        qval = html.escape(q)
        body = f"""<!doctype html><html><head><meta charset="utf-8">
        <title>StreetCapture — Artifacts</title><style>{PAGE_CSS}</style></head><body>
        <header><h1>StreetCapture — Artifact Viewer</h1>
        <div class="sub">{counts['artifacts']} artifacts · {counts['tracks']} tracks · {counts['embeddings']} embeddings · {counts['events']} events</div>
        <form class="q" method="get" action="/">
          <input type="text" name="q" value="{qval}" placeholder="Ask: how many vehicles passed yesterday?">
          <button type="submit">Ask</button>
        </form>
        <div class="ex">try: “quietest time for foot traffic” · “how many vehicles today” · “show people this week”</div>
        </header>
        {answer_html}
        <div class="grid">{''.join(cards) or '<p style="color:#888">No artifacts yet. Run the live system to build some.</p>'}</div>
        </body></html>"""
        data = body.encode("utf-8")
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


def main(argv=None):
    p = argparse.ArgumentParser(prog="streetcapture.viewer")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--host", default="127.0.0.1")
    a = p.parse_args(argv)

    cfg = Config()
    Handler.db = Database(cfg.db_path)
    Handler.db_path = str(cfg.db_path)
    Handler.images_root = os.path.abspath(str(cfg.images_dir))
    srv = ThreadingHTTPServer((a.host, a.port), Handler)
    print(f"Artifact viewer: http://{a.host}:{a.port}  (db: {cfg.db_path})  Ctrl+C to stop")
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
