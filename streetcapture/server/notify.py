"""Phone notifications via ntfy.sh.

Dead-simple: POST a message to https://ntfy.sh/<topic>. Subscribe to that same
topic in the free ntfy app on your phone and matches show up as push
notifications ("DPD van at the door"). Stdlib only.
"""

from __future__ import annotations

import urllib.request


def send(server: str, topic: str, title: str, message: str,
         click: str | None = None, tags=None, priority: str = "default") -> bool:
    if not topic:
        return False
    url = f"{server.rstrip('/')}/{topic}"
    # ntfy headers must be latin-1 safe; strip anything exotic.
    headers = {
        "Title": title.encode("ascii", "ignore").decode(),
        "Priority": priority,
    }
    if click:
        headers["Click"] = click
    if tags:
        headers["Tags"] = ",".join(tags)
    req = urllib.request.Request(
        url, data=message.encode("utf-8"), headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=8) as r:
            return 200 <= r.status < 300
    except Exception as e:
        print(f"[notify] failed: {e}")
        return False
