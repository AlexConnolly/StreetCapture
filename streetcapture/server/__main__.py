"""Run the StreetCapture web server: python -m streetcapture.server"""

from __future__ import annotations

import uvicorn

from ..config import Config
from .app import create_app


def main() -> None:
    cfg = Config()
    if cfg.web_password == "streetcapture":
        print("[server] WARNING: using default password 'streetcapture' — "
              "set STREETCAPTURE_PASSWORD to something private.")
    app = create_app(cfg)
    print(f"[server] http://{cfg.web_host}:{cfg.web_port}  (UI + API)")
    uvicorn.run(app, host=cfg.web_host, port=cfg.web_port, log_level="info")


if __name__ == "__main__":
    main()
