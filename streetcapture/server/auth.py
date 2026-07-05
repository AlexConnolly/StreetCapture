"""Password auth via signed JWT bearer tokens.

Single shared password (``STREETCAPTURE_PASSWORD``). The signing secret defaults
to a value derived from the password so tokens survive restarts without extra
config; override with ``STREETCAPTURE_SECRET`` for a stable independent secret.

Tokens are accepted as ``Authorization: Bearer <t>`` for API calls and as a
``?token=`` query param for <img>-driven endpoints (MJPEG stream, media) that
can't send headers.
"""

from __future__ import annotations

import hashlib
import time

import jwt
from fastapi import HTTPException, Request

ALGO = "HS256"
TTL_SECONDS = 7 * 24 * 3600  # a week


def _secret(cfg) -> str:
    if cfg.web_secret:
        return cfg.web_secret
    return hashlib.sha256(f"streetcapture::{cfg.web_password}".encode()).hexdigest()


def issue_token(cfg) -> str:
    now = int(time.time())
    return jwt.encode({"sub": "user", "iat": now, "exp": now + TTL_SECONDS}, _secret(cfg), algorithm=ALGO)


def verify_password(cfg, password: str) -> bool:
    return bool(password) and password == cfg.web_password


def _check(cfg, token: str) -> None:
    try:
        jwt.decode(token, _secret(cfg), algorithms=[ALGO])
    except Exception:
        raise HTTPException(status_code=401, detail="invalid or expired token")


def require_auth(cfg):
    """FastAPI dependency factory — validates header or ?token=."""
    def dep(request: Request) -> None:
        token = ""
        header = request.headers.get("authorization", "")
        if header.lower().startswith("bearer "):
            token = header[7:]
        if not token:
            token = request.query_params.get("token", "")
        if not token:
            raise HTTPException(status_code=401, detail="missing token")
        _check(cfg, token)
    return dep
