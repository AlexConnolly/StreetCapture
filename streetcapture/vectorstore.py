"""FAISS vector index of artifact embeddings.

Per the v1 spec the index is *built and maintained* now; similarity search
itself is a v2 feature. We still expose ``search`` (and test it) so v2 can plug
in clustering / nearest-neighbour without touching this layer.

Embeddings are L2-normalised, so an inner-product flat index == cosine
similarity. IDs stored in the index are the artifact primary keys.
"""

from __future__ import annotations

import threading
from pathlib import Path

import numpy as np

try:
    import faiss
    _HAVE_FAISS = True
except Exception:
    _HAVE_FAISS = False


class VectorStore:
    def __init__(self, path):
        self.path = Path(path)
        self.dim = None
        self.index = None
        self._lock = threading.Lock()
        self.available = _HAVE_FAISS
        if not _HAVE_FAISS:
            print("[vectorstore] faiss not installed; embedding index disabled.")
            return
        if self.path.exists():
            self.index = faiss.read_index(str(self.path))
            self.dim = self.index.d
            print(f"[vectorstore] loaded {self.index.ntotal} vectors (dim {self.dim})")

    def _ensure(self, dim: int) -> None:
        if self.index is None:
            self.dim = dim
            self.index = faiss.IndexIDMap2(faiss.IndexFlatIP(dim))

    def add(self, artifact_id: int, vector) -> None:
        if not self.available or vector is None:
            return
        v = np.asarray([vector], dtype="float32")
        with self._lock:
            self._ensure(v.shape[1])
            if v.shape[1] != self.dim:
                return  # dimension mismatch (e.g. stub vs CLIP) — skip
            self.index.add_with_ids(v, np.asarray([artifact_id], dtype="int64"))

    def search(self, vector, k: int = 5):
        """Return [(artifact_id, score), ...] — reserved for v2 similarity."""
        if not self.available or self.index is None or vector is None:
            return []
        v = np.asarray([vector], dtype="float32")
        if v.shape[1] != self.dim:
            return []
        with self._lock:
            scores, ids = self.index.search(v, min(k, max(1, self.index.ntotal)))
        return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i != -1]

    @property
    def ntotal(self) -> int:
        return int(self.index.ntotal) if self.index is not None else 0

    def save(self) -> None:
        if not self.available or self.index is None:
            return
        with self._lock:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            faiss.write_index(self.index, str(self.path))
