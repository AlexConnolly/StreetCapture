"""Embedding generation for completed artifacts.

Default backend is OpenCLIP (ViT-B-32 / laion2b). Runs once per artifact, on the
async artifact thread, so it never touches the live loop. If open_clip isn't
importable the engine still works — it falls back to a cheap grayscale stub and
records that in ``model_version`` so the DB stays honest.

No similarity search here — v0.2 only *builds* the vector database.
"""

from __future__ import annotations

import threading

import cv2
import numpy as np


class Embedder:
    def __init__(self, cfg):
        self.cfg = cfg
        self.model = None
        self.preprocess = None
        self.device = "cpu"
        self.model_version = "stub-gray16"
        # The model is shared by the artifact thread and the live-label thread;
        # serialise forward passes (PyTorch modules aren't concurrency-safe).
        self._lock = threading.Lock()
        if not cfg.embed_enabled:
            self.model_version = "disabled"
            return
        self.tokenizer = None
        try:
            import open_clip
            import torch

            self.device = "cuda" if (cfg.device != "cpu" and torch.cuda.is_available()) else "cpu"
            self.model, _, self.preprocess = open_clip.create_model_and_transforms(
                cfg.embed_model, pretrained=cfg.embed_pretrained, device=self.device
            )
            self.model.eval()
            self.tokenizer = open_clip.get_tokenizer(cfg.embed_model)
            self._torch = torch
            self.model_version = f"open_clip/{cfg.embed_model}:{cfg.embed_pretrained}"
        except Exception as e:  # missing package, no weights, OOM, etc.
            print(f"[embeddings] OpenCLIP unavailable ({e}); using stub.")
            self.model = None
            self.model_version = "stub-gray16"

    def embed(self, crop_bgr):
        """Return an L2-normalised float list for one representative crop."""
        if crop_bgr is None or crop_bgr.size == 0:
            return None
        if self.model is None:
            return self._stub(crop_bgr)
        try:
            from PIL import Image

            rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
            tensor = self.preprocess(Image.fromarray(rgb)).unsqueeze(0).to(self.device)
            with self._lock, self._torch.no_grad():
                feat = self.model.encode_image(tensor)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return [round(float(x), 6) for x in feat.squeeze(0).cpu().tolist()]
        except Exception as e:
            print(f"[embeddings] encode failed ({e}); using stub.")
            return self._stub(crop_bgr)

    def embed_text(self, text: str):
        """Encode a text prompt into the SAME space as image embeddings (CLIP).

        Enables zero-shot search: 'a DPD delivery van' -> matching artifacts.
        Returns None if the CLIP text encoder isn't available (stub mode).
        """
        if self.model is None or self.tokenizer is None:
            return None
        try:
            tokens = self.tokenizer([text]).to(self.device)
            with self._lock, self._torch.no_grad():
                feat = self.model.encode_text(tokens)
                feat = feat / feat.norm(dim=-1, keepdim=True)
            return [round(float(x), 6) for x in feat.squeeze(0).cpu().tolist()]
        except Exception as e:
            print(f"[embeddings] text encode failed ({e})")
            return None

    @property
    def text_available(self) -> bool:
        return self.model is not None and self.tokenizer is not None

    @staticmethod
    def _stub(crop_bgr):
        gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
        small = cv2.resize(gray, (16, 16)).astype(np.float32).flatten()
        norm = np.linalg.norm(small) or 1.0
        return [round(float(v), 6) for v in (small / norm)]
