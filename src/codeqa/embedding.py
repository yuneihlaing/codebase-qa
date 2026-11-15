"""Embedders turn text into unit-length vectors so that cosine similarity == dot product.

Two implementations share one tiny interface (`encode(texts) -> float32 array`):
  * SentenceTransformerEmbedder - the real thing (MiniLM by default, swap for a code model).
  * HashingEmbedder - dependency-free bag-of-subword-tokens vectors. Used by the unit tests and handy as
    a fast lexical baseline. Not a substitute for a learned model.
"""
from __future__ import annotations

import re
import zlib
from typing import Protocol

import numpy as np

from .chunking import Chunk
from .config import EmbedConfig


class Embedder(Protocol):
    def encode(self, texts: list[str], batch_size: int = 64, show_progress: bool = False) -> np.ndarray: ...


class SentenceTransformerEmbedder:
    def __init__(self, model_name: str, trust_remote_code: bool = False):
        from sentence_transformers import SentenceTransformer
        self.model = SentenceTransformer(model_name, trust_remote_code=trust_remote_code)

    def encode(self, texts, batch_size=64, show_progress=False):
        return self.model.encode(texts, batch_size=batch_size, normalize_embeddings=True,
                                 show_progress_bar=show_progress).astype("float32")


_SUBWORD = re.compile(r"[A-Z]?[a-z]+|[A-Z]+(?![a-z])|\d+")


class HashingEmbedder:
    def __init__(self, dim: int = 512):
        self.dim = dim

    def encode(self, texts, batch_size=64, show_progress=False):
        out = np.zeros((len(texts), self.dim), dtype="float32")
        for i, t in enumerate(texts):
            for tok in _SUBWORD.findall(t):
                h = zlib.crc32(tok.lower().encode())
                out[i, h % self.dim] += 1.0 if (h >> 16) & 1 else -1.0
        norms = np.linalg.norm(out, axis=1, keepdims=True)
        norms[norms == 0] = 1.0
        return (out / norms).astype("float32")


def chunk_embed_text(c: Chunk, max_chars: int) -> str:
    """What we actually embed: path + name give the model a lot of signal, then the code itself."""
    return f"{c.path} {c.name}\n{c.text}"[:max_chars]


def make_embedder(cfg: EmbedConfig) -> Embedder:
    if cfg.model == "hashing":
        return HashingEmbedder()
    return SentenceTransformerEmbedder(cfg.model, cfg.trust_remote_code)
