"""Vector index over chunks (Step 2). Flat inner-product search = exact cosine similarity on unit vectors.

FAISS is used when installed; otherwise a NumPy matrix product gives identical results. At a few thousand
chunks both are instant -- FAISS only starts to matter for the very large repos (django, sympy, ...).
"""
from __future__ import annotations

import json
from dataclasses import asdict
from pathlib import Path

import numpy as np

from .chunking import Chunk, load_chunks, save_chunks
from .config import ChunkConfig, EmbedConfig
from .embedding import Embedder, chunk_embed_text

try:  # optional dependency
    import faiss
except ImportError:  # pragma: no cover
    faiss = None


class IndexStore:
    def __init__(self, chunks: list[Chunk], emb: np.ndarray, meta: dict | None = None):
        assert len(chunks) == len(emb)
        self.chunks, self.emb, self.meta = chunks, emb.astype("float32"), meta or {}
        self._faiss = None
        if faiss is not None and len(chunks):
            self._faiss = faiss.IndexFlatIP(self.emb.shape[1])
            self._faiss.add(self.emb)

    # ------------------------------------------------------------------ search
    def search(self, qvec: np.ndarray, k: int) -> list[tuple[int, float]]:
        k = min(k, len(self.chunks))
        q = qvec.astype("float32").reshape(1, -1)
        if self._faiss is not None:
            scores, ids = self._faiss.search(q, k)
            return [(int(i), float(s)) for i, s in zip(ids[0], scores[0]) if i >= 0]
        s = (self.emb @ q[0])
        top = np.argpartition(-s, k - 1)[:k]
        top = top[np.argsort(-s[top])]
        return [(int(i), float(s[i])) for i in top]

    def similarity(self, qvec: np.ndarray, idx: int) -> float:
        return float(self.emb[idx] @ qvec)

    # ------------------------------------------------------------------ build / persist
    @classmethod
    def build(cls, chunks: list[Chunk], embedder: Embedder, ecfg: EmbedConfig, meta: dict | None = None,
              show_progress: bool = True, ccfg: ChunkConfig | None = None) -> "IndexStore":
        texts = [chunk_embed_text(c, ecfg.max_chars) for c in chunks]
        emb = embedder.encode(texts, batch_size=ecfg.batch_size, show_progress=show_progress)
        extra = {"chunk_config": _chunk_sig(ccfg)} if ccfg is not None else {}
        return cls(chunks, emb, {**(meta or {}), "embed_model": ecfg.model, "embed_max_chars": ecfg.max_chars,
                                 "n_chunks": len(chunks), **extra})

    def save(self, directory: str | Path) -> None:
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        save_chunks(self.chunks, d / "chunks.jsonl")
        np.save(d / "emb.npy", self.emb)
        (d / "meta.json").write_text(json.dumps(self.meta, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, directory: str | Path, ecfg: EmbedConfig | None = None,
             ccfg: ChunkConfig | None = None) -> "IndexStore":
        d = Path(directory)
        if not (d / "meta.json").exists():
            raise FileNotFoundError(f"No index in {d}. Run scripts/prepare_repo.py for this repo with the same "
                                    "--config (the index depends on the embedder).")
        meta = json.loads((d / "meta.json").read_text(encoding="utf-8"))
        if ecfg is not None and meta.get("embed_model") != ecfg.model:
            raise ValueError(
                f"Index in {d} was built with {meta.get('embed_model')!r} but config asks for {ecfg.model!r}. "
                "Rebuild with scripts/prepare_repo.py (mixing embedders silently ruins retrieval).")
        if ccfg is not None and "chunk_config" in meta and meta["chunk_config"] != _chunk_sig(ccfg):
            raise ValueError(f"Index in {d} was built with different chunk settings than the config asks for. "
                             "Rebuild with scripts/prepare_repo.py.")
        return cls(load_chunks(d / "chunks.jsonl"), np.load(d / "emb.npy"), meta)


def _chunk_sig(ccfg: ChunkConfig) -> dict:
    return json.loads(json.dumps(asdict(ccfg)))
