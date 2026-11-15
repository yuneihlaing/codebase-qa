"""Retrieval strategies (Step 3). Every strategy returns a `Retrieval` so they can be swapped freely.

  baseline          plain semantic top-k                                  (textbook RAG)
  baseline_budget   semantic ranking, filled up to a character budget    (the *fair* control)
  code_seeds        like structure but with NO graph edges: top code chunks first, then semantic fill
                    (separates "prefer code over docs" from "follow the graph")
  structure         code seeds + graph neighbours, same character budget (the proposed method)
  structure_refs    ablation: only reference/containment edges (ref, parent, header), no call edges
  structure_calls   ablation: only call edges (call, called_by) plus parent/header

Why `baseline_budget` exists: if structure-aware retrieval simply feeds the LLM *more text*, any gain could
come from the extra context rather than from the graph. Giving both the same budget isolates the graph's
contribution.

How `structure` builds its context (all knobs live in RetrievalConfig):
  1. Semantic search returns the top `pool` chunks.
  2. Seeds = the top `k_seed` *code* chunks of that list (`seed_from: code`). Documentation chunks have no
     graph edges, so seeding from them leaves nothing to expand (on Flask ~3/4 of top hits are docs).
  3. Every graph neighbour of a seed is scored:
        alpha * seed_score * edge_weight + (1 - alpha) * similarity(question, neighbour)
     Reverse ("called_by") neighbours are first ranked by similarity and capped at `max_in_edges` per seed.
  4. The best `max_expansion` neighbours are candidates. The top `reserve_expansion` of them are placed
     right after the seeds; the rest compete with ordinary semantic hits on score.
     (Without reserved slots a neighbour almost never outranks the semantic hits, and `structure`
     degenerates into `baseline_budget`.)
  5. Seeds, reserved neighbours, then everything else by score, until `context_char_budget` is used.

`seed_from: any` + `reserve_expansion: 0` reproduces the first version of this method.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .chunking import Chunk
from .config import RetrievalConfig
from .embedding import Embedder, encode_queries
from .index import IndexStore
from .structure import RepoGraph

STRATEGIES = ("baseline", "baseline_budget", "code_seeds", "structure", "structure_refs", "structure_calls")
_REF_EDGES = {"ref", "parent", "header"}             # "what does this use / where does it live"
_CALL_EDGES = {"call", "called_by", "parent", "header"}
_EDGE_FILTERS = {"structure": None, "structure_refs": _REF_EDGES, "structure_calls": _CALL_EDGES,
                 "code_seeds": set()}                # empty filter -> no neighbour qualifies -> no expansion


@dataclass
class Retrieval:
    chunks: list[Chunk]
    origins: list[str] = field(default_factory=list)   # "seed" | "semantic" | edge type that pulled it in
    scores: list[float] = field(default_factory=list)

    def labels(self) -> list[str]:
        return [c.label() for c in self.chunks]


def fill_budget(order: list[int], chunks: list[Chunk], budget: int, per_chunk: int) -> list[int]:
    """Take chunks in order until `budget` characters (after per-chunk truncation) are used."""
    picked, used = [], 0
    for i in order:
        cost = min(len(chunks[i].text), per_chunk)
        if picked and used + cost > budget:
            continue                       # a later, smaller chunk may still fit
        picked.append(i)
        used += cost
    return picked


def is_code(c: Chunk) -> bool:
    return c.kind != "text"


class Retriever:
    def __init__(self, store: IndexStore, embedder: Embedder, cfg: RetrievalConfig,
                 graph: RepoGraph | None = None):
        self.store, self.embedder, self.cfg, self.graph = store, embedder, cfg, graph

    def _qvec(self, question: str) -> np.ndarray:
        return encode_queries(self.embedder, [question])[0]

    def retrieve(self, strategy: str, question: str) -> Retrieval:
        if strategy == "baseline":
            return self.baseline(question)
        if strategy == "baseline_budget":
            return self.baseline_budget(question)
        if strategy in _EDGE_FILTERS:
            return self.structure_aware(question, _EDGE_FILTERS[strategy])
        raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")

    # ------------------------------------------------------------------ baselines
    def baseline(self, question: str) -> Retrieval:
        hits = self.store.search(self._qvec(question), self.cfg.k)
        return Retrieval([self.store.chunks[i] for i, _ in hits], ["seed"] * len(hits), [s for _, s in hits])

    def baseline_budget(self, question: str) -> Retrieval:
        hits = self.store.search(self._qvec(question), self.cfg.pool)
        score = dict(hits)
        picked = fill_budget([i for i, _ in hits], self.store.chunks, self.cfg.context_char_budget,
                             self.cfg.per_chunk_chars)
        return Retrieval([self.store.chunks[i] for i in picked], ["semantic"] * len(picked),
                         [score[i] for i in picked])

    # ------------------------------------------------------------------ structure-aware
    def _seeds(self, hits: list[tuple[int, float]]) -> list[tuple[int, float]]:
        if self.cfg.seed_from == "any":
            return hits[:self.cfg.k_seed]
        if self.cfg.seed_from == "code":
            code = [(i, s) for i, s in hits if is_code(self.store.chunks[i])]
            return code[:self.cfg.k_seed] or hits[:self.cfg.k_seed]
        raise ValueError(f"retrieval.seed_from must be 'code' or 'any', got {self.cfg.seed_from!r}")

    def _scored_neighbours(self, q: np.ndarray, seeds: list[tuple[int, float]], seed_ids: set,
                           edge_filter: set | None) -> dict[int, tuple[float, str]]:
        cfg, store = self.cfg, self.store
        expansion: dict[int, tuple[float, str]] = {}
        for sidx, sscore in seeds:
            nbrs = [(n, t) for n, t in self.graph.neighbors(sidx)
                    if n not in seed_ids and (edge_filter is None or t in edge_filter)]
            fwd = [(n, t) for n, t in nbrs if t != "called_by"]
            rev = sorted((n for n, t in nbrs if t == "called_by"), key=lambda n: -store.similarity(q, n))
            for nidx, etype in fwd + [(n, "called_by") for n in rev[:cfg.max_in_edges]]:
                w = cfg.edge_weights.get(etype, 0.5)
                score = cfg.alpha * sscore * w + (1 - cfg.alpha) * store.similarity(q, nidx)
                if nidx not in expansion or score > expansion[nidx][0]:
                    expansion[nidx] = (score, etype)
        return expansion

    def structure_aware(self, question: str, edge_filter: set | None = None) -> Retrieval:
        if self.graph is None:
            raise RuntimeError("structure-aware retrieval needs a RepoGraph")
        cfg, store = self.cfg, self.store
        q = self._qvec(question)
        hits = store.search(q, cfg.pool)
        seeds = self._seeds(hits)
        seed_ids = {i for i, _ in seeds}

        expansion = self._scored_neighbours(q, seeds, seed_ids, edge_filter)
        best = sorted(expansion.items(), key=lambda kv: -kv[1][0])[:cfg.max_expansion]
        reserved = [i for i, _ in best[:cfg.reserve_expansion]]

        pool: dict[int, tuple[float, str]] = {i: (s, "semantic") for i, s in hits if i not in seed_ids}
        for i, (s, t) in best:
            if i not in pool or s > pool[i][0] or i in reserved:
                pool[i] = (s, t)
        rest = [i for i, _ in sorted(pool.items(), key=lambda kv: -kv[1][0]) if i not in reserved]
        order = [i for i, _ in seeds] + reserved + rest

        picked = fill_budget(order, store.chunks, cfg.context_char_budget, cfg.per_chunk_chars)
        seed_score = dict(seeds)
        origins = ["seed" if i in seed_ids else pool[i][1] for i in picked]
        scores = [seed_score[i] if i in seed_ids else pool[i][0] for i in picked]
        return Retrieval([store.chunks[i] for i in picked], origins, scores)
