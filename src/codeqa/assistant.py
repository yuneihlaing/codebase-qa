"""The assistant itself: one object that answers a question about one repository.

Used by the interactive CLI (scripts/ask.py), the web UI (scripts/app.py) and the query-set runners.
The batch benchmark script (scripts/run_experiment.py) uses the same retriever + prompt + LLM.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

from .chunking import Chunk
from .config import Config
from .llm import LLM, make_llm
from .pipeline import answer_question, load_retriever
from .retrieval import STRATEGIES, Retriever


@dataclass
class Answer:
    question: str
    strategy: str
    text: str
    chunks: list[Chunk]
    origins: list[str] = field(default_factory=list)
    scores: list[float] = field(default_factory=list)
    seconds: float = 0.0

    def sources(self) -> list[dict]:
        return [{"path": c.path, "start": c.start, "end": c.end, "name": c.name, "kind": c.kind,
                 "origin": o, "score": round(s, 4)} for c, o, s in zip(self.chunks, self.origins, self.scores)]

    def format_sources(self, limit: int | None = None) -> str:
        rows = self.sources()[:limit]
        return "\n".join(f"  [{r['origin']:<9}] {r['path']}:{r['start']}-{r['end']}  ({r['kind']}: {r['name']})"
                         for r in rows)


class Assistant:
    def __init__(self, repo: str, cfg: Config, llm: LLM | None = None, retriever: Retriever | None = None):
        self.repo, self.cfg = repo, cfg
        self.retriever = retriever or load_retriever(repo, cfg)
        self.llm = llm or make_llm(cfg.llm)

    def ask(self, question: str, strategy: str = "structure", extra_chunks: list[Chunk] | None = None) -> Answer:
        """Retrieve context with `strategy`, optionally prepend extra chunks (used by injection tests), answer."""
        if strategy not in STRATEGIES:
            raise ValueError(f"Unknown strategy {strategy!r}; choose from {STRATEGIES}")
        t0 = time.time()
        r = self.retriever.retrieve(strategy, question)
        chunks, origins, scores = list(r.chunks), list(r.origins), list(r.scores)
        for c in reversed(extra_chunks or []):
            chunks.insert(0, c)
            origins.insert(0, "injected")
            scores.insert(0, 0.0)
        text = answer_question(self.llm, question, chunks, self.cfg)
        return Answer(question, strategy, text, chunks, origins, scores, round(time.time() - t0, 2))


class SessionLog:
    """Append-only JSONL log of interactive sessions -> evidence for the peer/user testing section."""

    def __init__(self, path: str | Path, tester: str):
        self.path, self.tester = Path(path), tester
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, ans: Answer, rating: str | None = None, comment: str = "", **extra) -> None:
        row = {"time": datetime.now(timezone.utc).isoformat(timespec="seconds"), "tester": self.tester,
               "question": ans.question, "strategy": ans.strategy, "answer": ans.text, "rating": rating,
               "comment": comment, "seconds": ans.seconds, "sources": ans.sources(), **extra}
        with open(self.path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
