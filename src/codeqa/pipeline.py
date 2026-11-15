"""End-to-end glue: repo -> index + graph -> retriever -> prompt -> answer."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .chunking import Chunk, chunk_repo
from .config import Config
from .embedding import Embedder, make_embedder
from .index import IndexStore
from .llm import LLM
from .prompts import SYSTEM_PROMPT
from .repos import RepoSpec, current_commit, ensure_repo
from .retrieval import Retrieval, Retriever
from .structure import RepoGraph


def format_context(chunks: list[Chunk], per_chunk_chars: int) -> str:
    return "\n\n".join(
        f"### {c.path} (lines {c.start}-{c.end}) [{c.kind}: {c.name}]\n{c.text[:per_chunk_chars]}"
        for c in chunks)


def build_user_prompt(question: str, chunks: list[Chunk], per_chunk_chars: int = 3000) -> str:
    return f"Repository excerpts:\n\n{format_context(chunks, per_chunk_chars)}\n\nQuestion: {question}"


def answer_question(llm: LLM, question: str, chunks: list[Chunk], cfg: Config) -> str:
    return llm.generate(SYSTEM_PROMPT, build_user_prompt(question, chunks, cfg.retrieval.per_chunk_chars),
                        cfg.llm.max_new_tokens)


def artifact_dir(cfg: Config, repo_name: str) -> Path:
    """artifacts/<repo>/<embedder>/ -- one index per embedder, so e.g. MiniLM and the offline hashing index coexist."""
    slug = cfg.embed.model.replace("/", "__")
    return Path(cfg.artifacts_dir) / repo_name / slug


def prepare(spec: RepoSpec, cfg: Config, embedder: Embedder | None = None) -> tuple[IndexStore, RepoGraph]:
    """Clone -> chunk -> embed -> index -> graph. Saves chunks/embeddings under artifacts/<repo>/."""
    repo_dir, sha = ensure_repo(spec, cfg.repos_dir)
    chunks = chunk_repo(repo_dir, cfg.chunk)
    embedder = embedder or make_embedder(cfg.embed)
    store = IndexStore.build(chunks, embedder, cfg.embed, meta={"repo": spec.name, "commit": sha}, ccfg=cfg.chunk)
    store.save(artifact_dir(cfg, spec.name))
    return store, RepoGraph(repo_dir, store.chunks)


def load_retriever(repo_name: str, cfg: Config, embedder: Embedder | None = None) -> Retriever:
    store = IndexStore.load(artifact_dir(cfg, repo_name), cfg.embed, cfg.chunk)
    repo_dir = Path(cfg.repos_dir) / repo_name
    # The graph is re-parsed from the checkout, so the checkout must be the commit the index was built from.
    head = current_commit(repo_dir)
    if store.meta.get("commit") and head and head != store.meta["commit"]:
        raise RuntimeError(f"{repo_dir} is at {head[:10]} but the index was built at {store.meta['commit'][:10]}. "
                           f"Run: git -C {repo_dir} checkout {store.meta['commit'][:10]}  (or re-run prepare_repo.py)")
    graph = RepoGraph(repo_dir, store.chunks)
    return Retriever(store, embedder or make_embedder(cfg.embed), cfg.retrieval, graph)


def run_strategy(retriever: Retriever, llm: LLM, strategy: str, question: str, cfg: Config
                 ) -> tuple[Retrieval, str]:
    r = retriever.retrieve(strategy, question)
    return r, answer_question(llm, question, r.chunks, cfg)
