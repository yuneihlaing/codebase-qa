"""All tunable settings live here, as plain dataclasses that can be overridden from YAML.

Design rule: a number that affects results must be a field here, not a literal buried in code.
That makes every experiment reproducible from (config file + git commit).

A YAML file may start with `extends: other.yaml` to inherit another file and override only a few keys
(see configs/offline.yaml and configs/small_model.yaml).
"""
from __future__ import annotations

import dataclasses
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml


@dataclass
class ChunkConfig:
    max_class_lines: int = 80
    window: int = 40
    overlap: int = 10
    text_exts: tuple = (".md", ".rst", ".txt", ".toml", ".cfg", ".ini", ".yml", ".yaml")
    skip_dirs: tuple = (".git", "node_modules", "__pycache__", ".tox", "build", "dist", ".venv", "venv")
    include_module_code: bool = False
    max_file_bytes: int = 1_000_000


QWEN3_QUERY_PROMPT = ("Instruct: Given a question about a Python code repository, retrieve the code or "
                      "documentation that answers it\nQuery:")


@dataclass
class EmbedConfig:
    model: str = "Qwen/Qwen3-Embedding-0.6B"
    max_chars: int = 4000                 # characters of "<path> <name>\n<code>" embedded per chunk (99% of Flask's chunks fit)
    max_seq_length: int | None = 1024     # token cap inside the embedder (None = the model's own limit)
    batch_size: int = 16
    trust_remote_code: bool = False
    query_prompt: str = QWEN3_QUERY_PROMPT  # prefixed to questions only; "" = embed questions as they are


@dataclass
class RetrievalConfig:
    k: int = 6
    context_char_budget: int = 18000
    per_chunk_chars: int = 3000
    pool: int = 60
    k_seed: int = 4
    seed_from: str = "code"          # "code": seeds are the top Python chunks; "any": top chunks of any kind
    max_expansion: int = 8
    reserve_expansion: int = 4       # best graph neighbours placed right after the seeds (0 = compete on score)
    max_in_edges: int = 20           # "called_by" neighbours kept per seed, chosen by similarity to the question
    alpha: float = 0.5
    edge_weights: dict = field(default_factory=lambda: {
        "call": 1.0, "ref": 0.8, "called_by": 0.7, "parent": 0.9, "header": 0.4})


@dataclass
class LLMConfig:
    backend: str = "hf"
    model: str = "Qwen/Qwen2.5-Coder-3B-Instruct"
    max_new_tokens: int = 700
    thinking_budget: int | None = 0  # Gemini 2.x only: 0 turns "thinking" off so it can't eat max_new_tokens


@dataclass
class JudgeConfig:
    backend: str = "gemini"
    model: str = "gemini-3.5-flash-lite"  # override for one run with the JUDGE_MODEL environment variable
    max_new_tokens: int = 600             # length of the visible reply (thinking gets extra room on top)
    thinking_level: str | None = "low"    # Gemini 3+: minimal | low | medium | high (not every model has all)
    thinking_budget: int | None = None    # Gemini 2.x only (older models); ignored by Gemini 3+
    requests_per_minute: float | None = None   # pace API calls to stay under the project's RPM limit


@dataclass
class Config:
    repos_dir: str = "repos"
    artifacts_dir: str = "artifacts"
    results_dir: str = "results"
    reports_dir: str = "reports"
    chunk: ChunkConfig = field(default_factory=ChunkConfig)
    embed: EmbedConfig = field(default_factory=EmbedConfig)
    retrieval: RetrievalConfig = field(default_factory=RetrievalConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    judge: JudgeConfig = field(default_factory=JudgeConfig)


def _update(obj: Any, values: dict) -> None:
    for key, val in values.items():
        if not hasattr(obj, key):
            raise KeyError(f"Unknown config key: {key!r} (in {type(obj).__name__})")
        cur = getattr(obj, key)
        if dataclasses.is_dataclass(cur) and isinstance(val, dict):
            _update(cur, val)
        elif isinstance(cur, tuple) and isinstance(val, list):
            setattr(obj, key, tuple(val))
        elif isinstance(cur, dict) and isinstance(val, dict):
            setattr(obj, key, {**cur, **val})
        else:
            setattr(obj, key, val)


def _read_yaml(path: Path, seen: tuple = ()) -> list[dict]:
    """Return the chain of YAML dicts, base first, following `extends:`."""
    path = Path(path).resolve()
    if path in seen:
        raise ValueError(f"Circular `extends:` involving {path}")
    with open(path, encoding="utf-8") as f:
        data = yaml.safe_load(f) or {}
    parent = data.pop("extends", None)
    chain = _read_yaml(path.parent / parent, seen + (path,)) if parent else []
    return chain + [data]


def load_config(path: str | Path | None = None, overrides: dict | None = None) -> Config:
    cfg = Config()
    if path:
        for layer in _read_yaml(Path(path)):
            _update(cfg, layer)
    if overrides:
        _update(cfg, overrides)
    return cfg


def to_dict(cfg: Config) -> dict:
    return dataclasses.asdict(cfg)


# Sections that change what the generator sees or says. If any of these differ, results are not comparable.
RESULT_SECTIONS = ("chunk", "embed", "retrieval", "llm")


def result_fingerprint(cfg: Config) -> dict:
    d = to_dict(cfg)
    return {k: d[k] for k in RESULT_SECTIONS}
