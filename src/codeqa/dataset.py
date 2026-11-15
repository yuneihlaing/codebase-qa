"""SWE-QA loader.

The questions ship with this project in data/swe_qa/<repo>.jsonl (copied from the benchmark's GitHub repo,
Apache-2.0; see data/swe_qa/SOURCE.md). Reading the local copy means:
  * no HuggingFace download is needed (useful on restricted networks),
  * question ids (flask-000 ... flask-047) are always assigned from the same file order.
If a repo has no local file, the HuggingFace dataset is used instead (one split per repo).
"""
from __future__ import annotations

import json
from pathlib import Path

QUESTION_FIELD = "question"
ANSWER_FIELD = "answer"
LOCAL_DIR = Path(__file__).resolve().parents[2] / "data" / "swe_qa"
HF_NAME = "swe-qa/SWE-QA-Benchmark"


def _rows_local(repo: str) -> list[dict] | None:
    p = LOCAL_DIR / f"{repo}.jsonl"
    if not p.exists():
        return None
    return [json.loads(line) for line in open(p, encoding="utf-8") if line.strip()]


def _rows_hf(repo: str) -> list[dict]:
    from datasets import load_dataset  # imported lazily so the rest of the package works without it

    ds = load_dataset(HF_NAME)
    if repo not in ds:
        raise KeyError(f"No split {repo!r} in SWE-QA. Available: {list(ds.keys())}")
    return list(ds[repo])


def load_swe_qa(repo: str, limit: int | None = None, source: str = "auto") -> list[dict]:
    """source: 'auto' (local file, else HuggingFace), 'local' or 'hf'."""
    rows = None if source == "hf" else _rows_local(repo)
    if rows is None:
        if source == "local":
            raise FileNotFoundError(f"{LOCAL_DIR / (repo + '.jsonl')} not found (run scripts/download_swe_qa.py)")
        rows = _rows_hf(repo)
    cols = set(rows[0].keys()) if rows else set()
    for f in (QUESTION_FIELD, ANSWER_FIELD):
        if f not in cols:
            raise KeyError(f"Expected column {f!r}; found {sorted(cols)}. Update dataset.py.")
    out = [{"id": f"{repo}-{i:03d}", "repo": repo, "question": r[QUESTION_FIELD], "reference": r[ANSWER_FIELD]}
           for i, r in enumerate(rows)]
    return out[:limit] if limit else out


def available_repos() -> list[str]:
    local = sorted(p.stem for p in LOCAL_DIR.glob("*.jsonl"))
    if local:
        return local
    from datasets import load_dataset
    return list(load_dataset(HF_NAME).keys())
