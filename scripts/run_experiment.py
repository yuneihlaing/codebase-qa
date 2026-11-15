#!/usr/bin/env python
"""Run one or more retrieval strategies over SWE-QA questions with the SAME generator and settings.

    python scripts/run_experiment.py --repo flask --strategy baseline baseline_budget structure
    python scripts/run_experiment.py --repo flask --strategy baseline_budget structure --limit 5
    python scripts/run_experiment.py --repo flask --strategy structure --config configs/my_sweep.yaml --tag alpha0.8

Writes results/<tag>/<repo>__<strategy>.jsonl (one line per question) and results/<tag>/config_used.json.
Re-running with the same tag resumes where it stopped. A different setting needs a different --tag: the
script refuses to mix settings inside one tag, so a sweep can never silently reuse old answers.
"""
import argparse
import json
import sys
import time

from tqdm import tqdm

from codeqa.config import load_config
from codeqa.dataset import load_swe_qa
from codeqa.llm import make_llm
from codeqa.pipeline import answer_question, format_context, load_retriever
from codeqa.retrieval import STRATEGIES
from codeqa.runs import DEFAULT_TAG, ConfigMismatch, claim_run_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", nargs="+", required=True)
    ap.add_argument("--strategy", nargs="+", default=["baseline", "baseline_budget", "structure"], choices=STRATEGIES)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default=DEFAULT_TAG, help="name of this setting; results go to results/<tag>/")
    a = ap.parse_args()

    cfg = load_config(a.config)
    try:
        out_dir = claim_run_dir(cfg, a.tag)
    except ConfigMismatch as e:
        sys.exit(f"error: {e}")
    llm = make_llm(cfg.llm)                      # loaded ONCE and shared -> identical generator for all strategies

    for repo in a.repo:
        retriever = load_retriever(repo, cfg)
        questions = load_swe_qa(repo, a.limit)
        for strategy in a.strategy:
            path = out_dir / f"{repo}__{strategy}.jsonl"
            done = {json.loads(line)["id"] for line in open(path, encoding="utf-8")} if path.exists() else set()
            todo = [q for q in questions if q["id"] not in done]
            with open(path, "a", encoding="utf-8") as f:
                for q in tqdm(todo, desc=f"{a.tag}/{repo}/{strategy}"):
                    t0 = time.time()
                    r = retriever.retrieve(strategy, q["question"])
                    ans = answer_question(llm, q["question"], r.chunks, cfg)
                    f.write(json.dumps({
                        **q, "strategy": strategy, "tag": a.tag, "generator": getattr(llm, "name", "?"),
                        "answer": ans,
                        "retrieved": [{"path": c.path, "start": c.start, "end": c.end, "name": c.name,
                                       "kind": c.kind, "origin": o, "score": round(s, 4)}
                                      for c, o, s in zip(r.chunks, r.origins, r.scores)],
                        "retrieved_text": format_context(r.chunks, cfg.retrieval.per_chunk_chars),
                        "seconds": round(time.time() - t0, 2)}) + "\n")
                    f.flush()
    print(f"results in {out_dir}")


if __name__ == "__main__":
    main()
