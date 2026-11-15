#!/usr/bin/env python
"""Compare embedders (or any retrieval settings) on retrieval quality alone: no generator, minutes not hours.

    python scripts/compare_retrieval.py --repo flask --configs configs/default.yaml configs/minilm.yaml configs/coderank.yaml
    python scripts/compare_retrieval.py --repo flask --configs configs/default.yaml configs/minilm.yaml --strategies baseline_budget

For each config: builds its index if missing (artifacts/<repo>/<embedder>/), retrieves context for every SWE-QA
question with each strategy, and scores only what retrieval controls: did the excerpts contain the files and the
code identifiers named in the reference answer? Every config is compared with the FIRST one, strategy by strategy,
with a paired bootstrap 95% interval.

Writes reports/retrieval_comparison_<repo>.md. Use it to justify the embedder choice before the long generation run.
"""
import argparse
import gc
import time
from pathlib import Path

from codeqa.config import load_config
from codeqa.dataset import load_swe_qa
from codeqa.eval.deterministic import aggregate, score_record
from codeqa.eval.stats import paired_bootstrap, win_tie_loss
from codeqa.pipeline import artifact_dir, load_retriever, prepare, retrieval_record
from codeqa.repos import get_spec
from codeqa.retrieval import STRATEGIES

METRICS = ["ref_file_recall", "ref_ident_recall_ctx", "graph_share", "context_chars", "n_chunks"]
PAIRED = ["ref_file_recall", "ref_ident_recall_ctx"]


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="flask")
    ap.add_argument("--configs", nargs="+", required=True, help="first one = the reference")
    ap.add_argument("--strategies", nargs="+", default=["baseline_budget", "structure"], choices=STRATEGIES)
    ap.add_argument("--limit", type=int)
    a = ap.parse_args()
    questions = load_swe_qa(a.repo, a.limit)
    names = [Path(c).stem for c in a.configs]
    per_q, timing, models = {}, {}, {}
    for path, name in zip(a.configs, names):
        cfg = load_config(path)
        models[name] = cfg.embed.model
        if not (artifact_dir(cfg, a.repo) / "meta.json").exists():
            print(f"[{name}] building index with {cfg.embed.model} ...")
            prepare(get_spec(a.repo), cfg)
        retriever = load_retriever(a.repo, cfg)
        for strategy in a.strategies:
            t0 = time.time()
            per_q[(name, strategy)] = {
                q["id"]: score_record(retrieval_record(q, retriever.retrieve(strategy, q["question"]),
                                                       cfg.retrieval.per_chunk_chars)) for q in questions}
            timing[(name, strategy)] = (time.time() - t0) / max(len(questions), 1)
        del retriever
        gc.collect()
        try:
            import torch
            torch.cuda.empty_cache()
        except Exception:
            pass

    ids = [q["id"] for q in questions]
    lines = [f"# Retrieval comparison: {a.repo} ({len(ids)} SWE-QA questions, no generator)", "",
             "| config | embedder | strategy | " + " | ".join(METRICS) + " | sec/question |",
             "|---|---|---|" + "---|" * (len(METRICS) + 1)]
    for name in names:
        for s in a.strategies:
            agg = aggregate([per_q[(name, s)][i] for i in ids])
            lines.append(f"| {name} | `{models[name]}` | {s} | " + " | ".join(fmt(agg[m]) for m in METRICS)
                         + f" | {timing[(name, s)]:.3f} |")
    base = names[0]
    lines += ["", f"## Paired comparison vs `{base}` (same strategy; positive = better than `{base}`)", "",
              "| config | strategy | metric | mean diff | 95% CI | wins/ties/losses | n |", "|---|---|---|---|---|---|---|"]
    for name in names[1:]:
        for s in a.strategies:
            for m in PAIRED:
                both = [i for i in ids if per_q[(base, s)][i][m] is not None and per_q[(name, s)][i][m] is not None]
                if len(both) < 5:
                    continue
                x = [per_q[(base, s)][i][m] for i in both]
                y = [per_q[(name, s)][i][m] for i in both]
                b, w = paired_bootstrap(x, y), win_tie_loss(x, y)
                lines.append(f"| {name} | {s} | {m} | {b['mean_diff']:+.3f} | [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] | "
                             f"{w['b_wins']}/{w['ties']}/{w['a_wins']} | {b['n']} |")
    out = Path(load_config(a.configs[0]).reports_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"retrieval_comparison_{a.repo}.md"
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {path}")


if __name__ == "__main__":
    main()
