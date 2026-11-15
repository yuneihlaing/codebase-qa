#!/usr/bin/env python
"""Score SWE-QA result files and compare strategies (and/or settings).

    python scripts/evaluate.py --repo flask --strategies baseline baseline_budget structure
    python scripts/evaluate.py --repo flask --strategies baseline_budget structure --judge --pairwise
    python scripts/evaluate.py --repo flask --strategies qwen3b/structure qwen1.5b/structure --base qwen3b/structure

Each entry in --strategies is `strategy` (read from results/<--tag>/) or `tag/strategy` (read from
results/<tag>/), so the same script compares retrieval strategies *and* generators/settings.

Paired comparisons are made against --base. By default that is `baseline_budget` when it is listed: it uses
the same context budget as `structure`, so it is the fair control. (Plain `baseline` gets half the context.)

Writes reports/<--tag>/summary_<repo>.md and prints it.
"""
import argparse
import json
import sys
from pathlib import Path

from codeqa.config import load_config
from codeqa.eval.deterministic import aggregate, score_record
from codeqa.eval.judge import DIMENSIONS, judge_pairwise, judge_pointwise
from codeqa.eval.stats import paired_bootstrap, win_tie_loss
from codeqa.llm import make_judge
from codeqa.runs import DEFAULT_TAG, reports_dir, results_dir

METRICS = ["ref_file_recall", "ref_ident_recall_ctx", "ref_ident_recall_ans", "token_f1",
           "cites_any", "citation_file_valid", "citation_line_valid", "abstained",
           "context_chars", "n_chunks", "graph_share", "seconds"]
PAIRED = ["ref_file_recall", "ref_ident_recall_ctx", "ref_ident_recall_ans", "token_f1", "citation_file_valid",
          "citation_line_valid"]


def load(path):
    if not path.exists():
        sys.exit(f"error: {path} not found. Run scripts/run_experiment.py first (same --tag/--config).")
    with open(path, encoding="utf-8") as f:
        return {r["id"]: r for r in map(json.loads, f)}


def fmt(x):
    return "n/a" if x is None else f"{x:.3f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", required=True)
    ap.add_argument("--strategies", nargs="+", required=True, help="strategy or tag/strategy")
    ap.add_argument("--base", help="entry to compare the others against (default: baseline_budget if listed)")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    ap.add_argument("--judge", action="store_true", help="pointwise LLM-judge scores (costs API calls)")
    ap.add_argument("--pairwise", action="store_true", help="pairwise judge: base vs each other entry")
    a = ap.parse_args()
    cfg = load_config(a.config)

    def path_of(entry):
        tag, _, strat = entry.rpartition("/")
        return results_dir(cfg, tag or a.tag) / f"{a.repo}__{strat}.jsonl"

    names = a.strategies
    res = {s: load(path_of(s)) for s in names}
    base = a.base or next((s for s in names if s.rpartition("/")[2] == "baseline_budget"), names[0])
    if base not in names:
        sys.exit(f"error: --base {base!r} must be one of --strategies")
    others = [s for s in names if s != base]
    ids = sorted(set.intersection(*[set(r) for r in res.values()]))     # only questions every entry answered
    lines = [f"# Results: {a.repo} ({len(ids)} questions answered by all entries)", ""]

    per_q = {s: {i: score_record(res[s][i]) for i in ids} for s in names}
    lines += ["## Deterministic metrics (mean over questions where defined)", "",
              "| metric | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
    aggs = {s: aggregate([per_q[s][i] for i in ids]) for s in names}
    for m in METRICS:
        lines.append(f"| {m} | " + " | ".join(fmt(aggs[s].get(m)) for s in names) + " |")

    lines += ["", f"## Paired comparison vs `{base}` (positive = better than `{base}`; 95% bootstrap CI)", "",
              "| entry | metric | mean diff | 95% CI | wins/ties/losses | n |", "|---|---|---|---|---|---|"]
    for s in others:
        for m in PAIRED:
            both = [i for i in ids if per_q[base][i][m] is not None and per_q[s][i][m] is not None]
            if len(both) < 5:
                continue
            x, y = [per_q[base][i][m] for i in both], [per_q[s][i][m] for i in both]
            b, w = paired_bootstrap(x, y), win_tie_loss(x, y)
            lines.append(f"| {s} | {m} | {b['mean_diff']:+.3f} | [{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] | "
                         f"{w['b_wins']}/{w['ties']}/{w['a_wins']} | {b['n']} |")

    out = reports_dir(cfg, a.tag)
    out.mkdir(parents=True, exist_ok=True)
    if a.judge or a.pairwise:
        judge = make_judge(cfg.judge)
    if a.judge:
        jr = {s: {i: judge_pointwise(judge, res[s][i]["question"], str(res[s][i]["reference"]),
                                     res[s][i]["answer"], cfg.judge.max_new_tokens) for i in ids}
              for s in names}
        (out / f"judge_pointwise_{a.repo}.json").write_text(json.dumps(jr, indent=2), encoding="utf-8")
        lines += ["", f"## LLM judge (1-5, pointwise; judge = {cfg.judge.backend}:{cfg.judge.model})", "",
                  "| dimension | " + " | ".join(names) + " |", "|---|" + "---|" * len(names)]
        for d in (*DIMENSIONS, "overall"):
            row = []
            for s in names:
                v = [jr[s][i][d] for i in ids if jr[s][i][d] is not None]
                row.append(fmt(sum(v) / len(v)) if v else "n/a")
            lines.append(f"| {d} | " + " | ".join(row) + " |")
        lines.append("| unparsable judge replies | " +
                     " | ".join(str(sum(not jr[s][i]["parse_ok"] for i in ids)) for s in names) + " |")
        for s in others:
            both = [i for i in ids if jr[base][i]["overall"] is not None and jr[s][i]["overall"] is not None]
            if len(both) >= 5:
                b = paired_bootstrap([jr[base][i]["overall"] for i in both], [jr[s][i]["overall"] for i in both])
                lines.append(f"\n`{s}` vs `{base}` overall: {b['mean_diff']:+.3f} "
                             f"[{b['ci_low']:+.3f}, {b['ci_high']:+.3f}] (n={b['n']})")
    if a.pairwise:
        lines += ["", "## Pairwise judge (both orders; disagreement counted as tie)", ""]
        for s in others:
            tally = {"A": 0, "B": 0, "tie": 0, "invalid": 0}
            for i in ids:
                tally[judge_pairwise(judge, res[base][i]["question"], str(res[base][i]["reference"]),
                                     res[base][i]["answer"], res[s][i]["answer"], cfg.judge.max_new_tokens)] += 1
            lines.append(f"- `{base}` wins {tally['A']}, `{s}` wins {tally['B']}, ties {tally['tie']}, "
                         f"unparsable {tally['invalid']}")

    (out / f"summary_{a.repo}.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    print(f"\nsaved {out / f'summary_{a.repo}.md'}")


if __name__ == "__main__":
    main()
