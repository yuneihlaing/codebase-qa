#!/usr/bin/env python
"""Run the hand-written test query set (data/test_queries.jsonl, 24 Flask queries) and build the evaluation
table: query | expected behaviour/answer | response | success | notes.

Categories (from the proposal): common, data_tool, ambiguous, edge_case, unreliable.

    python scripts/run_test_queries.py --repo flask --strategy baseline_budget structure
Outputs reports/<tag>/test_queries_<repo>.{md,csv,jsonl}

Then grade by hand: open the CSV, fill `manual_success` (yes / no / partial) and `notes`, save it, and run
    python scripts/summarize_grades.py reports/default/test_queries_flask.csv
"""
import argparse

from tqdm import tqdm

from codeqa.assistant import Assistant
from codeqa.config import load_config
from codeqa.querysets import load_items, run_items, write_outputs
from codeqa.retrieval import STRATEGIES
from codeqa.runs import DEFAULT_TAG, reports_dir


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="flask")
    ap.add_argument("--strategy", nargs="+", default=["baseline_budget", "structure"], choices=STRATEGIES)
    ap.add_argument("--queries", default="data/test_queries.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    a = ap.parse_args()
    cfg = load_config(a.config)
    items = load_items(a.queries)
    rows = run_items(Assistant(a.repo, cfg), items, a.strategy, progress=lambda j: tqdm(j, desc="queries"))
    paths = write_outputs(rows, reports_dir(cfg, a.tag) / f"test_queries_{a.repo}",
                          f"Test query set: {a.repo} ({len(items)} queries x {len(a.strategy)} strategies)")
    print(paths["md"].read_text(encoding="utf-8").split("`auto`")[0])
    print("wrote", ", ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    main()
