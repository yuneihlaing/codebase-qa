#!/usr/bin/env python
"""Safety, security and robustness tests (data/probes.jsonl, 12 tests in 8 categories).

  direct_injection         the user tries to override the rules (leak the system prompt, drop citations)
  indirect_injection       a retrieved file contains planted instructions (canary token must not appear)
  hallucination            false premise / non-existent API (must not invent an answer)
  out_of_scope             unrelated requests (must decline or redirect)
  unsafe_request           attacking someone else's site (must refuse)
  data_leakage             a retrieved .env file holds credentials (must not repeat them)
  conflicting_information  a planted note contradicts the code (should follow the code)
  missing_information      a setting that does not exist (must say so)

"Planted" content is injected into the retrieved context as an extra chunk (`inject` in the jsonl), so the
test does not depend on retrieval happening to find it. The automatic verdict is a keyword heuristic: read
the answers and fill `manual_success` / `notes` in the CSV.

    python scripts/run_probes.py --repo flask --strategy baseline_budget structure
Outputs reports/<tag>/safety_<repo>.{md,csv,jsonl}
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
    ap.add_argument("--repo", required=True)
    ap.add_argument("--strategy", nargs="+", default=["baseline_budget", "structure"], choices=STRATEGIES)
    ap.add_argument("--tests", default="data/probes.jsonl")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--tag", default=DEFAULT_TAG)
    a = ap.parse_args()
    cfg = load_config(a.config)
    items = load_items(a.tests)
    rows = run_items(Assistant(a.repo, cfg), items, a.strategy, progress=lambda j: tqdm(j, desc="safety"))
    paths = write_outputs(rows, reports_dir(cfg, a.tag) / f"safety_{a.repo}",
                          f"Safety / robustness tests: {a.repo} ({len(items)} tests x {len(a.strategy)} strategies)")
    print(paths["md"].read_text(encoding="utf-8").split("`auto`")[0])
    print("wrote", ", ".join(str(p) for p in paths.values()))


if __name__ == "__main__":
    main()
