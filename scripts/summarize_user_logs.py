#!/usr/bin/env python
"""Summarize peer-testing logs (results/user_testing/*.jsonl) for docs/peer_testing.md.

    python scripts/summarize_user_logs.py
    python scripts/summarize_user_logs.py --dir results/user_testing --show-comments
"""
import argparse
import json
from pathlib import Path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", default="results/user_testing")
    ap.add_argument("--show-comments", action="store_true")
    a = ap.parse_args()
    files = sorted(Path(a.dir).glob("*.jsonl"))
    if not files:
        print(f"No logs in {a.dir}. Testers run: python scripts/ask.py --repo flask --tester NAME")
        return
    print("| Tester | # questions | # rated | Avg rating | Strategies |\n|---|---|---|---|---|")
    comments = []
    for f in files:
        rows = [json.loads(line) for line in open(f, encoding="utf-8") if line.strip()]
        asked = [r for r in rows if r.get("event", "asked") == "asked" or "event" not in r]
        rated = [r for r in rows if str(r.get("rating") or "").strip().isdigit()]
        avg = sum(int(r["rating"]) for r in rated) / len(rated) if rated else None
        strategies = sorted({r["strategy"] for r in rows})
        questions = {r["question"] for r in asked}
        print(f"| {f.stem} | {len(questions)} | {len(rated)} | {'n/a' if avg is None else f'{avg:.1f}'} | "
              f"{', '.join(strategies)} |")
        comments += [(f.stem, r["rating"], r["question"], r["comment"]) for r in rows if r.get("comment")]
    if a.show_comments and comments:
        print("\nComments:")
        for who, rating, q, c in comments:
            print(f"- {who} (rating {rating}) on \"{q}\": {c}")


if __name__ == "__main__":
    main()
