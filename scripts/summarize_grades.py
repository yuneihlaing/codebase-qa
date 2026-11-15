#!/usr/bin/env python
"""Success rates from a hand-graded CSV (test queries or safety tests).

    python scripts/summarize_grades.py reports/default/test_queries_flask.csv reports/default/safety_flask.csv

For each row the `manual_success` column (yes / no / partial) is used when filled, otherwise the automatic
verdict. Also lists the rows where your grade disagrees with the automatic one: useful for the report's
"limitations of the automatic checks" paragraph.
"""
import argparse

from codeqa.querysets import grade, read_csv, summary_markdown


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("csv", nargs="+")
    a = ap.parse_args()
    for path in a.csv:
        rows = read_csv(path)
        print(f"## {path}\n")
        print(summary_markdown(rows))
        flips = [r for r in rows if str(r.get("manual_success", "")).strip()
                 and r.get("auto_success") and grade(r) != {"PASS": 1.0, "FAIL": 0.0}.get(r["auto_success"])]
        if flips:
            print(f"\nHand grade differs from the automatic check in {len(flips)} rows:")
            for r in flips:
                print(f"  {r['id']} [{r['strategy']}] auto={r['auto_success']} manual={r['manual_success']}  "
                      f"{r.get('notes', '')}")
        print()


if __name__ == "__main__":
    main()
