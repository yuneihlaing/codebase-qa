"""Run a hand-written query set (test queries or safety tests) and write the evaluation table.

The table has exactly the columns the course asks for: the query, the expected behaviour/answer, the
assistant's response, whether it was successful, and notes. `auto_success` is a heuristic first pass
(eval/behaviour.py); fill `manual_success` (yes / no / partial) and `notes` by hand in the CSV, then run
scripts/summarize_grades.py.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

from .assistant import Assistant
from .chunking import Chunk
from .eval.behaviour import check_item

COLUMNS = ["id", "category", "strategy", "query", "expected", "response", "auto_success", "auto_reason",
           "manual_success", "notes", "top_sources", "seconds"]


def load_items(path: str | Path) -> list[dict]:
    items = []
    for line in open(path, encoding="utf-8"):
        line = line.strip()
        if line:
            it = json.loads(line)
            it.setdefault("query", it.get("question"))
            items.append(it)
    ids = [it["id"] for it in items]
    if len(ids) != len(set(ids)):
        raise ValueError(f"Duplicate ids in {path}")
    return items


def injected_chunk(inject: dict) -> Chunk:
    """A fake repository file that is forced into the context (indirect injection / conflict / leakage tests)."""
    text = inject["text"]
    path = inject.get("path", "docs/NOTES.md")
    return Chunk(path, 1, text.count("\n") + 1, Path(path).name, "text", text)


def run_items(assistant: Assistant, items: list[dict], strategies: list[str], progress=None) -> list[dict]:
    rows = []
    jobs = [(s, it) for s in strategies for it in items]
    for strategy, it in (progress(jobs) if progress else jobs):
        extra = [injected_chunk(it["inject"])] if it.get("inject") else None
        ans = assistant.ask(it["query"], strategy, extra_chunks=extra)
        paths = [c.path for c in ans.chunks]
        ok, reason = check_item(it, ans.text, paths)
        rows.append({
            "id": it["id"], "category": it["category"], "strategy": strategy, "query": it["query"],
            "expected": it.get("expected", ""), "response": ans.text,
            "auto_success": {True: "PASS", False: "FAIL", None: ""}[ok], "auto_reason": reason,
            "manual_success": "", "notes": "",
            "top_sources": "; ".join(f"{c.path}:{c.start}-{c.end} [{o}]"
                                     for c, o in list(zip(ans.chunks, ans.origins))[:6]),
            "seconds": ans.seconds, "sources": ans.sources(),
        })
    return rows


def _cell(text: str, limit: int | None = None) -> str:
    t = " ".join(str(text).split())
    if limit and len(t) > limit:
        t = t[:limit] + " ..."
    return t.replace("|", "\\|")


def write_outputs(rows: list[dict], out_stem: Path, title: str) -> dict[str, Path]:
    """Writes <stem>.jsonl (everything), <stem>.csv (for hand grading) and <stem>.md (readable table)."""
    out_stem.parent.mkdir(parents=True, exist_ok=True)
    paths = {"jsonl": out_stem.with_suffix(".jsonl"), "csv": out_stem.with_suffix(".csv"),
             "md": out_stem.with_suffix(".md")}
    with open(paths["jsonl"], "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r) + "\n")
    with open(paths["csv"], "w", newline="", encoding="utf-8-sig") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    lines = [f"# {title}", "", summary_markdown(rows), "",
             "`auto` is a keyword heuristic. Final grades go in the `manual_success` column of the CSV.", "",
             "| id | category | strategy | query | expected | response (truncated) | auto | why |",
             "|---|---|---|---|---|---|---|---|"]
    for r in rows:
        lines.append(f"| {r['id']} | {r['category']} | {r['strategy']} | {_cell(r['query'])} | "
                     f"{_cell(r['expected'], 220)} | {_cell(r['response'], 400)} | {r['auto_success']} | "
                     f"{_cell(r['auto_reason'], 120)} |")
    paths["md"].write_text("\n".join(lines) + "\n", encoding="utf-8")
    return paths


def grade(row: dict) -> float | None:
    """Manual grade if given (yes/no/partial), else the automatic one. 1 = success, 0.5 = partial."""
    m = str(row.get("manual_success", "")).strip().lower()
    if m in ("y", "yes", "1", "pass", "true", "success", "ok"):
        return 1.0
    if m in ("n", "no", "0", "fail", "false"):
        return 0.0
    if m in ("p", "partial", "0.5", "half"):
        return 0.5
    a = str(row.get("auto_success", "")).strip().upper()
    return {"PASS": 1.0, "FAIL": 0.0}.get(a)


def summary_markdown(rows: list[dict]) -> str:
    strategies = sorted({r["strategy"] for r in rows}, key=[r["strategy"] for r in rows].index)
    cats = sorted({r["category"] for r in rows}, key=[r["category"] for r in rows].index)
    table = defaultdict(list)
    manual = sum(1 for r in rows if str(r.get("manual_success", "")).strip())
    for r in rows:
        g = grade(r)
        if g is not None:
            table[(r["category"], r["strategy"])].append(g)
            table[("ALL", r["strategy"])].append(g)
    out = [f"Success rate by category ({manual}/{len(rows)} rows hand-graded; the rest use the automatic check)",
           "", "| category | " + " | ".join(strategies) + " |", "|---|" + "---|" * len(strategies)]
    for c in cats + ["ALL"]:
        cells = []
        for s in strategies:
            v = table.get((c, s), [])
            cells.append(f"{sum(v):g}/{len(v)}" if v else "n/a")
        out.append(f"| {c} | " + " | ".join(cells) + " |")
    return "\n".join(out)


def read_csv(path: str | Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8-sig") as f:
        return list(csv.DictReader(f))
