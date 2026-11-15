"""Cheap, deterministic, reference-aware metrics. They complement (never replace) the LLM judge.

SWE-QA gives a question and a reference *answer*, but no gold list of files. So we use proxies:

  ref_file_recall      fraction of file paths named in the reference answer that appear in the retrieved context
  ref_ident_recall_ctx fraction of code identifiers in the reference found anywhere in the retrieved text
                       -> measures RETRIEVAL quality (did we even fetch the evidence?)
  ref_ident_recall_ans fraction of those identifiers found in the generated answer
                       -> measures end-to-end quality; the gap between ctx and ans is GENERATION loss
  token_f1             bag-of-words overlap with the reference (crude; penalises paraphrase)
  citation metrics     does the answer cite files/line ranges that were actually in its context?
                       (a hallucinated citation is worse than no citation)
  abstained            did the model say the evidence is insufficient?
  graph_share          fraction of context chunks that were added by graph edges (0 for the baselines)
  seconds              wall-clock time for retrieval + generation
"""
from __future__ import annotations

import re
from collections import Counter

PATH_RE = re.compile(r"(?<![\w/.-])((?:[\w.-]+/)*[\w.-]+\.(?:py|md|rst|txt|toml|cfg|ini|ya?ml|json|html|js))(?!\w)")
# Markdown around a path (`app.py`, **app.py**, "app.py") is skipped before looking for the line range.
_CLOSE_MARKUP = "`*_'\""
LINES_RE = re.compile(r"^\s*[\(\[:,#]?\s*(?:lines?\s*|L)?(\d+)(?:\s*(?:-|\u2013|to)\s*L?(\d+))?", re.I)
TOKEN_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
# "missing" alone is NOT an abstention signal: real answers say things like "raises KeyError if the key is
# missing" (2 of the 48 Flask reference answers would otherwise count as abstentions).
ABSTAIN_RE = re.compile(
    r"(not enough|insufficient|does not (?:contain|provide|include)|do not (?:contain|provide|include)|"
    r"cannot (?:be )?determin|can't determine|unable to determine|no (?:information|evidence)|"
    r"not (?:provided|present|shown|found|included|available) in the (?:provided |given )?(?:excerpts|context|code)|"
    r"(?:is|are) missing from the (?:provided |given )?(?:excerpts|context)|missing (?:information|context))", re.I)


def extract_paths(text: str) -> list[str]:
    return [p.lstrip("./") for p in PATH_RE.findall(text)]


def cited_spans(answer: str) -> list[tuple[str, int | None, int | None]]:
    spans = []
    for m in PATH_RE.finditer(answer):
        tail = LINES_RE.match(answer[m.end(): m.end() + 40].lstrip(_CLOSE_MARKUP))
        s = int(tail.group(1)) if tail else None
        e = int(tail.group(2)) if tail and tail.group(2) else s
        spans.append((m.group(1).lstrip("./"), s, e))
    return spans


def _path_match(cited: str, actual: str) -> bool:
    return actual == cited or actual.endswith("/" + cited) or cited.endswith("/" + actual)


def code_identifiers(text: str) -> set[str]:
    """Identifier-like tokens that look like code (snake_case, CamelCase, backticked, or called)."""
    ids = set()
    for tok in re.findall(r"`([^`\n]+)`", text):
        ids.update(t.lower() for t in TOKEN_RE.findall(tok) if len(t) >= 4)
    for m in TOKEN_RE.finditer(text):
        t = m.group(0)
        called = text[m.end(): m.end() + 1] == "("
        camel = re.search(r"[a-z][A-Z]", t) or (t[:1].isupper() and re.search(r"[a-z]", t) and
                                                 re.search(r"[A-Z]", t[1:]))
        if len(t) >= 4 and ("_" in t or camel or called):
            ids.add(t.lower())
    return ids


def _all_tokens(text: str) -> set[str]:
    return {t.lower() for t in TOKEN_RE.findall(text)}


def token_f1(pred: str, ref: str) -> float:
    p, r = Counter(re.findall(r"\w+", pred.lower())), Counter(re.findall(r"\w+", ref.lower()))
    overlap = sum((p & r).values())
    if not overlap:
        return 0.0
    prec, rec = overlap / sum(p.values()), overlap / sum(r.values())
    return 2 * prec * rec / (prec + rec)


def score_record(rec: dict) -> dict:
    """rec: {answer, reference, retrieved: [{path,start,end,...}], retrieved_text: str}"""
    ans, ref = rec["answer"], str(rec["reference"])
    retrieved = rec["retrieved"]
    r_paths = {r["path"] for r in retrieved}
    ctx_tokens = _all_tokens(rec.get("retrieved_text", ""))

    ref_files = {p for p in extract_paths(ref)}
    file_recall = None
    if ref_files:
        hit = sum(1 for rf in ref_files if any(_path_match(rf, rp) for rp in r_paths))
        file_recall = hit / len(ref_files)

    ref_ids = code_identifiers(ref)
    ident_ctx = ident_ans = None
    if ref_ids:
        ident_ctx = len(ref_ids & ctx_tokens) / len(ref_ids)
        ident_ans = len(ref_ids & _all_tokens(ans)) / len(ref_ids)

    spans = cited_spans(ans)
    file_ok = line_ok = n_lines = 0
    for path, s, e in spans:
        matches = [r for r in retrieved if _path_match(path, r["path"])]
        file_ok += bool(matches)
        if s is not None:
            n_lines += 1
            line_ok += any(r["start"] <= (e or s) and s <= r["end"] for r in matches)
    return {
        "ref_file_recall": file_recall,
        "ref_ident_recall_ctx": ident_ctx,
        "ref_ident_recall_ans": ident_ans,
        "token_f1": token_f1(ans, ref),
        "n_citations": len(spans),
        "cites_any": float(bool(spans)),
        "citation_file_valid": (file_ok / len(spans)) if spans else None,
        "citation_line_valid": (line_ok / n_lines) if n_lines else None,
        "abstained": float(bool(ABSTAIN_RE.search(ans))),
        "graph_share": (sum(r.get("origin") not in (None, "seed", "semantic") for r in retrieved) / len(retrieved))
                       if retrieved else 0.0,
        "seconds": rec.get("seconds"),
        "answer_chars": len(ans),
        "context_chars": len(rec.get("retrieved_text", "")),
        "n_chunks": len(retrieved),
    }


def aggregate(scores: list[dict]) -> dict:
    keys = scores[0].keys() if scores else []
    out = {}
    for k in keys:
        vals = [s[k] for s in scores if s[k] is not None]
        out[k] = (sum(vals) / len(vals)) if vals else None
        out[f"{k}__n"] = len(vals)
    return out
