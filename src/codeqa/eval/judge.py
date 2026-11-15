"""LLM-as-judge (reference-based). Mirrors the five SWE-QA dimensions on a 1-5 scale.

Good practice baked in:
  * the judge sees the reference answer, so it grades agreement with ground truth rather than style
  * the judge should be a *different model family* than the generator (self-preference bias)
  * pairwise comparisons are run in BOTH orders; disagreement => tie (position bias)
  * the judge is told to ignore length and formatting
Always spot-check ~20 judgements by hand before trusting the numbers.
"""
from __future__ import annotations

import json
import re

from ..llm import LLM

DIMENSIONS = ("correctness", "completeness", "relevance", "clarity", "reasoning")

POINTWISE_SYSTEM = "You are a strict, fair grader of answers to questions about a software repository."
POINTWISE_TEMPLATE = """Grade the CANDIDATE answer against the REFERENCE answer.

Question:
{question}

Reference answer (treat as ground truth):
{reference}

Candidate answer:
{answer}

Score each dimension from 1 (very poor) to 5 (excellent):
- correctness: facts, file names and behaviour agree with the reference; no invented details
- completeness: covers the key points of the reference
- relevance: stays on the question asked
- clarity: easy to follow
- reasoning: explains how the pieces of code fit together

Ignore answer length and formatting. Penalise confident claims that contradict the reference.
Reply with ONLY a JSON object: {{"correctness": n, "completeness": n, "relevance": n, "clarity": n, "reasoning": n, "rationale": "<one sentence>"}}"""

PAIRWISE_TEMPLATE = """Two candidate answers (A and B) respond to the same question about a software repository.
Decide which is closer to the REFERENCE in correctness and completeness. Ignore length and formatting.

Question:
{question}

Reference answer (ground truth):
{reference}

Answer A:
{a}

Answer B:
{b}

Reply with ONLY a JSON object: {{"winner": "A" | "B" | "tie", "rationale": "<one sentence>"}}"""


def _parse_json(text: str) -> dict | None:
    m = re.search(r"\{.*\}", text, re.S)
    if not m:
        return None
    try:
        return json.loads(m.group(0))
    except json.JSONDecodeError:
        return None


def judge_pointwise(judge: LLM, question: str, reference: str, answer: str, max_new_tokens: int = 600) -> dict:
    raw = judge.generate(POINTWISE_SYSTEM, POINTWISE_TEMPLATE.format(
        question=question, reference=reference, answer=answer), max_new_tokens)
    parsed = _parse_json(raw) or {}
    out = {d: parsed.get(d) for d in DIMENSIONS}
    vals = [v for v in out.values() if isinstance(v, (int, float))]
    out["overall"] = sum(vals) / len(vals) if len(vals) == len(DIMENSIONS) else None
    out["rationale"] = parsed.get("rationale", "")
    out["parse_ok"] = out["overall"] is not None
    return out


def judge_pairwise(judge: LLM, question: str, reference: str, ans_a: str, ans_b: str,
                   max_new_tokens: int = 600) -> str:
    """Returns 'A', 'B', 'tie' or 'invalid' after running both orders (position-bias control).

    'invalid' means the judge's reply could not be parsed in at least one order. It is reported separately
    instead of being folded into 'tie', so a broken judge cannot masquerade as "no difference".
    """
    def one(x, y):
        raw = judge.generate(POINTWISE_SYSTEM, PAIRWISE_TEMPLATE.format(
            question=question, reference=reference, a=x, b=y), max_new_tokens)
        w = (_parse_json(raw) or {}).get("winner")
        return w if w in ("A", "B", "tie") else None
    first, second_raw = one(ans_a, ans_b), one(ans_b, ans_a)     # second run: judge's A is ans_b
    if first is None or second_raw is None:
        return "invalid"
    second = {"A": "B", "B": "A", "tie": "tie"}[second_raw]
    return first if first == second else "tie"
