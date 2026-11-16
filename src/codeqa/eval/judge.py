"""LLM-as-judge (reference-based). Mirrors the five SWE-QA dimensions on a 1-5 scale.

Good practice baked in:
  * the judge sees the reference answer, so it grades agreement with ground truth rather than style
  * the judge should be a *different model family* than the generator (self-preference bias)
  * pairwise comparisons are run in BOTH orders; disagreement => tie (position bias)
  * the judge is told to ignore length and formatting
  * every reply is saved (CachedLLM), so a run stopped by an API limit resumes and reports can be rebuilt
Always spot-check ~20 judgements by hand before trusting the numbers.
"""
from __future__ import annotations

import hashlib
import json
import re
import sys
from pathlib import Path

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


class CachedLLM:
    """Wraps the judge so each reply is written to a JSONL file the moment it arrives.

    Running the same evaluation again reads replies from the file instead of calling the API, so:
      * a run stopped by a free-tier limit continues where it stopped (nothing already paid for is lost);
      * the report can be rebuilt for free, with exactly the same judge replies.
    The key covers `label` (model and thinking settings) and the exact prompt, so changing the judge or an
    answer asks again. Empty replies are not saved, so they are retried next time.
    """

    def __init__(self, llm: LLM, path: str | Path, label: str = ""):
        self.llm, self.path, self.label = llm, Path(path), label or llm.name
        self.name = llm.name
        self.cache: dict[str, str] = {}
        self.hits = self.calls = 0
        if self.path.exists():
            with open(self.path, encoding="utf-8") as f:
                for line in f:
                    try:
                        r = json.loads(line)
                    except json.JSONDecodeError:
                        continue                    # e.g. a last line cut off when Colab disconnected
                    self.cache[r["key"]] = r["reply"]

    def _key(self, system: str, user: str, max_new_tokens: int) -> str:
        raw = json.dumps([self.label, system, user, max_new_tokens], ensure_ascii=False)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def generate(self, system: str, user: str, max_new_tokens: int = 700) -> str:
        key = self._key(system, user, max_new_tokens)
        if key in self.cache:
            self.hits += 1
            return self.cache[key]
        reply = self.llm.generate(system, user, max_new_tokens)
        self.calls += 1
        if reply.strip():
            self.cache[key] = reply
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with open(self.path, "a", encoding="utf-8") as f:
                f.write(json.dumps({"key": key, "judge": self.label, "reply": reply}, ensure_ascii=False) + "\n")
        if self.calls % 10 == 0:
            print(f"[judge] {self.calls} API calls so far, {self.hits} replies reused from {self.path.name}",
                  file=sys.stderr)
        return reply
