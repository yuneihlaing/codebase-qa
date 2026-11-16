"""LLM backends behind one tiny interface: `generate(system, user, max_new_tokens) -> str`.

The generator (answers questions) and the judge (grades answers) both use it, so you can mix backends:
e.g. local Qwen as generator, a hosted model as judge. Greedy / temperature-0 decoding everywhere so a
re-run gives (nearly) the same answers.

Hosted backends retry transient errors (rate limits on free tiers are common) with exponential backoff.
A daily quota that has run out is not retried: waiting a minute does not help, so the error is raised at once.
"""
from __future__ import annotations

import os
import re
import sys
import time
from typing import Callable, Protocol

from .config import JudgeConfig, LLMConfig


class LLM(Protocol):
    name: str
    def generate(self, system: str, user: str, max_new_tokens: int = 700) -> str: ...


_TRANSIENT = ("429", "500", "502", "503", "504", "529", "resource_exhausted", "unavailable", "overloaded",
              "rate limit", "rate_limit", "timeout", "timed out", "connection")


# Google's quota errors name the quota, e.g. "GenerateRequestsPerDayPerProjectPerModel-FreeTier".
_DAILY_QUOTA = ("perday", "per day", "per_day")


def _is_transient(e: Exception) -> bool:
    msg = f"{type(e).__name__} {e}".lower()
    if any(t in msg for t in _DAILY_QUOTA):
        return False                                # resets at midnight Pacific; retrying now only wastes time
    return any(t in msg for t in _TRANSIENT)


def _suggested_delay(e: Exception) -> float:
    """Seconds the API asked us to wait ("Please retry in 37.5s" / "retryDelay': '37s'"), or 0."""
    m = re.search(r"retry(?:delay)?['\"]?\s*(?:in|:)\s*['\"]?(\d+(?:\.\d+)?)\s*s", str(e), re.I)
    return float(m.group(1)) if m else 0.0


def _with_retries(fn: Callable[[], str], what: str, tries: int = 5, base_delay: float = 5.0) -> str:
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:                      # SDKs raise many error types; retry only transient ones
            if attempt == tries - 1 or not _is_transient(e):
                raise
            delay = max(base_delay * 2 ** attempt, _suggested_delay(e) + 1)
            print(f"[{what}] {type(e).__name__}: {e} -- retrying in {delay:.0f}s", file=sys.stderr)
            time.sleep(delay)
    raise AssertionError("unreachable")


class HFLocalLLM:
    def __init__(self, model_name: str):
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
        self.name = model_name
        self._torch = torch
        self.tok = AutoTokenizer.from_pretrained(model_name)
        # Half precision on a GPU (fits Qwen2.5-Coder-3B on a 16 GB T4); full precision on CPU-only machines.
        dtype = torch.float16 if torch.cuda.is_available() else torch.float32
        if not torch.cuda.is_available():
            print(f"[hf] no CUDA GPU found: running {model_name} on the CPU (slow)", file=sys.stderr)
        try:
            self.lm = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype, device_map="auto")
        except TypeError:  # older transformers
            self.lm = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto")

    def generate(self, system, user, max_new_tokens=700):
        text = self.tok.apply_chat_template(
            [{"role": "system", "content": system}, {"role": "user", "content": user}],
            tokenize=False, add_generation_prompt=True)
        inputs = self.tok(text, return_tensors="pt").to(self.lm.device)
        with self._torch.inference_mode():
            out = self.lm.generate(**inputs, max_new_tokens=max_new_tokens, do_sample=False)
        return self.tok.decode(out[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)


# Gemini 2.x and older: thinking can be switched off and temperature 0 gives (near-)greedy replies.
_OLD_GEMINI = re.compile(r"^(models/)?gemini-[12]\.")
# Gemini 3+ counts thinking tokens inside max_output_tokens, so this much extra room is added on top of the
# visible reply length; otherwise a long think can use up the limit and leave an empty reply.
THINKING_HEADROOM = 8192


def gemini_request_settings(model: str, max_new_tokens: int, thinking_budget: int | None = None,
                            thinking_level: str | None = None) -> dict:
    """The generation settings for one Gemini request (a plain dict, so it can be unit-tested offline).

    Gemini 2.x: `thinking_budget` (0 = off; default 0) and temperature 0.
    Gemini 3+:  thinking cannot be switched off, only set to a level (`thinking_level`); temperature and other
                sampling settings are deprecated for these models, so none are sent. Replies can therefore vary
                a little between runs; the evaluation saves every judge reply so a report can be rebuilt.
    The two thinking settings are never sent together: the API rejects a request that has both.
    """
    if _OLD_GEMINI.match(model):
        out = {"temperature": 0.0, "max_output_tokens": max_new_tokens}
        if "2.5" in model:                          # 2.0 and older have no thinking at all
            out["thinking_config"] = {"thinking_budget": 0 if thinking_budget is None else thinking_budget}
        return out
    out = {"max_output_tokens": max_new_tokens + THINKING_HEADROOM}
    if thinking_level:
        out["thinking_config"] = {"thinking_level": thinking_level.upper()}
    return out


def _model_unavailable(e: Exception) -> bool:
    msg = str(e).lower()
    return ("404" in msg and "not_found" in msg) or "no longer available" in msg


class GeminiLLM:
    def __init__(self, model_name: str, thinking_budget: int | None = None, thinking_level: str | None = None):
        from google import genai
        self.name = model_name
        self.thinking_budget = thinking_budget
        self.thinking_level = thinking_level
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, system, user, max_new_tokens=700):
        from google.genai import types
        settings = gemini_request_settings(self.name, max_new_tokens, self.thinking_budget, self.thinking_level)
        if "thinking_config" in settings:
            settings["thinking_config"] = types.ThinkingConfig(**settings["thinking_config"])
        conf = types.GenerateContentConfig(system_instruction=system, **settings)

        def call():
            r = self.client.models.generate_content(model=self.name, contents=user, config=conf)
            text = r.text or ""
            if not text:
                reason = getattr((r.candidates or [None])[0], "finish_reason", "?")
                print(f"[gemini] empty response (finish_reason={reason})", file=sys.stderr)
            return text
        try:
            return _with_retries(call, "gemini")
        except Exception as e:
            if _model_unavailable(e):
                raise RuntimeError(
                    f"Gemini model {self.name!r} is not available to this API key. Google retires older models "
                    "(first for new keys). See which models your key can use with "
                    "`python scripts/list_gemini_models.py flash`, then set judge.model in configs/default.yaml "
                    "or set the JUDGE_MODEL environment variable for one run.") from e
            raise


class AnthropicLLM:
    def __init__(self, model_name: str):
        import anthropic
        self.name = model_name
        self.client = anthropic.Anthropic()          # reads ANTHROPIC_API_KEY

    def generate(self, system, user, max_new_tokens=700):
        def call():
            r = self.client.messages.create(model=self.name, max_tokens=max_new_tokens, temperature=0.0,
                                            system=system, messages=[{"role": "user", "content": user}])
            return "".join(b.text for b in r.content if getattr(b, "type", "") == "text")
        return _with_retries(call, "anthropic")


class RateLimited:
    """Spaces out calls so a free-tier requests-per-minute limit is not hit (fewer 429 errors and retries)."""

    def __init__(self, llm: LLM, requests_per_minute: float):
        self.llm, self.name = llm, llm.name
        self.interval = 60.0 / requests_per_minute
        self._last = None

    def generate(self, system, user, max_new_tokens=700):
        if self._last is not None:
            wait = self.interval - (time.monotonic() - self._last)
            if wait > 0:
                time.sleep(wait)
        self._last = time.monotonic()
        return self.llm.generate(system, user, max_new_tokens)


class EchoLLM:
    """Test double: returns a canned string built from the prompt. No model, no network."""
    name = "echo"

    def generate(self, system, user, max_new_tokens=700):
        return f"[echo] {user[:200]}"


def _make(backend: str, model: str, thinking_budget: int | None = None, thinking_level: str | None = None) -> LLM:
    if backend == "hf":
        return HFLocalLLM(model)
    if backend == "gemini":
        return GeminiLLM(model, thinking_budget, thinking_level)
    if backend == "anthropic":
        return AnthropicLLM(model)
    if backend == "echo":
        return EchoLLM()
    raise ValueError(f"Unknown LLM backend {backend!r}")


def make_llm(cfg: LLMConfig) -> LLM:
    return _make(cfg.backend, cfg.model, cfg.thinking_budget)


def judge_model_name(cfg: JudgeConfig) -> str:
    """judge.model from the config, unless the JUDGE_MODEL environment variable overrides it for this run."""
    return os.environ.get("JUDGE_MODEL", "").strip() or cfg.model


def make_judge(cfg: JudgeConfig) -> LLM:
    llm = _make(cfg.backend, judge_model_name(cfg), cfg.thinking_budget, cfg.thinking_level)
    if cfg.requests_per_minute and cfg.backend in ("gemini", "anthropic"):
        llm = RateLimited(llm, cfg.requests_per_minute)
    return llm
