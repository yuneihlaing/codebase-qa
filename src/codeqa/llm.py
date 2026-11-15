"""LLM backends behind one tiny interface: `generate(system, user, max_new_tokens) -> str`.

The generator (answers questions) and the judge (grades answers) both use it, so you can mix backends:
e.g. local Qwen as generator, a hosted model as judge. Greedy / temperature-0 decoding everywhere so a
re-run gives (nearly) the same answers.

Hosted backends retry transient errors (rate limits on free tiers are common) with exponential backoff.
"""
from __future__ import annotations

import os
import sys
import time
from typing import Callable, Protocol

from .config import JudgeConfig, LLMConfig


class LLM(Protocol):
    name: str
    def generate(self, system: str, user: str, max_new_tokens: int = 700) -> str: ...


_TRANSIENT = ("429", "500", "502", "503", "504", "529", "resource_exhausted", "unavailable", "overloaded",
              "rate limit", "rate_limit", "timeout", "timed out", "connection")


def _is_transient(e: Exception) -> bool:
    msg = f"{type(e).__name__} {e}".lower()
    return any(t in msg for t in _TRANSIENT)


def _with_retries(fn: Callable[[], str], what: str, tries: int = 5, base_delay: float = 5.0) -> str:
    for attempt in range(tries):
        try:
            return fn()
        except Exception as e:                      # SDKs raise many error types; retry only transient ones
            if attempt == tries - 1 or not _is_transient(e):
                raise
            delay = base_delay * 2 ** attempt
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


class GeminiLLM:
    def __init__(self, model_name: str, thinking_budget: int | None = 0):
        from google import genai
        self.name = model_name
        self.thinking_budget = thinking_budget
        self.client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])

    def generate(self, system, user, max_new_tokens=700):
        from google.genai import types
        extra = {}
        if self.thinking_budget is not None:
            # Gemini 2.5 counts "thinking" tokens against max_output_tokens; with thinking on, a 600-token
            # budget can be used up before any visible text is produced (empty reply -> unparsable grade).
            extra["thinking_config"] = types.ThinkingConfig(thinking_budget=self.thinking_budget)
        conf = types.GenerateContentConfig(system_instruction=system, temperature=0.0,
                                           max_output_tokens=max_new_tokens, **extra)

        def call():
            r = self.client.models.generate_content(model=self.name, contents=user, config=conf)
            text = r.text or ""
            if not text:
                reason = getattr((r.candidates or [None])[0], "finish_reason", "?")
                print(f"[gemini] empty response (finish_reason={reason})", file=sys.stderr)
            return text
        return _with_retries(call, "gemini")


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


class EchoLLM:
    """Test double: returns a canned string built from the prompt. No model, no network."""
    name = "echo"

    def generate(self, system, user, max_new_tokens=700):
        return f"[echo] {user[:200]}"


def _make(backend: str, model: str, thinking_budget: int | None = 0) -> LLM:
    if backend == "hf":
        return HFLocalLLM(model)
    if backend == "gemini":
        return GeminiLLM(model, thinking_budget)
    if backend == "anthropic":
        return AnthropicLLM(model)
    if backend == "echo":
        return EchoLLM()
    raise ValueError(f"Unknown LLM backend {backend!r}")


def make_llm(cfg: LLMConfig) -> LLM:
    return _make(cfg.backend, cfg.model, cfg.thinking_budget)


def make_judge(cfg: JudgeConfig) -> LLM:
    return _make(cfg.backend, cfg.model, cfg.thinking_budget)
