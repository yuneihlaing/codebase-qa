"""Tiny statistics helpers. With only ~48 questions per repo, always report uncertainty."""
from __future__ import annotations

import numpy as np


def paired_bootstrap(a: list[float], b: list[float], n: int = 10_000, seed: int = 0) -> dict:
    """Mean of (b - a) with a 95% bootstrap CI. a, b are per-question scores for the SAME questions."""
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    rng = np.random.default_rng(seed)
    means = rng.choice(d, size=(n, len(d)), replace=True).mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return {"mean_diff": float(d.mean()), "ci_low": float(lo), "ci_high": float(hi),
            "significant": bool(lo > 0 or hi < 0), "n": int(len(d))}


def win_tie_loss(a: list[float], b: list[float], tol: float = 1e-9) -> dict:
    d = np.asarray(b, dtype=float) - np.asarray(a, dtype=float)
    return {"b_wins": int((d > tol).sum()), "ties": int((np.abs(d) <= tol).sum()), "a_wins": int((d < -tol).sum())}
