"""Run directories: one folder per experimental setting, so results from different settings never mix.

    results/<tag>/flask__structure.jsonl
    results/<tag>/config_used.json      <- the settings that produced every file in this folder
    reports/<tag>/summary_flask.md

Re-running with the same tag resumes. Re-running with the same tag but *different* settings is refused,
because the resumed files would silently keep the old answers (e.g. an alpha sweep that changes nothing).
"""
from __future__ import annotations

import json
from pathlib import Path

from .config import Config, result_fingerprint, to_dict

DEFAULT_TAG = "default"


class ConfigMismatch(RuntimeError):
    pass


def _norm(x):
    return json.loads(json.dumps(x))       # tuples -> lists, so a saved config compares equal to a live one


def results_dir(cfg: Config, tag: str = DEFAULT_TAG) -> Path:
    return Path(cfg.results_dir) / tag


def reports_dir(cfg: Config, tag: str = DEFAULT_TAG) -> Path:
    return Path(cfg.reports_dir) / tag


def claim_run_dir(cfg: Config, tag: str = DEFAULT_TAG) -> Path:
    """Create results/<tag>/ (saving the config) or verify that the existing one used the same settings."""
    d = results_dir(cfg, tag)
    d.mkdir(parents=True, exist_ok=True)
    saved_path = d / "config_used.json"
    current = _norm(result_fingerprint(cfg))
    if saved_path.exists():
        saved = _norm(json.loads(saved_path.read_text(encoding="utf-8")))
        saved = {k: saved.get(k) for k in current}
        if saved != current:
            diffs = [f"{sec}.{key}: {saved.get(sec, {}).get(key)!r} -> {val!r}"
                     for sec, vals in current.items() for key, val in vals.items()
                     if (saved.get(sec) or {}).get(key) != val]
            raise ConfigMismatch(
                f"{d} was produced with different settings ({'; '.join(diffs)}). "
                f"Use a new --tag for this setting (e.g. --tag alpha0.8) so old answers are not reused.")
    else:
        saved_path.write_text(json.dumps(to_dict(cfg), indent=2), encoding="utf-8")
    return d
