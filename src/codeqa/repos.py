"""Repository registry + cloning.

SWE-QA questions are written against specific repositories, so every repo is checked out at the commit the
benchmark authors used. The commits come from repo_commit.txt in https://github.com/peng-weihan/SWE-QA-Bench
(Flask's matches the Check-in 1 notebook). The last three repos were added in the benchmark's second version.
"""
from __future__ import annotations

import subprocess
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class RepoSpec:
    name: str
    url: str
    commit: str | None = None
    size: str = "?"          # rough guide for planning compute: S / M / L / XL


REPOS: dict[str, RepoSpec] = {r.name: r for r in [
    RepoSpec("flask", "https://github.com/pallets/flask.git", "85c5d93", "S"),
    RepoSpec("requests", "https://github.com/psf/requests.git", "46e939b", "S"),
    RepoSpec("pytest", "https://github.com/pytest-dev/pytest.git", "5989efe", "M"),
    RepoSpec("pylint", "https://github.com/pylint-dev/pylint.git", "44740e5", "M"),
    RepoSpec("sphinx", "https://github.com/sphinx-doc/sphinx.git", "6c9e320", "L"),
    RepoSpec("sqlfluff", "https://github.com/sqlfluff/sqlfluff.git", "db9801b", "M"),
    RepoSpec("xarray", "https://github.com/pydata/xarray.git", "40119bf", "L"),
    RepoSpec("django", "https://github.com/django/django.git", "14fc2e9", "XL"),
    RepoSpec("matplotlib", "https://github.com/matplotlib/matplotlib.git", "a5e1f60", "XL"),
    RepoSpec("scikit-learn", "https://github.com/scikit-learn/scikit-learn.git", "adb1ae7", "XL"),
    RepoSpec("sympy", "https://github.com/sympy/sympy.git", "3c817ed", "XL"),
    RepoSpec("astropy", "https://github.com/astropy/astropy.git", "0a041d3", "XL"),
    RepoSpec("conan", "https://github.com/conan-io/conan.git", "52f43d9", "M"),
    RepoSpec("reflex", "https://github.com/reflex-dev/reflex.git", "fe0f946", "M"),
    RepoSpec("streamlink", "https://github.com/streamlink/streamlink.git", "ab1f365", "M"),
]}


def get_spec(name: str, url: str | None = None, commit: str | None = None) -> RepoSpec:
    if url:  # custom repo supplied on the command line
        return RepoSpec(name, url, commit)
    if name not in REPOS:
        raise KeyError(f"Unknown repo {name!r}. Known: {sorted(REPOS)} (or pass --repo-url).")
    spec = REPOS[name]
    return RepoSpec(spec.name, spec.url, commit or spec.commit, spec.size)


def _git(args: list[str], cwd: Path | None = None) -> str:
    out = subprocess.run(["git", *args], cwd=cwd, check=True, capture_output=True, text=True, encoding="utf-8")
    return out.stdout.strip()


def current_commit(repo_dir: str | Path) -> str | None:
    try:
        return _git(["rev-parse", "HEAD"], cwd=Path(repo_dir))
    except (subprocess.CalledProcessError, FileNotFoundError, NotADirectoryError):
        return None


def ensure_repo(spec: RepoSpec, repos_dir: str | Path) -> tuple[Path, str]:
    """Clone if needed, check out the pinned commit, return (path, resolved HEAD sha)."""
    repo_dir = Path(repos_dir) / spec.name
    if not repo_dir.exists():
        repo_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(["git", "clone", spec.url, str(repo_dir)], check=True)
    if spec.commit:
        _git(["checkout", "-q", spec.commit], cwd=repo_dir)
    return repo_dir, _git(["rev-parse", "HEAD"], cwd=repo_dir)
