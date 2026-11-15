#!/usr/bin/env python
"""Refresh data/swe_qa/ from the SWE-QA-Bench GitHub repository (the copy shipped here is already complete).

    python scripts/download_swe_qa.py            # needs git and access to github.com
"""
import shutil
import subprocess
import tempfile
from pathlib import Path

URL = "https://github.com/peng-weihan/SWE-QA-Bench.git"
DEST = Path(__file__).resolve().parents[1] / "data" / "swe_qa"


def main():
    with tempfile.TemporaryDirectory() as tmp:
        subprocess.run(["git", "clone", "-q", "--depth", "1", URL, tmp], check=True)
        sha = subprocess.run(["git", "-C", tmp, "rev-parse", "HEAD"], capture_output=True, text=True, encoding="utf-8",
                             check=True).stdout.strip()
        DEST.mkdir(parents=True, exist_ok=True)
        files = sorted(Path(tmp, "Benchmark").glob("*.jsonl"))
        for f in files:
            shutil.copy(f, DEST / f.name)
        shutil.copy(Path(tmp, "repo_commit.txt"), DEST / "repo_commit.txt")
        shutil.copy(Path(tmp, "LICENSE"), DEST / "LICENSE")
    print(f"copied {len(files)} files from {URL} @ {sha[:10]} into {DEST}")


if __name__ == "__main__":
    main()
