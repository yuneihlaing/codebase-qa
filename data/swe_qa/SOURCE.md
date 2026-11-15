# SWE-QA benchmark questions (copied)

* Source: https://github.com/peng-weihan/SWE-QA-Bench (folder `Benchmark/`), commit `c13deac7a0d99b0ca2e593e004c4739475785b08`
* Also published as https://huggingface.co/datasets/swe-qa/SWE-QA-Benchmark
* Paper: Peng et al., "SWE-QA: Can Language Models Answer Repository-level Code Questions?" (ACL Findings 2026, arXiv:2509.14635)
* License: Apache-2.0 (copy in `LICENSE` in this folder). Files are unmodified.
* `repo_commit.txt`: the commit of each repository the questions were written against (also pinned in `src/codeqa/repos.py`).

Each line is `{"question": ..., "answer": ...}`; 48 lines per repository. Refresh with `python scripts/download_swe_qa.py`.
