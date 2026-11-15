#!/usr/bin/env python
"""Clone a repo, chunk it, embed it, build the structure graph, and print diagnostics.

    python scripts/prepare_repo.py --repo flask
    python scripts/prepare_repo.py --repo flask --config configs/offline.yaml     # no downloads (hashing embedder)
    python scripts/prepare_repo.py --repo requests pytest
    python scripts/prepare_repo.py --repo myproj --repo-url https://github.com/me/myproj.git

Outputs: repos/<repo>/ (git checkout at the pinned commit) and artifacts/<repo>/<embedder>/
(chunks.jsonl, emb.npy, meta.json). The graph is rebuilt from the checkout whenever it is needed.
"""
import argparse
import json
from collections import Counter

from codeqa.config import load_config
from codeqa.pipeline import prepare
from codeqa.repos import get_spec


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", nargs="+", required=True)
    ap.add_argument("--repo-url")
    ap.add_argument("--commit")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--embed-model", help="override embed.model (use 'hashing' for a no-download smoke test)")
    a = ap.parse_args()
    cfg = load_config(a.config, {"embed": {"model": a.embed_model}} if a.embed_model else None)

    for name in a.repo:
        spec = get_spec(name, a.repo_url, a.commit)
        store, graph = prepare(spec, cfg)
        kinds = Counter(c.kind for c in store.chunks)
        print(f"\n=== {name} @ {store.meta['commit'][:10]} ===")
        print(f"chunks: {len(store.chunks)}  by kind: {dict(kinds)}")
        print("graph:", json.dumps(graph.stats(), indent=2))


if __name__ == "__main__":
    main()
