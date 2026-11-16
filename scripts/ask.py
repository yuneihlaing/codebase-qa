#!/usr/bin/env python
"""Ask the assistant questions about a repository (one question, or an interactive session).

    python scripts/ask.py --repo flask "Where is url_for defined?"          # one question
    python scripts/ask.py --repo flask                                      # interactive session
    python scripts/ask.py --repo flask --config configs/offline.yaml        # no GPU / no downloads

Inside a session:
    :strategy structure | baseline_budget | baseline | structure_refs | structure_calls
    :compare            toggle side-by-side answers from baseline_budget and structure
    :sources            toggle the list of retrieved chunks (path:lines [why it was retrieved])
    :help  :quit
"""
import argparse
import sys

from codeqa.assistant import Assistant
from codeqa.config import load_config
from codeqa.retrieval import STRATEGIES

HELP = __doc__.split("Inside a session:")[1]


def show(ans, show_sources: bool) -> None:
    print(f"\n--- {ans.strategy} ({ans.seconds:.1f}s) " + "-" * 40)
    print(ans.text.strip())
    if show_sources:
        print(f"\nRetrieved context ({len(ans.chunks)} chunks):")
        print(ans.format_sources())
    print()


def main():
    ap = argparse.ArgumentParser(description="Ask questions about a code repository.")
    ap.add_argument("question", nargs="*", help="ask one question and exit")
    ap.add_argument("--repo", default="flask")
    ap.add_argument("--strategy", default="structure", choices=STRATEGIES)
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--compare", action="store_true", help="answer with baseline_budget AND structure")
    ap.add_argument("--no-sources", action="store_true")
    a = ap.parse_args()

    cfg = load_config(a.config)
    print(f"Loading the assistant for '{a.repo}' (generator: {cfg.llm.backend}:{cfg.llm.model}) ...",
          file=sys.stderr)
    bot = Assistant(a.repo, cfg)
    strategy, compare, sources = a.strategy, a.compare, not a.no_sources

    def answer(q):
        for s in (["baseline_budget", "structure"] if compare else [strategy]):
            ans = bot.ask(q, s)
            show(ans, sources)

    if a.question:
        answer(" ".join(a.question))
        return

    print(f"Ask about the {a.repo} repository. Strategy: {strategy}. Type :help for commands.\n")
    while True:
        try:
            q = input("you> ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not q:
            continue
        if q in (":q", ":quit", ":exit"):
            break
        if q == ":help":
            print(HELP)
        elif q.startswith(":strategy"):
            name = q.split(maxsplit=1)[1].strip() if " " in q else ""
            if name in STRATEGIES:
                strategy = name
                print(f"strategy = {strategy}")
            else:
                print(f"choose one of {', '.join(STRATEGIES)}")
        elif q == ":compare":
            compare = not compare
            print(f"compare = {compare}")
        elif q == ":sources":
            sources = not sources
            print(f"show sources = {sources}")
        elif q.startswith(":"):
            print("unknown command; :help")
        else:
            answer(q)


if __name__ == "__main__":
    main()
