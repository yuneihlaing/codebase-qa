#!/usr/bin/env python
"""Small web UI for asking the assistant questions (optional; needs `pip install gradio`).

    python scripts/app.py --repo flask                  # http://127.0.0.1:7860
    python scripts/app.py --repo flask --share          # public link, e.g. when running on Colab
"""
import argparse

from codeqa.assistant import Assistant
from codeqa.config import load_config

MODES = ["structure", "baseline_budget", "compare both"]


def sources_md(ans) -> str:
    return "\n".join(f"- `{c.path}:{c.start}-{c.end}` ({c.kind} `{c.name}`) — *{o}*"
                     for c, o in zip(ans.chunks, ans.origins))


def build(bot: Assistant):
    import gradio as gr

    def ask(question, mode):
        if not question.strip():
            return "Type a question first.", ""
        strategies = ["baseline_budget", "structure"] if mode == "compare both" else [mode]
        answers = [bot.ask(question, s) for s in strategies]
        md = "\n\n".join(f"### {a.strategy} ({a.seconds:.1f}s)\n{a.text}" for a in answers)
        src = "\n\n".join(f"**{a.strategy}**\n{sources_md(a)}" for a in answers)
        return md, src

    with gr.Blocks(title=f"Codebase Q&A: {bot.repo}") as demo:
        gr.Markdown(f"# Codebase Q&A assistant — `{bot.repo}`\nAnswers cite `file:line` evidence from the "
                    "repository. *Structure* adds code the retrieved functions call / are called by.")
        mode = gr.Radio(MODES, value="structure", label="Retrieval")
        q = gr.Textbox(label="Question", placeholder="e.g. What happens after Flask.wsgi_app is called?")
        go = gr.Button("Ask", variant="primary")
        out = gr.Markdown()
        with gr.Accordion("Retrieved context", open=False):
            src = gr.Markdown()
        go.click(ask, [q, mode], [out, src])
        q.submit(ask, [q, mode], [out, src])
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="flask")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--share", action="store_true", help="create a public gradio link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    build(Assistant(a.repo, load_config(a.config))).launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
