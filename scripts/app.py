#!/usr/bin/env python
"""Small web UI for demos and peer testing (optional; needs `pip install gradio`).

    python scripts/app.py --repo flask                  # http://127.0.0.1:7860
    python scripts/app.py --repo flask --share          # public link (e.g. from Colab) for classmates

Testers type their name, ask questions, optionally compare strategies, and rate each answer. Every
interaction is appended to results/user_testing/<name>.jsonl, same format as scripts/ask.py.
"""
import argparse
from pathlib import Path

from codeqa.assistant import Assistant, SessionLog
from codeqa.config import load_config

MODES = ["structure", "baseline_budget", "compare both"]


def sources_md(ans) -> str:
    return "\n".join(f"- `{c.path}:{c.start}-{c.end}` ({c.kind} `{c.name}`) — *{o}*"
                     for c, o in zip(ans.chunks, ans.origins))


def build(bot: Assistant, results_dir: str):
    import gradio as gr

    def log_for(name):
        name = (name or "anonymous").strip().replace("/", "_") or "anonymous"
        return SessionLog(Path(results_dir) / "user_testing" / f"{name}.jsonl", name)

    def ask(name, question, mode, state):
        if not question.strip():
            return "Type a question first.", "", state
        strategies = ["baseline_budget", "structure"] if mode == "compare both" else [mode]
        answers = [bot.ask(question, s) for s in strategies]
        md = "\n\n".join(f"### {a.strategy} ({a.seconds:.1f}s)\n{a.text}" for a in answers)
        src = "\n\n".join(f"**{a.strategy}**\n{sources_md(a)}" for a in answers)
        for a in answers:
            log_for(name).write(a, rating=None, comment="", event="asked")
        return md, src, answers

    def feedback(name, rating, comment, state):
        if not state:
            return "Ask a question first."
        for a in state:
            log_for(name).write(a, rating=str(rating), comment=comment, event="rated")
        return "Thanks, feedback saved."

    with gr.Blocks(title=f"Codebase Q&A: {bot.repo}") as demo:
        gr.Markdown(f"# Codebase Q&A assistant — `{bot.repo}`\nAnswers cite `file:line` evidence from the "
                    "repository. *Structure* adds code the retrieved functions call / are called by.")
        state = gr.State([])
        with gr.Row():
            name = gr.Textbox(label="Your name (for the testing log)", scale=1)
            mode = gr.Radio(MODES, value="structure", label="Retrieval", scale=2)
        q = gr.Textbox(label="Question", placeholder="e.g. What happens after Flask.wsgi_app is called?")
        go = gr.Button("Ask", variant="primary")
        out = gr.Markdown()
        with gr.Accordion("Retrieved context", open=False):
            src = gr.Markdown()
        with gr.Row():
            rating = gr.Slider(1, 5, value=3, step=1, label="How helpful was this? (1-5)")
            comment = gr.Textbox(label="What was wrong or confusing?")
        send = gr.Button("Send feedback")
        status = gr.Markdown()
        go.click(ask, [name, q, mode, state], [out, src, state])
        q.submit(ask, [name, q, mode, state], [out, src, state])
        send.click(feedback, [name, rating, comment, state], status)
    return demo


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--repo", default="flask")
    ap.add_argument("--config", default="configs/default.yaml")
    ap.add_argument("--share", action="store_true", help="create a public gradio link")
    ap.add_argument("--port", type=int, default=7860)
    a = ap.parse_args()
    cfg = load_config(a.config)
    build(Assistant(a.repo, cfg), cfg.results_dir).launch(share=a.share, server_port=a.port)


if __name__ == "__main__":
    main()
