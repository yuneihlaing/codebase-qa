#!/usr/bin/env python
"""Print the Gemini models your API key can call, so you can pick one for judge.model.

    python scripts/list_gemini_models.py          # every model that can generate text
    python scripts/list_gemini_models.py flash    # only names containing "flash"

Google retires older models (first for new API keys), so the model in configs/default.yaml can stop working.
Use a name from this list in judge.model, or for one run: JUDGE_MODEL=<name> python scripts/evaluate.py ...
Needs GEMINI_API_KEY.
"""
import os
import sys


def main():
    if not os.environ.get("GEMINI_API_KEY"):
        sys.exit("error: set GEMINI_API_KEY first (in Colab: the Secrets panel, then the key cell)")
    from google import genai
    word = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    client = genai.Client(api_key=os.environ["GEMINI_API_KEY"])
    for m in client.models.list():
        name = (m.name or "").removeprefix("models/")
        if "generateContent" in (m.supported_actions or []) and word in name.lower():
            print(f"{name:40s} {m.display_name or ''}")


if __name__ == "__main__":
    main()
