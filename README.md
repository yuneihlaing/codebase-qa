# LLM-based Assistant for Codebase Q&A

Ask natural-language questions about a Python repository and get answers that cite `file:line` evidence.
Two kinds of retrieval are compared: plain **semantic retrieval** over AST chunks, and **structure-aware**
retrieval that also follows the repository's relationships (function calls, class membership, references
resolved through imports). The test repository is **Flask** at commit `85c5d93`, evaluated with the SWE-QA
benchmark, a hand-written test-query set and a set of safety tests.

## Setup
**Colab (recommended, T4 GPU):** open `notebooks/run_on_colab.ipynb`, choose *Runtime > Change runtime type >
T4 GPU*, and run the cells top to bottom. Put `GEMINI_API_KEY` in Colab *Secrets* if you want the LLM judge.

**Locally** (Python 3.10+; a GPU is needed for the generator):
```bash
pip install -e ".[retrieval,llm,judge,ui,dev]"
python -m pytest -q          # 76 tests, a few seconds, no GPU or network
```
Run every command from the project folder. To check that the pipeline runs without a GPU, model downloads or
API keys, add `--config configs/offline.yaml` to any command (word-counting embedder + echo generator; its
outputs are not real results).

## Run, step by step
```bash
# 1. Clone Flask at the pinned commit, cut it into chunks (1,409), embed them, build the code graph
python scripts/prepare_repo.py --repo flask

# 1b. Compare embedders on retrieval only (no generator, minutes)
python scripts/compare_retrieval.py --repo flask --configs configs/default.yaml configs/minilm.yaml configs/coderank.yaml

# 2. Answer the 48 SWE-QA questions with each strategy (try --limit 5 first; re-running resumes)
python scripts/run_experiment.py --repo flask --strategy baseline baseline_budget code_seeds structure

# 3. Score and compare (paired comparisons against baseline_budget)
python scripts/evaluate.py --repo flask --strategies baseline baseline_budget code_seeds structure
#    ... plus the LLM judge (needs GEMINI_API_KEY; resumes if stopped)
python scripts/evaluate.py --repo flask --strategies baseline baseline_budget code_seeds structure --judge --pairwise

# 4. Hand-written test queries (24) and safety tests (12)
python scripts/run_test_queries.py --repo flask --strategy baseline_budget structure
python scripts/run_probes.py --repo flask --strategy baseline_budget structure
#    after filling manual_success / notes in the CSVs:
python scripts/summarize_grades.py reports/default/test_queries_flask.csv reports/default/safety_flask.csv

# 5. Optional: edge-type ablations, and a smaller generator under its own tag
python scripts/run_experiment.py --repo flask --strategy structure_refs structure_calls
python scripts/run_experiment.py --repo flask --strategy baseline_budget structure --config configs/small_model.yaml --tag qwen1.5b

# 6. Ask the assistant yourself
python scripts/ask.py --repo flask "What happens after Flask.wsgi_app is called?"
python scripts/app.py --repo flask --share          # small web page
```
Outputs go to `results/<tag>/` (answers, plus the settings that produced them in `config_used.json`) and
`reports/<tag>/` (tables). Both folders are git-ignored. A tag folder refuses to mix runs with different
settings; use a new `--tag` for each setting.

## Strategies
All strategies use the same chunks, embedder, generator (Qwen2.5-Coder-3B-Instruct, greedy, 700 tokens) and
system prompt. `baseline_budget`, `code_seeds` and the `structure*` strategies also share the same context budget
(18,000 characters, at most 3,000 per chunk).

| Strategy | Context |
|---|---|
| `baseline` | top-6 semantic chunks (plain RAG reference point; smaller context) |
| `baseline_budget` | semantic chunks up to the budget (fair control) |
| `code_seeds` | top-4 code chunks first, then semantic fill; no graph |
| `structure` | top-4 code chunks plus their graph neighbours: `call`, `called_by`, `ref` (names resolved through imports), `parent` (method → class), `header` (module imports and top-level code) |
| `structure_refs` | as `structure`, with `ref` + `parent` + `header` links only |
| `structure_calls` | as `structure`, with `call` + `called_by` + `parent` + `header` links only |

## Configurations (`configs/`)
| File | Use |
|---|---|
| `default.yaml` | every setting: Qwen3-Embedding-0.6B, Qwen2.5-Coder-3B generator, Gemini 3.5 Flash-Lite judge |
| `offline.yaml` | no GPU / downloads / keys: checks the pipeline runs |
| `small_model.yaml` | Qwen2.5-Coder-1.5B generator |
| `minilm.yaml`, `coderank.yaml` | alternative embedders for `compare_retrieval.py` |

The judge model can be changed for one run with the `JUDGE_MODEL` environment variable;
`python scripts/list_gemini_models.py flash` lists the Gemini models your key can use.

## Project layout
```
configs/      settings (YAML)
data/         SWE-QA questions for 15 repos (data/swe_qa/, Apache-2.0), test_queries.jsonl, probes.jsonl
notebooks/    run_on_colab.ipynb: every step on Colab
scripts/      one command per step (above)
src/codeqa/   the library: chunking, embedding, index, code graph, retrieval, prompts, LLM backends,
              the assistant, evaluation (metrics, LLM judge, statistics, keyword checks)
tests/        unit tests on a small fake repository
```
