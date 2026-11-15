# LLM-based Assistant for Codebase Q&A

Ask natural-language questions about a Python repository and get answers that cite `file:line` evidence.
The project compares a **semantic-retrieval** assistant with a **structure-aware** one that also follows the
repository's import / call / class relationships, on the **SWE-QA** benchmark, a hand-written Flask test set
and a set of safety tests.

## Use case and target users
Developers who must understand an unfamiliar, poorly documented codebase: someone onboarding to an existing
project, or a student studying an open-source library. Typical questions: *where is X implemented, what
happens after Y is called, what breaks if I change Z.* The prototype is evaluated on **Flask** (commit
`85c5d93`), a small but real web framework (~80 Python files).

## How to run
**Colab (T4 GPU):** open `notebooks/run_on_colab.ipynb` and run the cells top to bottom.
**Locally:**
```bash
pip install -e ".[retrieval,llm,judge,ui,dev]"
python -m pytest -q                                   # 61 tests, ~1 s, no GPU/network
python scripts/prepare_repo.py --repo flask           # clone @ pinned commit, chunk (1409), embed, graph
python scripts/ask.py --repo flask "What happens after Flask.wsgi_app is called?"   # use the assistant
python scripts/run_experiment.py --repo flask --strategy baseline baseline_budget code_seeds structure
python scripts/evaluate.py --repo flask --strategies baseline baseline_budget code_seeds structure
python scripts/run_test_queries.py --repo flask       # 24-query evaluation table
python scripts/run_probes.py --repo flask             # 12 safety / robustness tests
python scripts/app.py --repo flask --share            # web UI for the demo and peer testers
```
No GPU or downloads? Add `--config configs/offline.yaml` to any command (hashing embedder + echo generator)
to check that the pipeline runs; those outputs are not results.

## Data sources and tools
| Source / tool | Role |
|---|---|
| Flask GitHub repository (code, docs, config files) | the knowledge base the assistant searches |
| Python `ast` | splits code into functions / classes / methods and extracts imports, calls and references |
| `all-MiniLM-L6-v2` + FAISS | semantic search over chunks |
| Qwen2.5-Coder-3B-Instruct (local, T4) | generator, identical for every configuration |
| Gemini 2.5 Flash (API) | LLM judge for the benchmark (different model family from the generator) |
| SWE-QA benchmark (`data/swe_qa/`, Apache-2.0) | 48 Flask questions with reference answers |

## System design
```
question ─► semantic search (MiniLM + FAISS, top 60)
              ├─ baseline         top-6 chunks                                       (Check-in 1)
              ├─ baseline_budget  semantic order, filled to 18,000 chars              (fair control)
              ├─ code_seeds       top-4 code chunks first, then semantic fill          (control: no graph)
              └─ structure        top-4 code chunks + their graph neighbours, 18,000 chars   (proposed)
                                    graph edges: call · ref (names resolved through imports) ·
                                    called_by · parent class · module header
          ─► prompt: excerpts with path + line range, "answer ONLY from excerpts, cite, say what is missing"
          ─► Qwen2.5-Coder-3B ─► answer with file:line citations
```

## Configurations compared
| Configuration | What changes | Why it is in the comparison |
|---|---|---|
| `baseline` | top-6 semantic chunks | Check-in 1 starting point |
| `baseline_budget` | same 18,000-char budget as `structure`, no graph | isolates "more context" from "better context" |
| `code_seeds` | code chunks ranked first, no graph | isolates "prefer code over docs" from "follow the graph" |
| `structure` | code seeds + graph neighbours | proposed method |
| `structure_refs` / `structure_calls` | only reference or only call edges | which relationships matter |
| optional: `configs/small_model.yaml` | Qwen2.5-Coder-1.5B instead of 3B | smaller vs. larger generator |

Trade-offs to discuss: <!-- TODO after running: answer quality vs. latency (`seconds`), how often the graph
helps vs. adds noise (`graph_share`), docs vs. code in the context, cost of the judge. -->

## Evaluation
1. **SWE-QA benchmark (48 Flask questions)** — `scripts/evaluate.py`: retrieval recall of the files and
   identifiers named in the reference answer, citation validity, abstention rate, LLM-judge scores (1–5 on
   five dimensions) and pairwise preferences in both orders; paired bootstrap 95% CIs against `baseline_budget`.
2. **Test query set (24 queries)** — `data/test_queries.jsonl`: common (5), data/tool-use (5), ambiguous (4),
   edge cases (6), unreliable-to-answer (4), each with an expected answer verified against the Flask source.
   `scripts/run_test_queries.py` writes the table *query | expected | response | success | notes*;
   success is first checked automatically, then graded by hand.

Results: <!-- TODO paste reports/default/summary_flask.md (key rows) and the summary tables of
reports/default/test_queries_flask.md after hand grading (python scripts/summarize_grades.py ...). -->

## Safety, security and robustness tests
12 tests in `data/probes.jsonl` (`scripts/run_probes.py`): direct prompt injection (×2: leak the system prompt,
drop the grounding rule), indirect injection through a planted repository file (×2, canary tokens),
hallucination via false premise / non-existent API (×2), out-of-scope requests (×2), an unsafe request,
credential leakage from a retrieved `.env`, conflicting documentation, and a non-existent setting. The
ambiguous and unanswerable test queries (q11–q14, q21–q24) double as robustness tests.

Findings: <!-- TODO summarize reports/default/safety_flask.md: what held, what failed, what it means. -->

## Peer / user testing
See [docs/peer_testing.md](docs/peer_testing.md) (testers, query types, what worked, what failed, changes).
Logs: `results/user_testing/`.

## Limitations
* Name resolution is static and approximate: no type inference, so `obj.method()` is linked only when the
  method name is unambiguous; dynamic dispatch, decorators that register things and `getattr` are invisible.
* Only Python is parsed structurally; documentation has no graph edges (it is reached by semantic search only).
* MiniLM truncates each chunk to ~256 tokens, so long functions are embedded by their beginning.
* The automatic metrics are proxies (SWE-QA has no gold file lists); the keyword checks for the test queries and
  safety tests are heuristics, which is why every row is also graded by hand.
* 48 questions per repository gives wide confidence intervals; one repository is not a general conclusion.
* The 3B generator may ignore instructions that a larger model would follow (see the safety results).

## Future improvements
* Link documentation to code (e.g. Sphinx roles like `:meth:` and backticked names) so docs can seed the graph.
* An agentic mode where the LLM chooses which files/functions to open next (the proposal's stretch goal).
* Scale to `requests` and `pytest` (commits already pinned in `src/codeqa/repos.py`) and pool the results.
* A code-specific embedder (`embed.model`) and a larger generator.

## Demo recording
<!-- TODO: link -->
