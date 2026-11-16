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
python -m pytest -q                                   # 73 tests, ~2 s, no GPU/network
python scripts/prepare_repo.py --repo flask           # clone @ pinned commit, chunk (1409), embed, graph
python scripts/compare_retrieval.py --repo flask --configs configs/default.yaml configs/minilm.yaml configs/coderank.yaml
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
| Qwen3-Embedding-0.6B + FAISS | semantic search over chunks (questions carry a task instruction) |
| Qwen2.5-Coder-3B-Instruct (local, T4) | generator, identical for every configuration |
| Gemini 3.5 Flash-Lite (API) | LLM judge for the benchmark (different model family from the generator); `JUDGE_MODEL` overrides it for one run |
| SWE-QA benchmark (`data/swe_qa/`, Apache-2.0) | 48 Flask questions with reference answers |

## System design
```
question ─► semantic search (Qwen3-Embedding + FAISS, top 60)
              ├─ baseline         top-6 chunks                                       (plain RAG)
              ├─ baseline_budget  semantic order, filled to 18,000 chars              (fair control)
              ├─ code_seeds       top-4 code chunks first, then semantic fill          (control: no graph)
              └─ structure        top-4 code chunks + their graph neighbours, 18,000 chars   (proposed)
                                    graph edges: call · ref (names resolved through imports) ·
                                    called_by · parent class · module header
          ─► prompt: excerpts with path + line range, "answer ONLY from excerpts, cite, say what is missing"
          ─► Qwen2.5-Coder-3B ─► answer with file:line citations
```

## Design decisions (and the alternatives)
| Area | Choice | Main alternatives | Why this one |
|---|---|---|---|
| Test repository | Flask @ `85c5d93` | larger SWE-QA repos (django, sympy) | smallest SWE-QA repo (~80 Python files, 1,409 chunks): indexes in minutes on a free GPU, and testers know it well enough to judge answers |
| Benchmark | SWE-QA (48 Flask questions + reference answers) | snippet-level code QA; issue-thread QA sets | repository-level, cross-file questions with validated answers, pinned to a commit, in Python |
| Chunking | AST units: function / class / method (classes over 80 lines split by method); docs in 40-line windows | fixed-size windows; whole files | one chunk = one unit of meaning and one citation, and the place graph links attach; 46 of Flask's 64 classes stay whole |
| Embedder | Qwen3-Embedding-0.6B, questions prefixed with a task instruction | all-MiniLM-L6-v2; CodeRankEmbed; jina code models | trained on text and code (39% of Flask's chunks are docs), instruction-aware, 32K context, Apache-2.0; MiniLM reads ~256 tokens, so ~44% of chunks would be cut. Checked with `compare_retrieval.py` |
| Vector index | FAISS exact inner product | approximate (HNSW/IVF); a vector database; BM25 | 1,409 vectors: exact search is instant, with no approximation error and no server |
| Code graph | Python `ast` + own import resolution, one node per chunk | tree-sitter; a type checker / language server; line-level graph (RepoGraph) | no extra dependencies; nodes match what is retrieved; ambiguous links are skipped rather than guessed |
| Structure retrieval | top code chunks as seeds + 4 reserved slots for graph neighbours | seeds of any kind; neighbours compete on score | without these, graph neighbours almost never reached the context (offline check) |
| Context budget | 18,000 characters, at most 3,000 per chunk, same for every strategy | top-k only | equal budgets make the comparison fair; 99% of chunks are under 3,000 characters |
| Generator | Qwen2.5-Coder-3B-Instruct, greedy, 700 tokens | 1.5B (optional run); 7B; hosted APIs | code-trained and instruction-tuned, fits a free T4 with room for the prompt; local + greedy = reproducible and free |
| Judge | Gemini 3.5 Flash-Lite, thinking level `low`; pointwise + pairwise in both orders; replies cached | Gemini 2.5 Flash (retired for new API keys); Gemini 3.8 Flash (free tier: 20 requests/day); a Qwen judge; humans only | a different family avoids self-preference; swapping order cancels position bias; Google positions 3.5 Flash-Lite as a replacement for 2.5 Flash, and its free tier (500 requests/day) covers the ~336 judge requests in one day; the cache lets a stopped run resume; humans grade the 36 hand-written items |
| Statistics | paired bootstrap, 95% interval | paired t-test | no normality assumption for bounded, tie-heavy per-question scores |
| Interface | terminal (`ask.py`) + Gradio page (`app.py`) | Streamlit; notebook only | a Gradio share link from Colab lets classmates test remotely |

## Configurations compared
| Configuration | What changes | Why it is in the comparison |
|---|---|---|
| `baseline` | top-6 semantic chunks | textbook RAG reference point |
| `baseline_budget` | same 18,000-char budget as `structure`, no graph | isolates "more context" from "better context" |
| `code_seeds` | code chunks ranked first, no graph | isolates "prefer code over docs" from "follow the graph" |
| `structure` | code seeds + graph neighbours | proposed method |
| `structure_refs` / `structure_calls` | only reference or only call edges | which relationships matter |
| optional: `configs/small_model.yaml` | Qwen2.5-Coder-1.5B instead of 3B | smaller vs. larger generator |
| embedders (retrieval only) | Qwen3-Embedding-0.6B vs. MiniLM vs. CodeRankEmbed | evidence for the embedder choice (`scripts/compare_retrieval.py`) |

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
* Chunks are embedded from their first 4,000 characters (1,024 tokens). That covers 99% of Flask's chunks; the few
  longer ones (e.g. the `App` class header) are embedded by their beginning.
* The automatic metrics are proxies (SWE-QA has no gold file lists); the keyword checks for the test queries and
  safety tests are heuristics, which is why every row is also graded by hand.
* 48 questions per repository gives wide confidence intervals; one repository is not a general conclusion.
* The 3B generator may ignore instructions that a larger model would follow (see the safety results).

## Future improvements
* Link documentation to code (e.g. Sphinx roles like `:meth:` and backticked names) so docs can seed the graph.
* An agentic mode where the LLM chooses which files/functions to open next (the proposal's stretch goal).
* Scale to `requests` and `pytest` (commits already pinned in `src/codeqa/repos.py`) and pool the results.
* A reranker or a larger embedder (Qwen3-Embedding-4B), and a larger generator.

## Demo recording
<!-- TODO: link -->
