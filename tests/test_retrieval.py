import numpy as np

from codeqa.chunking import chunk_repo
from codeqa.config import ChunkConfig, EmbedConfig, RetrievalConfig
from codeqa.embedding import HashingEmbedder
from codeqa.index import IndexStore
from codeqa.retrieval import Retriever, fill_budget
from codeqa.structure import RepoGraph


def make(toy_repo, **rcfg):
    chunks = chunk_repo(toy_repo, ChunkConfig(max_class_lines=3))
    emb = HashingEmbedder()
    store = IndexStore.build(chunks, emb, EmbedConfig(model="hashing"), show_progress=False)
    cfg = RetrievalConfig(**rcfg)
    return Retriever(store, emb, cfg, RepoGraph(toy_repo, chunks))


def test_index_search_matches_bruteforce(toy_repo):
    r = make(toy_repo)
    q = r._qvec("render template with context")
    top = r.store.search(q, 3)
    brute = np.argsort(-(r.store.emb @ q))[:3]
    assert [i for i, _ in top] == [int(i) for i in brute]


def test_baseline_returns_k(toy_repo):
    assert len(make(toy_repo, k=3).baseline("render a template").chunks) == 3


def test_fill_budget_skips_too_big_but_keeps_first(toy_repo):
    chunks = chunk_repo(toy_repo)
    order = list(range(len(chunks)))
    picked = fill_budget(order, chunks, budget=1, per_chunk=3000)
    assert picked == [0]                              # always at least one chunk
    total = sum(min(len(chunks[i].text), 3000) for i in fill_budget(order, chunks, 120, 3000))
    assert total <= 120 or len(fill_budget(order, chunks, 120, 3000)) == 1


def test_structure_pulls_in_callee_that_semantic_misses(toy_repo):
    r = make(toy_repo, k_seed=1, k=1, pool=1)
    q = "what does App.run do"
    base = {c.name for c in r.baseline(q).chunks}
    res = r.structure_aware(q)
    names = {c.name for c in res.chunks}
    assert "App.handle" in names                      # callee reached through the call graph
    assert len(names) > len(base)
    assert res.origins[0] == "seed"


def test_edge_filter_ablation(toy_repo):
    r = make(toy_repo, k_seed=1, pool=1)
    only_refs = r.retrieve("structure_refs", "what does App.run do")
    assert all(o in ("seed", "ref", "parent", "header", "semantic") for o in only_refs.origins)
    assert "call" not in only_refs.origins


def test_code_seeds_skip_docs(toy_repo):
    r = make(toy_repo, k_seed=1, seed_from="code")
    res = r.structure_aware("A toy web framework")          # README text is the best semantic hit
    assert r.baseline("A toy web framework").chunks[0].kind == "text"
    assert res.origins[0] == "seed" and res.chunks[0].kind != "text"


def test_reserved_neighbours_follow_seeds(toy_repo):
    r = make(toy_repo, k_seed=1, seed_from="code", reserve_expansion=2, max_expansion=4)
    res = r.structure_aware("what does App.run do")
    assert res.origins[0] == "seed"
    assert all(o not in ("seed", "semantic") for o in res.origins[1:3])   # graph neighbours come next


def test_called_by_ranked_by_similarity_not_path(toy_repo):
    # `App` is used by make_app (src/, early in path order) and by test_app (tests/, late in path order).
    # With room for one caller, the kept one must be the caller most similar to the question.
    r = make(toy_repo, max_in_edges=1)
    app = next(c.id for c in r.store.chunks if c.name == "App")
    q = r._qvec("test assert run")
    exp = r._scored_neighbours(q, [(app, 1.0)], {app}, {"called_by"})
    assert [r.store.chunks[i].name for i in exp] == ["test_app"]
    by_path = [r.store.chunks[j].name for j, t in r.graph.neighbors(app, max_in=1) if t == "called_by"]
    assert by_path == ["make_app"]                     # what the old path-order truncation would have kept


def test_same_budget_as_control(toy_repo):
    r = make(toy_repo, context_char_budget=300, per_chunk_chars=200)
    q = "what does App.run do"
    for res in (r.baseline_budget(q), r.structure_aware(q)):
        used = sum(min(len(c.text), 200) for c in res.chunks)
        assert used <= 300 or len(res.chunks) == 1


def test_code_seeds_control_uses_no_graph(toy_repo):
    r = make(toy_repo, k_seed=1, seed_from="code")
    res = r.retrieve("code_seeds", "what does App.run do")
    assert set(res.origins) <= {"seed", "semantic"} and res.origins[0] == "seed"
