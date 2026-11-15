from codeqa.chunking import chunk_repo
from codeqa.config import ChunkConfig
from codeqa.structure import RepoGraph


def build(toy_repo):
    chunks = chunk_repo(toy_repo, ChunkConfig(max_class_lines=3))
    return chunks, RepoGraph(toy_repo, chunks)


def idx(chunks, name):
    return next(c.id for c in chunks if c.name == name)


def test_module_names_use_package_root(toy_repo):
    _, g = build(toy_repo)
    assert g.module_to_file["toy.app"] == "src/toy/app.py"
    assert g.module_to_file["toy"] == "src/toy/__init__.py"


def test_file_level_import_edges(toy_repo):
    _, g = build(toy_repo)
    assert "src/toy/helpers.py" in g.file_imports["src/toy/app.py"]
    assert "src/toy/app.py" in g.file_imports["src/toy/__init__.py"]


def test_call_edge_through_relative_import(toy_repo):
    chunks, g = build(toy_repo)
    assert g.out_edges[idx(chunks, "App.handle")][idx(chunks, "render")] == "call"


def test_attr_call_resolves_module_function(toy_repo):
    chunks, g = build(toy_repo)
    assert idx(chunks, "dumps") in g.out_edges[idx(chunks, "App.to_json")]


def test_self_method_call_resolves_by_name(toy_repo):
    chunks, g = build(toy_repo)
    assert idx(chunks, "App.handle") in g.out_edges[idx(chunks, "App.run")]


def test_reexport_resolution(toy_repo):
    chunks, g = build(toy_repo)
    test_fn = idx(chunks, "test_app")
    assert idx(chunks, "App") in g.out_edges[test_fn]       # tests/ -> toy/__init__ -> toy/app


def test_reverse_parent_and_header_edges(toy_repo):
    chunks, g = build(toy_repo)
    types = dict((j, t) for j, t in g.neighbors(idx(chunks, "App.handle")))
    assert types[idx(chunks, "App")] == "parent"
    assert any(t == "header" for t in types.values())
    assert any(t == "called_by" for _, t in g.neighbors(idx(chunks, "render")))
