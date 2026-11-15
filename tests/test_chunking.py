from codeqa.chunking import chunk_repo
from codeqa.config import ChunkConfig


def test_chunk_kinds_and_lines(toy_repo):
    chunks = chunk_repo(toy_repo)
    by = {(c.path, c.name): c for c in chunks}
    assert by[("src/toy/app.py", "App")].kind == "class"
    assert by[("src/toy/app.py", "make_app")].kind == "function"
    assert by[("src/toy/app.py", "<module>")].kind == "module_header"
    assert by[("README.md", "README.md")].kind == "text"
    app = by[("src/toy/app.py", "App")]
    assert app.text.splitlines()[0].startswith("class App")
    assert [c.id for c in chunks] == list(range(len(chunks)))


def test_large_class_split_into_methods(toy_repo):
    chunks = chunk_repo(toy_repo, ChunkConfig(max_class_lines=3))
    names = {c.name for c in chunks if c.path == "src/toy/app.py"}
    assert {"App.run", "App.handle", "App.to_json"} <= names
    m = next(c for c in chunks if c.name == "App.run")
    assert m.kind == "method" and m.parent == "App"


def test_refs_recorded(toy_repo):
    chunks = {c.name: c for c in chunk_repo(toy_repo, ChunkConfig(max_class_lines=3))}
    assert "render" in chunks["App.handle"].calls
    assert "helpers.dumps" in chunks["App.to_json"].attr_calls


def test_module_code_flag(tmp_path):
    (tmp_path / "m.py").write_text("import os\n\ndef f():\n    pass\n\nX = f()\n", encoding="utf-8")
    assert not any(c.kind == "module_code" for c in chunk_repo(tmp_path))
    assert any(c.kind == "module_code" for c in chunk_repo(tmp_path, ChunkConfig(include_module_code=True)))


def test_syntax_error_falls_back_to_text(tmp_path):
    (tmp_path / "bad.py").write_text("def broken(:\n    pass\n", encoding="utf-8")
    assert [c.kind for c in chunk_repo(tmp_path)] == ["text"]
