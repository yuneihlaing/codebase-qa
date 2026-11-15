"""AST-based chunking (Step 1 of the pipeline).

Why not split every N characters? Code has natural units (functions, classes, methods). A chunk that is
exactly one function embeds as one coherent idea and can be cited as `file:start-end`. Plain text/docs
fall back to overlapping line windows.

Besides the text, each Python chunk records *what it references* (names, calls, attribute calls). Those
are collected here, while we still hold the AST node, and are consumed later by structure.py to build
the repository graph.
"""
from __future__ import annotations

import ast
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from .config import ChunkConfig


@dataclass
class Chunk:
    path: str                     # path relative to repo root
    start: int                    # 1-indexed start line
    end: int                      # inclusive end line
    name: str                     # function/class name, "Class.method", "<module>" ...
    kind: str                     # function | class | method | module_header | module_code | text
    text: str
    names: list = field(default_factory=list)       # identifiers read in this chunk
    calls: list = field(default_factory=list)       # f(...) -> "f"
    attr_calls: list = field(default_factory=list)  # a.b.c(...) -> "a.c"  (root name . final attribute)
    parent: str | None = None     # for methods: the owning class name
    id: int = -1                  # position in the chunk list (assigned by chunk_repo)

    def label(self) -> str:
        return f"{self.path}:{self.start}-{self.end}"


# ---------------------------------------------------------------- reference collection
def _root_name(node: ast.AST) -> str:
    while isinstance(node, ast.Attribute):
        node = node.value
    return node.id if isinstance(node, ast.Name) else ""


def collect_refs(nodes: list[ast.AST]) -> tuple[list, list, list]:
    names, calls, attr_calls = set(), set(), set()
    for root in nodes:
        for n in ast.walk(root):
            if isinstance(n, ast.Name) and isinstance(n.ctx, ast.Load):
                names.add(n.id)
            elif isinstance(n, ast.Call):
                f = n.func
                if isinstance(f, ast.Name):
                    calls.add(f.id)
                elif isinstance(f, ast.Attribute):
                    attr_calls.add(f"{_root_name(f.value)}.{f.attr}")
    return sorted(names), sorted(calls), sorted(attr_calls)


# ---------------------------------------------------------------- python
_FUNCS = (ast.FunctionDef, ast.AsyncFunctionDef)


def _seg(lines: list[str], start: int, end: int) -> str:
    return "\n".join(lines[start - 1:end])


def _node_start(node: ast.AST) -> int:
    decos = [d.lineno for d in getattr(node, "decorator_list", [])]
    return min([node.lineno] + decos)


def _mk(rel, s, e, name, kind, lines, nodes, parent=None) -> Chunk:
    names, calls, attr_calls = collect_refs(nodes)
    return Chunk(rel, s, e, name, kind, _seg(lines, s, e), names, calls, attr_calls, parent)


def chunk_python(path: Path, rel: str, cfg: ChunkConfig) -> list[Chunk]:
    src = path.read_text(encoding="utf-8", errors="ignore")
    lines = src.splitlines()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return chunk_text(path, rel, cfg)

    chunks: list[Chunk] = []
    defs: list[tuple[int, int]] = []
    loose: list[ast.stmt] = []            # top-level statements that are not def/class (for module_code)

    def flush_loose(after_first_def: bool):
        if loose and after_first_def and cfg.include_module_code:
            s, e = _node_start(loose[0]), loose[-1].end_lineno
            text = _seg(lines, s, e).strip()
            if text:
                chunks.append(_mk(rel, s, e, "<module-code>", "module_code", lines, list(loose)))
        loose.clear()

    seen_def = False
    for node in tree.body:
        if isinstance(node, _FUNCS):
            flush_loose(seen_def)
            seen_def = True
            s, e = _node_start(node), node.end_lineno
            chunks.append(_mk(rel, s, e, node.name, "function", lines, [node]))
            defs.append((s, e))
        elif isinstance(node, ast.ClassDef):
            flush_loose(seen_def)
            seen_def = True
            s, e = _node_start(node), node.end_lineno
            defs.append((s, e))
            if e - s + 1 <= cfg.max_class_lines:
                chunks.append(_mk(rel, s, e, node.name, "class", lines, [node]))
            else:
                methods = [m for m in node.body if isinstance(m, _FUNCS)]
                first_m = min([_node_start(m) for m in methods] or [e + 1])
                head_end = max(s, first_m - 1)
                head_nodes = (list(node.decorator_list) + list(node.bases) + [k.value for k in node.keywords]
                              + [b for b in node.body if not isinstance(b, _FUNCS)])
                chunks.append(_mk(rel, s, head_end, node.name, "class", lines, head_nodes))
                for m in methods:
                    ms, me = _node_start(m), m.end_lineno
                    chunks.append(_mk(rel, ms, me, f"{node.name}.{m.name}", "method", lines, [m],
                                      parent=node.name))
        else:
            loose.append(node)
    flush_loose(seen_def)

    # module header: everything before the first def/class (docstring, imports, constants)
    first_def = min([d[0] for d in defs] or [len(lines) + 1])
    if first_def > 1:
        header = _seg(lines, 1, first_def - 1).strip()
        if header:
            chunks.append(Chunk(rel, 1, first_def - 1, "<module>", "module_header", header))
    return chunks


# ---------------------------------------------------------------- docs / config / non-parsable
def chunk_text(path: Path, rel: str, cfg: ChunkConfig) -> list[Chunk]:
    lines = path.read_text(encoding="utf-8", errors="ignore").splitlines()
    chunks, i = [], 0
    while i < len(lines):
        j = min(i + cfg.window, len(lines))
        t = "\n".join(lines[i:j]).strip()
        if t:
            chunks.append(Chunk(rel, i + 1, j, Path(rel).name, "text", t))
        if j == len(lines):
            break
        i = j - cfg.overlap
    return chunks


# ---------------------------------------------------------------- repository
def chunk_repo(repo_dir: str | Path, cfg: ChunkConfig | None = None) -> list[Chunk]:
    cfg = cfg or ChunkConfig()
    repo_dir = Path(repo_dir)
    out: list[Chunk] = []
    for p in sorted(repo_dir.rglob("*")):
        if not p.is_file() or any(part in cfg.skip_dirs for part in p.relative_to(repo_dir).parts):
            continue
        if p.stat().st_size > cfg.max_file_bytes:
            continue
        rel = p.relative_to(repo_dir).as_posix()
        if p.suffix == ".py":
            out += chunk_python(p, rel, cfg)
        elif p.suffix in cfg.text_exts:
            out += chunk_text(p, rel, cfg)
    for i, c in enumerate(out):
        c.id = i
    return out


def save_chunks(chunks: list[Chunk], path: str | Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(asdict(c)) + "\n")


def load_chunks(path: str | Path) -> list[Chunk]:
    with open(path, encoding="utf-8") as f:
        return [Chunk(**json.loads(line)) for line in f]
