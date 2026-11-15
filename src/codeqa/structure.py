"""Repository structure graph (the "structure-aware" half of the project).

Nodes are chunks. Edges come from static analysis of the Python AST:

  call       chunk A calls a function/class that is defined in chunk B  (A -> B)
  ref        chunk A mentions a name that resolves to chunk B (not a call: a type, constant, decorator...)
             Both call and ref edges are found by following import statements, which is how the proposal's
             "import relationships" enter the graph: `from .helpers import render` tells us which chunk
             `render(...)` refers to.
  called_by  reverse of call AND ref: B <- A  ("where is this used?")
  parent     method chunk -> the class chunk that owns it
  header     chunk -> the module header (imports, constants) of its own file

Name resolution is deliberately simple and *honest about its limits*:
  * It follows `import x`, `from x import y [as z]`, relative imports and one-hop re-exports through
    `__init__.py` (e.g. `from flask import Flask`).
  * It does NOT do type inference, so `obj.method()` is resolved by method name only, and only when the
    name is unambiguous (<= 3 candidates in the same file or in files this file imports).
  * Dynamic behaviour (getattr, decorators that register things, string-based imports) is invisible.
"""
from __future__ import annotations

import ast
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from .chunking import Chunk

MAX_METHOD_CANDIDATES = 3
MAX_REEXPORT_DEPTH = 5


@dataclass
class _FileInfo:
    path: str
    module: str
    is_pkg: bool
    bindings: dict = field(default_factory=dict)   # local name -> (absolute module, original name | None)
    imported_modules: set = field(default_factory=set)


class RepoGraph:
    def __init__(self, repo_dir: str | Path, chunks: list[Chunk]):
        self.repo = Path(repo_dir)
        self.chunks = chunks
        self.files: dict[str, _FileInfo] = {}
        self.module_to_file: dict[str, str] = {}
        self.file_imports: dict[str, set] = defaultdict(set)
        self.file_imported_by: dict[str, set] = defaultdict(set)
        self.out_edges: dict[int, dict[int, str]] = defaultdict(dict)   # idx -> {neighbour: "call"|"ref"}
        self.in_edges: dict[int, set] = defaultdict(set)
        self.parent_edge: dict[int, int] = {}
        self.header_of: dict[str, int] = {}
        self._top_defs: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(list))
        self._methods_by_name: dict[str, list[int]] = defaultdict(list)
        self._build()

    # ================================================================== build
    def _build(self) -> None:
        py_files = sorted({c.path for c in self.chunks if c.path.endswith(".py")})
        self._index_modules(py_files)
        self._parse_imports()
        self._index_defs()
        self._link_files()
        self._link_chunks()

    # ---- modules ------------------------------------------------------
    def _module_name(self, rel: str) -> tuple[str, bool]:
        p = Path(rel)
        is_pkg = p.name == "__init__.py"
        parts: list[str] = []
        d = (self.repo / rel).parent
        while d != self.repo and (d / "__init__.py").exists():
            parts.insert(0, d.name)
            d = d.parent
        if not is_pkg:
            parts.append(p.stem)
        return ".".join(parts), is_pkg

    def _index_modules(self, py_files: list[str]) -> None:
        seen: dict[str, list[str]] = defaultdict(list)
        for rel in py_files:
            mod, is_pkg = self._module_name(rel)
            self.files[rel] = _FileInfo(rel, mod, is_pkg)
            if mod:
                seen[mod].append(rel)
        # ambiguous module names (e.g. several `conftest`) are dropped rather than guessed
        self.module_to_file = {m: fs[0] for m, fs in seen.items() if len(fs) == 1}

    def _abs_module(self, info: _FileInfo, level: int, module: str | None) -> str | None:
        if level == 0:
            return module
        pkg = info.module if info.is_pkg else info.module.rpartition(".")[0]
        parts = pkg.split(".") if pkg else []
        up = level - 1
        if up > len(parts):
            return None
        base = ".".join(parts[:len(parts) - up])
        if module:
            return f"{base}.{module}" if base else module
        return base or None

    def _parse_imports(self) -> None:
        for rel, info in self.files.items():
            try:
                tree = ast.parse((self.repo / rel).read_text(encoding="utf-8", errors="ignore"))
            except SyntaxError:
                continue
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for a in node.names:
                        info.imported_modules.add(a.name)
                        if a.asname:
                            info.bindings[a.asname] = (a.name, None)
                        else:
                            top = a.name.split(".")[0]
                            info.bindings[top] = (top, None)
                elif isinstance(node, ast.ImportFrom):
                    base = self._abs_module(info, node.level, node.module)
                    if base is None:
                        continue
                    info.imported_modules.add(base)
                    for a in node.names:
                        if a.name == "*":
                            continue
                        info.bindings[a.asname or a.name] = (base, a.name)
                        info.imported_modules.add(f"{base}.{a.name}")   # may be a submodule

    # ---- definitions --------------------------------------------------
    def _index_defs(self) -> None:
        classes: dict[tuple[str, str], int] = {}
        for i, c in enumerate(self.chunks):
            if not c.path.endswith(".py"):
                continue
            if c.kind in ("function", "class"):
                self._top_defs[c.path][c.name].append(i)
                if c.kind == "class":
                    classes[(c.path, c.name)] = i
            elif c.kind == "method":
                self._methods_by_name[c.name.rpartition(".")[2]].append(i)
            elif c.kind == "module_header":
                self.header_of[c.path] = i
        for i, c in enumerate(self.chunks):
            if c.kind == "method" and c.parent and (c.path, c.parent) in classes:
                self.parent_edge[i] = classes[(c.path, c.parent)]

    def _link_files(self) -> None:
        for rel, info in self.files.items():
            for mod in info.imported_modules:
                tgt = self.module_to_file.get(mod)
                if tgt and tgt != rel:
                    self.file_imports[rel].add(tgt)
                    self.file_imported_by[tgt].add(rel)

    # ---- name resolution ---------------------------------------------
    def _resolve_symbol(self, file: str, name: str, depth: int = 0) -> list[int]:
        """Chunks that define `name` as seen from `file` (local def, import, or one-hop re-export)."""
        if name in self._top_defs.get(file, {}):
            return self._top_defs[file][name]
        if depth >= MAX_REEXPORT_DEPTH:
            return []
        info = self.files.get(file)
        b = info.bindings.get(name) if info else None
        if b and b[1] is not None:
            mod, orig = b
            tgt = self.module_to_file.get(mod)
            if tgt and tgt != file:
                return self._resolve_symbol(tgt, orig, depth + 1)
        return []

    def _resolve_attr(self, file: str, base: str, attr: str) -> list[int]:
        info = self.files[file]
        b = info.bindings.get(base) if base else None
        if b:
            mod, orig = b
            module = mod if orig is None else f"{mod}.{orig}"    # `import m` or `from pkg import submodule`
            tgt = self.module_to_file.get(module)
            if tgt:
                return self._resolve_symbol(tgt, attr)
            return []        # attribute of an imported class/object: needs types, so skip
        # unknown base (self, cls, local variable...) -> method lookup by bare name, if unambiguous
        cands = self._methods_by_name.get(attr, [])
        same = [i for i in cands if self.chunks[i].path == file]
        if same:
            return same[:MAX_METHOD_CANDIDATES] if len(same) <= MAX_METHOD_CANDIDATES else []
        near = [i for i in cands if self.chunks[i].path in self.file_imports.get(file, ())]
        return near if 0 < len(near) <= MAX_METHOD_CANDIDATES else []

    def _link_chunks(self) -> None:
        for i, c in enumerate(self.chunks):
            if not c.path.endswith(".py") or c.kind in ("module_header", "text"):
                continue
            found: dict[int, str] = {}
            calls = set(c.calls)
            for name in c.names:
                for j in self._resolve_symbol(c.path, name):
                    found[j] = "call" if name in calls else found.get(j, "ref")
            for ac in c.attr_calls:
                base, _, attr = ac.partition(".")
                for j in self._resolve_attr(c.path, base, attr):
                    found[j] = "call"
            for j, etype in found.items():
                if j != i:
                    self.out_edges[i][j] = etype
                    self.in_edges[j].add(i)

    # ================================================================== queries
    def neighbors(self, idx: int, max_in: int | None = None) -> list[tuple[int, str]]:
        """All structural neighbours of a chunk as (chunk index, edge type).

        `max_in` truncates the reverse edges in chunk-id (i.e. file path) order. Retrieval does NOT use it:
        it ranks reverse edges by similarity to the question first (see retrieval.py), because path order
        would keep e.g. examples/ and src/ callers and silently drop every caller under tests/.
        """
        out = [(j, t) for j, t in self.out_edges.get(idx, {}).items()]
        out += [(j, "called_by") for j in sorted(self.in_edges.get(idx, ()))[:max_in]]
        if idx in self.parent_edge:
            out.append((self.parent_edge[idx], "parent"))
        h = self.header_of.get(self.chunks[idx].path)
        if h is not None and h != idx:
            out.append((h, "header"))
        return out

    def stats(self) -> dict:
        n_py = sum(1 for c in self.chunks if c.path.endswith(".py"))
        with_out = sum(1 for i in self.out_edges if self.out_edges[i])
        return {
            "python_files": len(self.files),
            "resolvable_modules": len(self.module_to_file),
            "python_chunks": n_py,
            "file_import_edges": sum(len(v) for v in self.file_imports.values()),
            "chunk_edges": sum(len(v) for v in self.out_edges.values()),
            "chunks_with_out_edges": with_out,
            "pct_python_chunks_with_out_edges": round(100 * with_out / max(n_py, 1), 1),
        }
