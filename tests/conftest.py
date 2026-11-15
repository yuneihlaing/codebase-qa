import textwrap
from pathlib import Path

import pytest


def write(root: Path, rel: str, body: str) -> None:
    p = root / rel
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(textwrap.dedent(body), encoding="utf-8")


@pytest.fixture
def toy_repo(tmp_path) -> Path:
    """A miniature src-layout package that exercises relative imports and a re-export."""
    r = tmp_path / "toyrepo"
    write(r, "README.md", "# Toy\nA toy web framework.\n")
    write(r, "src/toy/__init__.py", """\
        from .app import App as App
        from .helpers import render
    """)
    write(r, "src/toy/helpers.py", '''\
        """Helper utilities."""
        import json

        def render(template, **ctx):
            """Render a template with context."""
            return template.format(**ctx)

        def dumps(obj):
            return json.dumps(obj)
    ''')
    write(r, "src/toy/app.py", '''\
        """Application object."""
        from . import helpers
        from .helpers import render

        DEFAULT = 1

        class App:
            def run(self):
                return self.handle("/")

            def handle(self, path):
                return render("hi {x}", x=path)

            def to_json(self, obj):
                return helpers.dumps(obj)

        def make_app():
            return App()
    ''')
    write(r, "tests/test_app.py", """\
        from toy import App

        def test_app():
            assert App().run()
    """)
    return r
