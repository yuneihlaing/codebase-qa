from codeqa.eval.deterministic import (aggregate, cited_spans, code_identifiers, extract_paths, score_record,
                                       token_f1)
from codeqa.eval.judge import _parse_json, judge_pairwise
from codeqa.eval.stats import paired_bootstrap, win_tie_loss


def test_extract_paths():
    assert extract_paths("See `src/flask/app.py` and README.md, also docs/conf.py.") == [
        "src/flask/app.py", "README.md", "docs/conf.py"]


def test_cited_spans_formats():
    s = cited_spans("src/a.py (lines 10-20) and b.py:5 and c.py")
    assert s == [("src/a.py", 10, 20), ("b.py", 5, 5), ("c.py", None, None)]


def test_code_identifiers():
    ids = code_identifiers("The `Flask` class calls make_response() and uses url_map; also the word these.")
    assert {"flask", "make_response", "url_map"} <= ids and "these" not in ids


def test_token_f1_bounds():
    assert token_f1("a b c", "a b c") == 1.0 and token_f1("x", "y") == 0.0


def test_score_record_citations_and_recall():
    rec = {"answer": "It lives in app.py (lines 5-9) and fake.py (lines 1-2).",
           "reference": "See `src/toy/app.py`: App.make_app() builds the object.",
           "retrieved": [{"path": "src/toy/app.py", "start": 1, "end": 12}],
           "retrieved_text": "def make_app(): return App()"}
    s = score_record(rec)
    assert s["ref_file_recall"] == 1.0
    assert s["citation_file_valid"] == 0.5 and s["citation_line_valid"] == 0.5
    assert s["ref_ident_recall_ctx"] is not None and s["abstained"] == 0.0


def test_aggregate_skips_none():
    a = aggregate([{"x": 1.0}, {"x": None}, {"x": 0.0}])
    assert a["x"] == 0.5 and a["x__n"] == 2


def test_bootstrap_detects_shift():
    a = [0.1, 0.2, 0.15, 0.1, 0.2, 0.3, 0.25, 0.2]
    b = [x + 0.2 for x in a]
    r = paired_bootstrap(a, b, n=2000)
    assert r["significant"] and abs(r["mean_diff"] - 0.2) < 1e-9
    assert win_tie_loss(a, b) == {"b_wins": 8, "ties": 0, "a_wins": 0}


def test_parse_json_tolerates_chatter():
    assert _parse_json('Sure! ```json\n{"winner": "A"}\n```') == {"winner": "A"}


class _Scripted:
    name = "scripted"
    def __init__(self, replies): self.replies = list(replies)
    def generate(self, s, u, n=0): return self.replies.pop(0)


def test_pairwise_position_bias_becomes_tie():
    # judge always says "A" regardless of order -> disagreement after mapping -> tie
    assert judge_pairwise(_Scripted(['{"winner":"A"}', '{"winner":"A"}']), "q", "r", "x", "y") == "tie"
    # consistent preference for the same underlying answer -> that answer wins
    assert judge_pairwise(_Scripted(['{"winner":"B"}', '{"winner":"A"}']), "q", "r", "x", "y") == "B"
