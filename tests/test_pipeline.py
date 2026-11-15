from codeqa.chunking import chunk_repo
from codeqa.config import Config, EmbedConfig
from codeqa.embedding import HashingEmbedder
from codeqa.index import IndexStore
from codeqa.llm import EchoLLM
from codeqa.pipeline import answer_question, build_user_prompt
from codeqa.prompts import SYSTEM_PROMPT


def test_prompt_contains_cited_header_and_question(toy_repo):
    chunks = chunk_repo(toy_repo)
    p = build_user_prompt("What is App?", chunks[:2])
    assert p.startswith("Repository excerpts:") and "(lines " in p and p.endswith("Question: What is App?")


def test_answer_question_uses_llm(toy_repo):
    out = answer_question(EchoLLM(), "Q?", chunk_repo(toy_repo)[:1], Config())
    assert out.startswith("[echo]")


def test_index_roundtrip_and_model_guard(toy_repo, tmp_path):
    import pytest
    chunks = chunk_repo(toy_repo)
    store = IndexStore.build(chunks, HashingEmbedder(), EmbedConfig(model="hashing"), show_progress=False)
    store.save(tmp_path / "idx")
    again = IndexStore.load(tmp_path / "idx", EmbedConfig(model="hashing"))
    assert len(again.chunks) == len(chunks) and again.chunks[3].text == chunks[3].text
    with pytest.raises(ValueError):
        IndexStore.load(tmp_path / "idx", EmbedConfig(model="other-model"))
    assert "ONLY" in SYSTEM_PROMPT
