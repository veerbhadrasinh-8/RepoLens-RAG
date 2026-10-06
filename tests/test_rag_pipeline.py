"""Tests for RAG Pipeline Orchestrator (Phase 9)."""

import pytest
from langchain_core.documents import Document

from src.rag_pipeline import RepoLensPipeline
from src.loader import load_from_directory


@pytest.fixture
def test_pipeline(tmp_path):
    index_dir = str(tmp_path / "faiss_pipe")
    pipeline = RepoLensPipeline(index_dir=index_dir)
    return pipeline


def test_query_before_indexing(test_pipeline):
    result = test_pipeline.query("How is authentication handled?")
    assert result["status"] == "not_indexed"
    assert "Please index the repository before asking questions" in result["answer"]
    assert result["sources"] == []


def test_query_empty_question(test_pipeline):
    # Manually mark as indexed with dummy
    test_pipeline.vector_store = "mock_store"
    result = test_pipeline.query("   ")
    assert result["status"] == "empty_question"
    assert "Please provide a non-empty question" in result["answer"]


def test_index_and_query_flow(test_pipeline):
    docs = load_from_directory("sample_repo")
    assert len(docs) > 0

    stats = test_pipeline.index_documents(docs, chunk_size=500, chunk_overlap=50)
    assert stats["status"] == "success"
    assert stats["files_indexed"] >= 5
    assert stats["chunks_created"] >= 5
    assert test_pipeline.is_indexed() is True

    # Custom LLM response simulator
    def mock_llm_fn(prompt: str) -> str:
        assert "REPOSITORY CONTEXT:" in prompt
        assert "USER QUESTION:" in prompt
        return "Authentication is implemented using JWT tokens in backend/auth.js."

    result = test_pipeline.query(
        question="How does authentication work?",
        top_k=3,
        custom_llm_fn=mock_llm_fn,
    )

    assert result["status"] == "success"
    assert "JWT tokens in backend/auth.js" in result["answer"]
    assert len(result["sources"]) > 0
    assert "backend/auth.js" in result["sources"] or "backend/middleware.js" in result["sources"]
    assert len(result["retrieved_chunks"]) == 3
    for chunk in result["retrieved_chunks"]:
        assert "source" in chunk
        assert "score" in chunk
        assert "content" in chunk


def test_missing_api_key_handling(test_pipeline, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    docs = [Document(page_content="test code", metadata={"source": "test.js"})]
    test_pipeline.index_documents(docs)

    result = test_pipeline.query("What does this code do?")
    assert result["status"] == "missing_api_key"
    assert "Please add GEMINI_API_KEY to .env" in result["answer"]
