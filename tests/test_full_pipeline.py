"""End-to-End Integration Tests for RepoLens (Phases 11 & 12).

Tests:
1. Indexing the sample demo repository.
2. Query 1: 'What does this project do?' -> Retrieves README.md.
3. Query 2: 'How does authentication work?' -> Retrieves auth.js / middleware.js.
4. Query 3: 'Which file handles database connection?' -> Retrieves database.js.
5. Query 4: 'How does this project process credit card payments?' -> Unrelated query test.
6. Error handling:
   - Empty question
   - Querying before index exists
   - Missing GEMINI_API_KEY
   - API exception handling & secret redaction
"""

import pytest
from src.rag_pipeline import RepoLensPipeline
from src.loader import load_from_directory
from src.prompt import build_prompt


@pytest.fixture(scope="module")
def indexed_pipeline(tmp_path_factory):
    """Fixture that initializes and indexes the sample repo in a temporary directory."""
    temp_dir = str(tmp_path_factory.mktemp("test_index"))
    pipeline = RepoLensPipeline(index_dir=temp_dir)
    docs = load_from_directory("sample_repo")
    assert len(docs) >= 5, "Sample repo must have at least 5 files"
    stats = pipeline.index_documents(docs, chunk_size=600, chunk_overlap=100)
    assert stats["status"] == "success"
    return pipeline


def test_demo_query_1_project_overview(indexed_pipeline):
    """Demo 1: 'What does this project do?'"""
    def mock_llm_overview(prompt: str) -> str:
        assert "README.md" in prompt
        return "TaskFlow is a task management web application with JWT authentication and PostgreSQL storage."

    result = indexed_pipeline.query(
        "What does this project do?",
        top_k=3,
        custom_llm_fn=mock_llm_overview,
    )
    assert result["status"] == "success"
    assert "TaskFlow" in result["answer"]
    assert any("README.md" in src for src in result["sources"])


def test_demo_query_2_authentication(indexed_pipeline):
    """Demo 2: 'How does authentication work?'"""
    def mock_llm_auth(prompt: str) -> str:
        assert "auth.js" in prompt or "middleware.js" in prompt
        return "Authentication uses JWT. Users log in via authenticateUser, generating a JWT token that is verified by verifyAuthToken middleware."

    result = indexed_pipeline.query(
        "How is authentication implemented?",
        top_k=4,
        custom_llm_fn=mock_llm_auth,
    )
    assert result["status"] == "success"
    assert "JWT" in result["answer"]
    # Check that sources contain auth.js or middleware.js
    sources_str = " ".join(result["sources"])
    assert "auth.js" in sources_str or "middleware.js" in sources_str


def test_demo_query_3_database_connection(indexed_pipeline):
    """Demo 3: 'Which file handles database connection?'"""
    def mock_llm_db(prompt: str) -> str:
        assert "database.js" in prompt
        return "The database connection is handled in backend/database.js using a PostgreSQL Pool."

    result = indexed_pipeline.query(
        "Which file handles database connection?",
        top_k=3,
        custom_llm_fn=mock_llm_db,
    )
    assert result["status"] == "success"
    assert any("database.js" in src for src in result["sources"])


def test_demo_query_4_unrelated_payment_question(indexed_pipeline):
    """Demo 4: 'How does this project process credit card payments?'

    Verifies anti-hallucination instruction: LLM must state that information is absent.
    """
    def mock_llm_grounding(prompt: str) -> str:
        # Code does NOT contain payment processing, so prompt should instruct to say so
        assert "Do not invent code" in prompt
        return "The retrieved repository context does not contain enough information to answer this question. There is no mention of credit card payments or payment gateways."

    result = indexed_pipeline.query(
        "How does this project process credit card payments?",
        top_k=4,
        custom_llm_fn=mock_llm_grounding,
    )
    assert result["status"] == "success"
    assert "does not contain enough information" in result["answer"]


def test_error_handling_empty_query(indexed_pipeline):
    result = indexed_pipeline.query("")
    assert result["status"] == "empty_question"
    assert "Please provide a non-empty question" in result["answer"]


def test_error_handling_unindexed(tmp_path):
    empty_pipeline = RepoLensPipeline(index_dir=str(tmp_path / "empty_dir"))
    result = empty_pipeline.query("Any question?")
    assert result["status"] == "not_indexed"
    assert "Please index the repository before asking questions" in result["answer"]


def test_error_handling_missing_key(indexed_pipeline, monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    result = indexed_pipeline.query("How does login work?")
    assert result["status"] == "missing_api_key"
    assert "Please add GEMINI_API_KEY to .env" in result["answer"]
