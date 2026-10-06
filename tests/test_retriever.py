"""Tests for Retriever module (Phase 6)."""

import pytest
from langchain_core.documents import Document
from src.vector_store import create_vector_store
from src.retriever import retrieve_relevant_chunks, extract_unique_sources


@pytest.fixture
def populated_vector_store():
    chunks = [
        Document(
            page_content="function authenticateUser(email, password) { const token = jwt.sign(...); return token; }",
            metadata={"source": "backend/auth.js", "file_path": "backend/auth.js", "file_type": "javascript", "chunk_index": 0},
        ),
        Document(
            page_content="function verifyAuthToken(req, res, next) { jwt.verify(token); next(); }",
            metadata={"source": "backend/middleware.js", "file_path": "backend/middleware.js", "file_type": "javascript", "chunk_index": 0},
        ),
        Document(
            page_content="const pool = new Pool({ host: 'localhost', database: 'taskflow_db' });",
            metadata={"source": "backend/database.js", "file_path": "backend/database.js", "file_type": "javascript", "chunk_index": 0},
        ),
        Document(
            page_content="# TaskFlow API - A task manager with JWT auth and PostgreSQL storage",
            metadata={"source": "README.md", "file_path": "README.md", "file_type": "markdown", "chunk_index": 0},
        ),
    ]
    return create_vector_store(chunks)


def test_retrieve_auth_query(populated_vector_store):
    chunks = retrieve_relevant_chunks(populated_vector_store, "How is authentication handled?", top_k=2)
    assert len(chunks) == 2
    sources = [c["source"] for c in chunks]
    # Either auth.js or middleware.js should be the top retrieved source
    assert "backend/auth.js" in sources or "backend/middleware.js" in sources
    for c in chunks:
        assert "content" in c
        assert "score" in c
        assert isinstance(c["score"], float)


def test_retrieve_database_query(populated_vector_store):
    chunks = retrieve_relevant_chunks(populated_vector_store, "Where is the database connection configured?", top_k=1)
    assert len(chunks) == 1
    assert chunks[0]["source"] == "backend/database.js"


def test_retrieve_empty_query(populated_vector_store):
    chunks = retrieve_relevant_chunks(populated_vector_store, "   ", top_k=4)
    assert chunks == []


def test_extract_unique_sources():
    sample_retrieved = [
        {"source": "backend/auth.js"},
        {"source": "backend/middleware.js"},
        {"source": "backend/auth.js"},  # Duplicate
        {"source": "README.md"},
    ]
    unique = extract_unique_sources(sample_retrieved)
    assert unique == ["backend/auth.js", "backend/middleware.js", "README.md"]
