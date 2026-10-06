"""Tests for Vector Store module (Phase 5)."""

import os
import shutil
import pytest
from langchain_core.documents import Document

from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
)


@pytest.fixture
def sample_chunks():
    return [
        Document(
            page_content="function authenticateUser(user, pass) { return jwt.sign(user); }",
            metadata={"source": "backend/auth.js", "file_path": "backend/auth.js", "file_type": "javascript"},
        ),
        Document(
            page_content="const pool = new Pool({ database: 'test_db' });",
            metadata={"source": "backend/database.js", "file_path": "backend/database.js", "file_type": "javascript"},
        ),
        Document(
            page_content="# TaskFlow Documentation and User Guide",
            metadata={"source": "README.md", "file_path": "README.md", "file_type": "markdown"},
        ),
    ]


def test_create_vector_store(sample_chunks):
    vs = create_vector_store(sample_chunks)
    assert vs is not None

    # Perform a test similarity query
    results = vs.similarity_search("database connection", k=1)
    assert len(results) == 1
    assert "database.js" in results[0].metadata["source"]


def test_save_and_load_vector_store(sample_chunks, tmp_path):
    temp_dir = str(tmp_path / "test_faiss")

    assert vector_store_exists(temp_dir) is False

    vs = create_vector_store(sample_chunks)
    saved = save_vector_store(vs, temp_dir)
    assert saved is True
    assert vector_store_exists(temp_dir) is True

    # Load from disk
    loaded_vs = load_vector_store(temp_dir)
    assert loaded_vs is not None

    results = loaded_vs.similarity_search("authenticate user token", k=1)
    assert len(results) == 1
    assert "auth.js" in results[0].metadata["source"]


def test_create_vector_store_empty_error():
    with pytest.raises(ValueError, match="Cannot create vector store from empty chunk list"):
        create_vector_store([])
