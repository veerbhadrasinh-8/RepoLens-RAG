"""Tests for Embeddings module (Phase 4)."""

import math
from src.embeddings import get_embedding_model, embed_query, embed_documents, is_model_cached, DEFAULT_MODEL_NAME


def test_is_model_cached():
    # Built-in model is already cached locally
    assert is_model_cached(DEFAULT_MODEL_NAME) is True
    # Non-existent model should return False
    assert is_model_cached("non-existent-org/non-existent-model-98765") is False


def test_embedding_model_init():
    model = get_embedding_model()
    assert model is not None


def test_embed_query_dimension():
    vec = embed_query("How is authentication implemented?")
    assert isinstance(vec, list)
    assert len(vec) == 384  # MiniLM-L6-v2 dimensionality is 384
    assert all(isinstance(val, float) for val in vec)


def test_embed_documents():
    texts = [
        "const jwt = require('jsonwebtoken');",
        "const { Pool } = require('pg');",
    ]
    vectors = embed_documents(texts)
    assert len(vectors) == 2
    assert len(vectors[0]) == 384
    assert len(vectors[1]) == 384


def test_normalized_embeddings():
    vec = embed_query("Check normalization vector")
    # L2 norm of normalized embeddings should be approximately 1.0
    l2_norm = math.sqrt(sum(v * v for v in vec))
    assert math.isclose(l2_norm, 1.0, rel_tol=1e-3)


def test_uncached_model_downloads_online():
    from unittest.mock import patch
    from src import embeddings

    old_cached = embeddings._cached_embeddings
    try:
        embeddings._cached_embeddings = None
        with patch("src.embeddings.is_model_cached", return_value=False), \
             patch("src.embeddings.HuggingFaceEmbeddings") as mock_hf:
            mock_hf.return_value = "online_model"
            model = embeddings.get_embedding_model("new-org/new-model")
            assert model == "online_model"
            call_kwargs = mock_hf.call_args[1]
            assert "local_files_only" not in call_kwargs["model_kwargs"]
    finally:
        embeddings._cached_embeddings = old_cached


def test_cached_failure_retries_online():
    from unittest.mock import patch, MagicMock
    from src import embeddings

    old_cached = embeddings._cached_embeddings
    try:
        embeddings._cached_embeddings = None
        calls = []

        def mock_init(*args, **kwargs):
            calls.append(kwargs)
            if len(calls) == 1:
                raise OSError("Simulated cache read error")
            return MagicMock()

        with patch("src.embeddings.is_model_cached", return_value=True), \
             patch("src.embeddings.HuggingFaceEmbeddings", side_effect=mock_init):
            model = embeddings.get_embedding_model()
            assert len(calls) == 2
            assert calls[0]["model_kwargs"].get("local_files_only") is True
            assert "local_files_only" not in calls[1]["model_kwargs"]
    finally:
        embeddings._cached_embeddings = old_cached


def test_gemini_fallback_when_hf_fails():
    from unittest.mock import patch
    from src import embeddings

    old_cached = embeddings._cached_embeddings
    try:
        embeddings._cached_embeddings = None
        with patch("src.embeddings.is_model_cached", return_value=False), \
             patch("src.embeddings.HuggingFaceEmbeddings", side_effect=OSError("Cannot reach huggingface.co")), \
             patch("src.llm.get_gemini_api_key", return_value="fake-gemini-key"), \
             patch("langchain_google_genai.GoogleGenerativeAIEmbeddings") as mock_gemini:
            mock_gemini.return_value = "gemini_embeddings_instance"
            model = embeddings.get_embedding_model()
            assert model == "gemini_embeddings_instance"
            assert "Gemini" in embeddings.get_active_embedding_desc()
    finally:
        embeddings._cached_embeddings = old_cached


def test_rag_pipeline_lazy_embedding():
    from src.rag_pipeline import RepoLensPipeline
    pipeline = RepoLensPipeline(index_dir="non_existent_test_dir_123")
    assert pipeline._embedding_model is None
    # Accessing property initializes embedding
    model = pipeline.embedding_model
    assert model is not None
    assert pipeline._embedding_model is not None


