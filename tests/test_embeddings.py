"""Tests for Embeddings module (Phase 4)."""

import math
from src.embeddings import get_embedding_model, embed_query, embed_documents


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
