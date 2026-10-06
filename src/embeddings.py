"""Embeddings Module for RepoLens.

Responsible for:
- Initializing the local sentence-transformers embedding model.
- Converting text chunks and user queries into dense numerical vectors.
- Running entirely locally without external API dependencies.

Why Embeddings are Used in RAG:
1. Semantic Similarity: Rather than relying on exact keyword matching, embeddings map code
   and natural-language questions into a continuous vector space where semantically similar
   concepts are located close to each other.
2. Cross-Lingual Concept Matching: A question like "How do users authenticate?" maps closely
   to `function login(user, pass)` even without literal keyword matches.
3. Dimensionality: The all-MiniLM-L6-v2 model produces 384-dimensional vectors, optimizing
   both search speed and semantic representation quality.
"""

import os
from typing import List, Optional

# Prefer local cached models to avoid network timeout delays during offline/sandbox execution
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache to avoid reloading the model weights on repeated calls
_cached_embeddings: Optional[HuggingFaceEmbeddings] = None


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME) -> HuggingFaceEmbeddings:
    """Load or retrieve the cached Hugging Face embedding model.

    Runs locally using PyTorch and SentenceTransformers.
    """
    global _cached_embeddings
    if _cached_embeddings is None:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return _cached_embeddings


def embed_documents(texts: List[str], model_name: str = DEFAULT_MODEL_NAME) -> List[List[float]]:
    """Generate dense vector embeddings for a list of text strings."""
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


def embed_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> List[float]:
    """Generate a dense vector embedding for a single user query."""
    model = get_embedding_model(model_name)
    return model.embed_query(query)
