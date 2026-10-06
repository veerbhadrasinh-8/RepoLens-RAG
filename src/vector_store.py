"""Vector Store Module for RepoLens.

Responsible for:
- Managing the local FAISS (Facebook AI Similarity Search) vector database.
- Inserting document chunks and their associated embeddings.
- Persisting vector index and document metadata to disk.
- Loading the persisted FAISS index for fast query-time retrieval.

Why FAISS:
1. Purely local, lightweight, and in-process (no separate database server, Redis, or Docker needed).
2. Industry-standard vector indexing algorithms (L2 distance / Inner Product for cosine similarity).
3. Ideal for reproducible demonstrations and codebase exploration.
"""

import os
from typing import List, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.embeddings import get_embedding_model

DEFAULT_INDEX_DIR = "data/faiss_index"


def vector_store_exists(folder_path: str = DEFAULT_INDEX_DIR) -> bool:
    """Check if a serialized FAISS index exists in the target directory."""
    index_file = os.path.join(folder_path, "index.faiss")
    pkl_file = os.path.join(folder_path, "index.pkl")
    return os.path.exists(index_file) and os.path.exists(pkl_file)


def create_vector_store(
    chunks: List[Document],
    embedding_model=None,
) -> FAISS:
    """Create and populate a FAISS vector store from document chunks.

    Args:
        chunks: List of chunked Document objects with metadata.
        embedding_model: Embedding model instance (defaults to get_embedding_model()).

    Returns:
        Populated FAISS vector store instance.
    """
    if not chunks:
        raise ValueError("Cannot create vector store from empty chunk list.")

    if embedding_model is None:
        embedding_model = get_embedding_model()

    vector_store = FAISS.from_documents(
        documents=chunks,
        embedding=embedding_model,
    )
    return vector_store


def save_vector_store(
    vector_store: FAISS,
    folder_path: str = DEFAULT_INDEX_DIR,
) -> bool:
    """Persist the FAISS index and docstore metadata to the local filesystem.

    Args:
        vector_store: Instantiated FAISS vector store.
        folder_path: Directory path where index files will be stored.

    Returns:
        True if successfully saved, False otherwise.
    """
    try:
        os.makedirs(folder_path, exist_ok=True)
        vector_store.save_local(folder_path)
        return True
    except Exception as e:
        print(f"Error saving FAISS vector store to {folder_path}: {e}")
        return False


def load_vector_store(
    folder_path: str = DEFAULT_INDEX_DIR,
    embedding_model=None,
) -> Optional[FAISS]:
    """Load a persisted FAISS vector store from the local filesystem.

    Args:
        folder_path: Directory path where index files reside.
        embedding_model: Embedding model instance (defaults to get_embedding_model()).

    Returns:
        FAISS vector store instance, or None if files are missing or corrupted.
    """
    if not vector_store_exists(folder_path):
        return None

    if embedding_model is None:
        embedding_model = get_embedding_model()

    try:
        return FAISS.load_local(
            folder_path=folder_path,
            embeddings=embedding_model,
            allow_dangerous_deserialization=True,
        )
    except Exception as e:
        print(f"Error loading FAISS vector store from {folder_path}: {e}")
        return None
