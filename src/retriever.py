"""Retriever Module for RepoLens.

Responsible for:
- Receiving the user's natural language question.
- Querying FAISS to find the top-K semantically closest chunks.
- Extracting chunk content, source file metadata, and similarity distance scores.
- Providing helper utilities to extract deduplicated source lists.

Important Conceptual Distinction:
- The retriever ONLY finds relevant chunks from the vector database.
- The retriever does NOT generate answers or hallucinate responses.
- Scores returned are distance/similarity metric scores, NOT probabilities.
"""

from typing import List, Dict, Any
from langchain_community.vectorstores import FAISS

DEFAULT_TOP_K = 4


def retrieve_relevant_chunks(
    vector_store: FAISS,
    query: str,
    top_k: int = DEFAULT_TOP_K,
) -> List[Dict[str, Any]]:
    """Retrieve top-K most semantically relevant chunks for a user question.

    Args:
        vector_store: Instantiated and populated FAISS vector store.
        query: The user's natural language question.
        top_k: Number of most relevant chunks to return (default: 4).

    Returns:
        List of dictionaries with keys:
            - 'content': text content of the chunk
            - 'source': source file path
            - 'file_path': relative file path
            - 'file_type': language / extension type
            - 'chunk_index': chunk sequence index in original file
            - 'score': vector distance metric (lower is closer in L2 space)
    """
    cleaned_query = query.strip()
    if not cleaned_query:
        return []

    # FAISS similarity_search_with_score returns List[Tuple[Document, float]]
    # The float is the L2 distance between the query vector and chunk vector
    results_with_scores = vector_store.similarity_search_with_score(cleaned_query, k=top_k)

    retrieved: List[Dict[str, Any]] = []
    for doc, score in results_with_scores:
        metadata = doc.metadata or {}
        retrieved.append({
            "content": doc.page_content,
            "source": metadata.get("source", "unknown"),
            "file_path": metadata.get("file_path", metadata.get("source", "unknown")),
            "file_type": metadata.get("file_type", "unknown"),
            "chunk_index": metadata.get("chunk_index", 0),
            "score": round(float(score), 4),
        })

    return retrieved


def extract_unique_sources(retrieved_chunks: List[Dict[str, Any]]) -> List[str]:
    """Extract an ordered, deduplicated list of source file paths from retrieved chunks."""
    sources: List[str] = []
    seen = set()
    for chunk in retrieved_chunks:
        src = chunk.get("source")
        if src and src not in seen:
            seen.add(src)
            sources.append(src)
    return sources
