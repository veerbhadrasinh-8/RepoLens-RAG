"""Chunker Module for RepoLens.

Responsible for:
- Splitting repository documents into semantically coherent chunks.
- Maintaining all source metadata (source, file_path, file_type, chunk_index).
- Providing configurable chunk size and overlap parameters.

Why Chunking is Essential in RAG:
1. Retrieval Precision: Splitting large files ensures vector similarity retrieves the exact
   function or block of code answering the user's question, rather than diluting relevance.
2. Context-Window Efficiency: Prevents filling the LLM context window with thousands of
   unrelated lines from the same file.
3. Lower Token Usage & Latency: Sending fewer, targeted chunks reduces API token cost and speeds up response time.
4. Better Semantic Embeddings: Embedding models perform best when encoding coherent, bounded
   units of text (300-1000 characters) rather than entire multi-page files.
"""

from typing import List, Dict, Any
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

DEFAULT_CHUNK_SIZE = 800
DEFAULT_CHUNK_OVERLAP = 100

# Separators prioritizing code structure and markdown headings before arbitrary line breaks
CODE_SEPARATORS = [
    "\nclass ",
    "\ndef ",
    "\nfunction ",
    "\nconst ",
    "\n## ",
    "\n### ",
    "\n\n",
    "\n",
    " ",
    "",
]


def split_documents(
    documents: List[Document],
    chunk_size: int = DEFAULT_CHUNK_SIZE,
    chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> List[Document]:
    """Split a list of documents into chunks while preserving metadata.

    Args:
        documents: List of input Document objects.
        chunk_size: Target maximum character length per chunk.
        chunk_overlap: Number of characters shared between consecutive chunks.

    Returns:
        List of chunked Document objects with preserved metadata.
    """
    if not documents:
        return []

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=CODE_SEPARATORS,
        keep_separator=True,
    )

    all_chunks: List[Document] = []

    for doc in documents:
        # Split document text into chunk documents
        chunks = splitter.split_documents([doc])
        for idx, chunk in enumerate(chunks):
            # Ensure all original metadata is preserved and track chunk sequence
            chunk.metadata = dict(doc.metadata)
            chunk.metadata["chunk_index"] = idx
            all_chunks.append(chunk)

    return all_chunks


def get_chunking_stats(
    documents: List[Document],
    chunks: List[Document],
) -> Dict[str, Any]:
    """Calculate summary statistics for display in UI or logs."""
    return {
        "files_loaded": len(documents),
        "chunks_created": len(chunks),
        "avg_chunks_per_file": round(len(chunks) / len(documents), 2) if documents else 0,
    }
