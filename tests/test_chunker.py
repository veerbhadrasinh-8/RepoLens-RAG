"""Tests for Chunker module (Phase 3)."""

from langchain_core.documents import Document
from src.chunker import split_documents, get_chunking_stats


def test_split_documents_empty():
    chunks = split_documents([])
    assert chunks == []


def test_split_documents_metadata_preservation():
    doc = Document(
        page_content="function calculateTotal() {\n  return 100;\n}",
        metadata={"source": "utils.js", "file_path": "utils.js", "file_type": "javascript"},
    )
    chunks = split_documents([doc], chunk_size=500, chunk_overlap=50)
    assert len(chunks) == 1
    assert chunks[0].metadata["source"] == "utils.js"
    assert chunks[0].metadata["file_path"] == "utils.js"
    assert chunks[0].metadata["file_type"] == "javascript"
    assert chunks[0].metadata["chunk_index"] == 0


def test_split_large_document():
    # Long text with repeated lines
    long_content = "\n".join([f"line_{i} = 'sample code statement';" for i in range(100)])
    doc = Document(
        page_content=long_content,
        metadata={"source": "big_file.py", "file_path": "big_file.py", "file_type": "python"},
    )

    chunks = split_documents([doc], chunk_size=300, chunk_overlap=50)
    assert len(chunks) > 1

    # Check each chunk retains metadata
    for i, chunk in enumerate(chunks):
        assert chunk.metadata["source"] == "big_file.py"
        assert chunk.metadata["file_path"] == "big_file.py"
        assert chunk.metadata["chunk_index"] == i
        assert len(chunk.page_content) <= 350  # Allows small boundary tolerance


def test_chunking_stats():
    docs = [
        Document(page_content="short content 1", metadata={"source": "f1.txt"}),
        Document(page_content="short content 2", metadata={"source": "f2.txt"}),
    ]
    chunks = split_documents(docs)
    stats = get_chunking_stats(docs, chunks)
    assert stats["files_loaded"] == 2
    assert stats["chunks_created"] == 2
    assert stats["avg_chunks_per_file"] == 1.0
