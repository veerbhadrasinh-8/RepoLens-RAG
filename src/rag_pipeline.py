"""RAG Pipeline Orchestrator Module for RepoLens.

Responsible for orchestrating the complete end-to-end pipeline:
1. Indexing Flow:
   Repository Files → Text Extraction → Chunking → Embeddings → FAISS Index

2. Query Flow:
   User Question → Query Embedding → FAISS Similarity Search → Top-K Chunks
   → Grounded Prompt Construction → Gemini LLM → Final Answer + Verified Sources

Key Architectural Principles:
- Separation of Concerns: The retriever finds information; the LLM understands and synthesizes it.
- Grounded Source Attribution: Sources are strictly extracted from retrieval metadata,
  never parsed from potentially hallucinated LLM text.
- Token Efficiency: Only the top-K relevant chunks are sent to the LLM, never the entire repository.
"""

from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS

from src.loader import load_from_uploaded_files, load_from_directory, load_from_zip
from src.chunker import split_documents, get_chunking_stats, DEFAULT_CHUNK_SIZE, DEFAULT_CHUNK_OVERLAP
from src.embeddings import get_embedding_model
from src.vector_store import (
    create_vector_store,
    save_vector_store,
    load_vector_store,
    vector_store_exists,
    DEFAULT_INDEX_DIR,
)
from src.retriever import retrieve_relevant_chunks, extract_unique_sources, DEFAULT_TOP_K
from src.prompt import build_prompt
from src.llm import generate_answer, is_gemini_configured, DEFAULT_MODEL, GeminiAPIKeyError


class RepoLensPipeline:
    """End-to-end RAG orchestrator for software repositories."""

    def __init__(
        self,
        index_dir: str = DEFAULT_INDEX_DIR,
        embedding_model=None,
    ):
        self.index_dir = index_dir
        self.embedding_model = embedding_model or get_embedding_model()
        self.vector_store: Optional[FAISS] = None
        self.indexed_files_count: int = 0
        self.indexed_chunks_count: int = 0

        # Attempt to load any existing persisted index on initialization
        if vector_store_exists(self.index_dir):
            loaded = load_vector_store(self.index_dir, self.embedding_model)
            if loaded is not None:
                self.vector_store = loaded

    def is_indexed(self) -> bool:
        """Return True if an active vector store is ready for queries."""
        return self.vector_store is not None

    def index_documents(
        self,
        documents: List[Document],
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Index a list of documents into FAISS.

        Args:
            documents: List of loaded Document objects.
            chunk_size: Character size per chunk.
            chunk_overlap: Overlap characters between chunks.
            persist: Whether to save the FAISS index to disk.

        Returns:
            Dictionary containing indexing statistics.
        """
        if not documents:
            raise ValueError("No valid documents provided to index.")

        # Step 1: Chunk documents with metadata preservation
        chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)
        if not chunks:
            raise ValueError("No chunks generated from documents.")

        # Step 2: Generate embeddings and build FAISS vector store
        self.vector_store = create_vector_store(chunks, self.embedding_model)

        # Step 3: Persist to disk if requested
        if persist:
            save_vector_store(self.vector_store, self.index_dir)

        # Update pipeline state
        stats = get_chunking_stats(documents, chunks)
        self.indexed_files_count = stats["files_loaded"]
        self.indexed_chunks_count = stats["chunks_created"]

        return {
            "status": "success",
            "files_indexed": self.indexed_files_count,
            "chunks_created": self.indexed_chunks_count,
            "avg_chunks_per_file": stats["avg_chunks_per_file"],
        }

    def index_from_directory(
        self,
        directory_path: str,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Load files from local directory and index them."""
        documents = load_from_directory(directory_path)
        return self.index_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist=persist,
        )

    def index_from_zip(
        self,
        zip_source,
        chunk_size: int = DEFAULT_CHUNK_SIZE,
        chunk_overlap: int = DEFAULT_CHUNK_OVERLAP,
        persist: bool = True,
    ) -> Dict[str, Any]:
        """Load files from a ZIP archive and index them."""
        documents = load_from_zip(zip_source)
        return self.index_documents(
            documents,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            persist=persist,
        )

    def query(
        self,
        question: str,
        top_k: int = DEFAULT_TOP_K,
        llm_model: str = DEFAULT_MODEL,
        custom_llm_fn=None,
    ) -> Dict[str, Any]:
        """Execute the query pipeline for a user question.

        Args:
            question: Natural language question.
            top_k: Number of relevant chunks to retrieve.
            llm_model: Gemini model identifier.
            custom_llm_fn: Optional callable for testing/mocking LLM generation.

        Returns:
            Dictionary containing:
                - 'answer': Final generated answer
                - 'sources': Deduplicated list of source files from metadata
                - 'retrieved_chunks': Detailed chunk list with scores
                - 'prompt': Full prompt text passed to the LLM
        """
        cleaned_question = question.strip()
        if not cleaned_question:
            return {
                "answer": "Please provide a non-empty question.",
                "sources": [],
                "retrieved_chunks": [],
                "prompt": "",
                "status": "empty_question",
            }

        if not self.is_indexed():
            return {
                "answer": "Please index the repository before asking questions.",
                "sources": [],
                "retrieved_chunks": [],
                "prompt": "",
                "status": "not_indexed",
            }

        # Step 1: Similarity search via FAISS
        retrieved_chunks = retrieve_relevant_chunks(
            self.vector_store,
            cleaned_question,
            top_k=top_k,
        )

        # Step 2: Extract verified sources directly from chunk metadata
        sources = extract_unique_sources(retrieved_chunks)

        # Step 3: Format grounded prompt
        prompt = build_prompt(cleaned_question, retrieved_chunks)

        # Step 4: Generate answer via Gemini (or custom callable if provided)
        try:
            if custom_llm_fn is not None:
                answer = custom_llm_fn(prompt)
            else:
                answer = generate_answer(prompt, model_name=llm_model)

            return {
                "answer": answer,
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
                "status": "success",
            }
        except GeminiAPIKeyError as key_err:
            return {
                "answer": str(key_err),
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
                "status": "missing_api_key",
            }
        except Exception as e:
            return {
                "answer": f"An error occurred while communicating with Gemini: {e}",
                "sources": sources,
                "retrieved_chunks": retrieved_chunks,
                "prompt": prompt,
                "status": "api_error",
            }
