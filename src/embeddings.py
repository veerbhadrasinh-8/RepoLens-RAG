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

try:
    from langchain_huggingface import HuggingFaceEmbeddings
except ImportError:
    from langchain_community.embeddings import HuggingFaceEmbeddings

DEFAULT_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Module-level cache to avoid reloading the model weights on repeated calls
_cached_embeddings = None
_active_model_desc: str = DEFAULT_MODEL_NAME


def get_hf_token() -> Optional[str]:
    """Retrieve Hugging Face token from environment variables or Streamlit secrets."""
    token = (
        os.getenv("HF_TOKEN")
        or os.getenv("HUGGINGFACE_TOKEN")
        or os.getenv("HUGGINGFACEHUB_API_TOKEN")
    )
    if not token:
        try:
            import streamlit as st
            if hasattr(st, "secrets"):
                token = (
                    st.secrets.get("HF_TOKEN")
                    or st.secrets.get("HUGGINGFACE_TOKEN")
                    or st.secrets.get("HUGGINGFACEHUB_API_TOKEN")
                )
        except Exception:
            pass
    if token:
        token = str(token).strip()
        if token and not token.startswith("your_"):
            return token
    return None


def is_model_cached(model_name: str) -> bool:
    """Check if model files are already available locally or in HuggingFace cache."""
    if os.path.isdir(model_name):
        return True
    try:
        from huggingface_hub import try_to_load_from_cache
        res = try_to_load_from_cache(model_name, "config.json")
        return isinstance(res, str) and os.path.exists(res)
    except Exception:
        return False


def get_active_embedding_desc() -> str:
    """Return description of the current embedding model."""
    global _active_model_desc
    return _active_model_desc


def get_embedding_model(model_name: str = DEFAULT_MODEL_NAME):
    """Load or retrieve the cached embedding model.

    Multi-tier resilient loading strategy:
    1. Fast Local: If model files are cached locally on disk, load offline with zero network overhead.
    2. Online HuggingFace: If not cached, attempt downloading from Hugging Face Hub (using HF_TOKEN if available).
    3. HF Mirror Fallback: If huggingface.co is unreachable, retry via the hf-mirror endpoint.
    4. Gemini Cloud Fallback: If Hugging Face is blocked/rate-limited by the host, fall back to
       Google Gemini text-embedding-004 using the configured GEMINI_API_KEY.
    """
    global _cached_embeddings, _active_model_desc
    if _cached_embeddings is not None:
        return _cached_embeddings

    hf_token = get_hf_token()
    base_kwargs = {"device": "cpu"}
    if hf_token:
        base_kwargs["token"] = hf_token

    # Tier 1: Fast local load if already cached
    if is_model_cached(model_name):
        try:
            local_kwargs = dict(base_kwargs)
            local_kwargs["local_files_only"] = True
            _cached_embeddings = HuggingFaceEmbeddings(
                model_name=model_name,
                model_kwargs=local_kwargs,
                encode_kwargs={"normalize_embeddings": True},
            )
            _active_model_desc = f"{model_name} (Local)"
            return _cached_embeddings
        except Exception:
            pass

    # Tier 2: Online Hugging Face Hub download
    try:
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=base_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        _active_model_desc = f"{model_name} (Hugging Face)"
        return _cached_embeddings
    except Exception:
        pass

    # Tier 3: Hugging Face Mirror fallback (bypasses direct huggingface.co IP restrictions)
    orig_endpoint = os.environ.get("HF_ENDPOINT")
    try:
        os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
        _cached_embeddings = HuggingFaceEmbeddings(
            model_name=model_name,
            model_kwargs=base_kwargs,
            encode_kwargs={"normalize_embeddings": True},
        )
        _active_model_desc = f"{model_name} (HF Mirror)"
        return _cached_embeddings
    except Exception:
        pass
    finally:
        if orig_endpoint is not None:
            os.environ["HF_ENDPOINT"] = orig_endpoint
        else:
            os.environ.pop("HF_ENDPOINT", None)

    # Tier 4: Google Gemini Embeddings fallback (reliable cloud API, never blocked on Streamlit Cloud)
    from src.llm import get_gemini_api_key
    gemini_key = get_gemini_api_key()
    if gemini_key:
        try:
            from langchain_google_genai import GoogleGenerativeAIEmbeddings
            _cached_embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=gemini_key,
            )
            _active_model_desc = "Gemini text-embedding-004 (Google AI Cloud)"
            return _cached_embeddings
        except Exception:
            pass

    # Tier 5: Informative actionable error
    raise RuntimeError(
        f"Could not connect to Hugging Face Hub to download '{model_name}'.\n"
        "Streamlit Cloud network restrictions or rate limits blocked the download.\n"
        "To resolve this, please enter your GEMINI_API_KEY in the sidebar (or Streamlit Secrets) "
        "to enable cloud embeddings, or provide an HF_TOKEN."
    )


def embed_documents(texts: List[str], model_name: str = DEFAULT_MODEL_NAME) -> List[List[float]]:
    """Generate dense vector embeddings for a list of text strings."""
    model = get_embedding_model(model_name)
    return model.embed_documents(texts)


def embed_query(query: str, model_name: str = DEFAULT_MODEL_NAME) -> List[float]:
    """Generate a dense vector embedding for a single user query."""
    model = get_embedding_model(model_name)
    return model.embed_query(query)
