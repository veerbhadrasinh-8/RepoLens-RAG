"""Basic sanity and environment tests for RepoLens."""

import os
import sys
import pytest


def test_python_version():
    """Verify supported Python version (3.10+)."""
    assert sys.version_info >= (3, 10), "RepoLens requires Python 3.10 or higher"


def test_core_dependencies():
    """Verify that all core libraries can be imported without error."""
    import streamlit
    import langchain
    import langchain_community
    import langchain_google_genai
    import sentence_transformers
    import faiss
    import dotenv

    assert streamlit is not None
    assert langchain is not None
    assert sentence_transformers is not None
    assert faiss is not None


def test_package_structure():
    """Verify the presence of root directories and files."""
    assert os.path.exists("requirements.txt")
    assert os.path.exists(".env.example")
    assert os.path.exists(".gitignore")
    assert os.path.exists("README.md")
    assert os.path.exists("src/__init__.py")
    assert os.path.isdir("data/faiss_index")
