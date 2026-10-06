"""Tests for Prompt module (Phase 7)."""

from src.prompt import (
    format_retrieved_context,
    build_prompt,
    SYSTEM_INSTRUCTIONS,
)


def test_format_retrieved_context_empty():
    context_str = format_retrieved_context([])
    assert "No relevant repository context found" in context_str


def test_format_retrieved_context():
    chunks = [
        {"source": "backend/auth.js", "content": "const jwt = require('jsonwebtoken');"},
        {"source": "backend/server.js", "content": "app.listen(4000);"},
    ]
    context_str = format_retrieved_context(chunks)
    assert "[Snippet 1 | SOURCE: backend/auth.js]" in context_str
    assert "const jwt = require('jsonwebtoken');" in context_str
    assert "[Snippet 2 | SOURCE: backend/server.js]" in context_str
    assert "app.listen(4000);" in context_str


def test_build_prompt():
    chunks = [
        {"source": "backend/auth.js", "content": "function login() { return token; }"},
    ]
    prompt = build_prompt("How does login work?", chunks)

    # Verify anti-hallucination instructions are included
    assert "You are RepoLens" in prompt
    assert "Do not invent code" in prompt
    assert "REPOSITORY CONTEXT:" in prompt
    assert "USER QUESTION:" in prompt
    assert "How does login work?" in prompt
    assert "backend/auth.js" in prompt
