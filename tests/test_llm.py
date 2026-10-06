"""Tests for LLM module (Phase 8)."""

import os
from unittest.mock import MagicMock, patch
import pytest

from src.llm import (
    get_gemini_api_key,
    is_gemini_configured,
    get_llm,
    generate_answer,
    GeminiAPIKeyError,
)


def test_missing_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)

    assert get_gemini_api_key() is None
    assert is_gemini_configured() is False

    with pytest.raises(GeminiAPIKeyError, match="Please add GEMINI_API_KEY to .env"):
        get_llm()

    with pytest.raises(GeminiAPIKeyError, match="Please add GEMINI_API_KEY to .env"):
        generate_answer("Hello")


def test_placeholder_api_key(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "your_key_here")
    assert get_gemini_api_key() is None
    assert is_gemini_configured() is False


def test_mocked_llm_generation(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "mock_valid_key_12345678")

    mock_response = MagicMock()
    mock_response.content = "This project implements JWT authentication in backend/auth.js."

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.return_value = mock_response

    with patch("src.llm.get_llm", return_value=mock_llm_instance):
        answer = generate_answer("How is authentication implemented?")
        assert "JWT authentication" in answer
        mock_llm_instance.invoke.assert_called_once()


def test_error_redaction(monkeypatch):
    test_key = "secret_ai_key_999"
    monkeypatch.setenv("GEMINI_API_KEY", test_key)

    mock_llm_instance = MagicMock()
    mock_llm_instance.invoke.side_effect = Exception(f"HTTP 401 Unauthorized for {test_key}")

    with patch("src.llm.get_llm", return_value=mock_llm_instance):
        with pytest.raises(RuntimeError) as exc_info:
            generate_answer("How does it work?")

        assert test_key not in str(exc_info.value)
        assert "[REDACTED_API_KEY]" in str(exc_info.value)
