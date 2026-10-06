"""LLM Module for RepoLens.

Responsible for:
- Initializing the Google Gemini Large Language Model via langchain-google-genai.
- Safe retrieval and verification of the GEMINI_API_KEY.
- Generating grounded answers using retrieved repository context.
- Graceful error handling for missing keys, network issues, or quota errors without leaking secrets.

Role of the LLM:
- Reading, understanding, and synthesizing the retrieved code chunks.
- Answering user queries clearly with code explanations and architectural relationships.
- Strictly adhering to grounding constraints (refusing to invent non-existent code).
"""

import os
from typing import Optional, List, Any
from dotenv import load_dotenv

# Ensure environment variables are loaded
load_dotenv()

DEFAULT_MODEL = "gemini-3.8-flash"
FALLBACK_MODELS = ["gemini-2.0-flash", "gemini-1.5-flash"]


class GeminiAPIKeyError(Exception):
    """Raised when the Gemini API key is missing or unconfigured."""
    pass


def get_gemini_api_key() -> Optional[str]:
    """Retrieve the Gemini API key from environment variables.

    Checks GEMINI_API_KEY first, followed by GOOGLE_API_KEY.
    Filters out empty strings or example template placeholders.
    """
    key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not key:
        return None

    cleaned_key = key.strip()
    if not cleaned_key or cleaned_key.startswith("your_") or "your_key" in cleaned_key:
        return None

    return cleaned_key


def is_gemini_configured() -> bool:
    """Return True if a valid Gemini API key format is present in the environment."""
    return get_gemini_api_key() is not None


def extract_content_text(content: Any) -> str:
    """Extract plain string text from LLM response content.

    Handles string, list of message chunk dicts, and multimodal response objects.
    """
    if isinstance(content, str):
        return content.strip()

    if isinstance(content, list):
        parts: List[str] = []
        for part in content:
            if isinstance(part, dict) and "text" in part:
                parts.append(str(part["text"]))
            elif isinstance(part, str):
                parts.append(part)
            elif hasattr(part, "text"):
                parts.append(str(getattr(part, "text")))
        return "".join(parts).strip()

    return str(content).strip()


def get_llm(
    model_name: str = DEFAULT_MODEL,
    temperature: float = 0.2,
    api_key: Optional[str] = None,
):
    """Initialize the ChatGoogleGenerativeAI instance.

    Args:
        model_name: Gemini model name (default: gemini-3.8-flash).
        temperature: Sampling temperature (0.2 ensures factual, deterministic code answers).
        api_key: Optional override for the API key.

    Returns:
        ChatGoogleGenerativeAI instance.

    Raises:
        GeminiAPIKeyError: If no valid API key is found.
    """
    active_key = api_key or get_gemini_api_key()
    if not active_key:
        raise GeminiAPIKeyError(
            "Gemini API key is not configured. Please add GEMINI_API_KEY to .env."
        )

    try:
        from langchain_google_genai import ChatGoogleGenerativeAI
        return ChatGoogleGenerativeAI(
            model=model_name,
            temperature=temperature,
            api_key=active_key,
        )
    except Exception as e:
        raise RuntimeError(f"Failed to initialize Gemini model '{model_name}': {e}")


def generate_answer(
    prompt_text: str,
    model_name: str = DEFAULT_MODEL,
    api_key: Optional[str] = None,
) -> str:
    """Send prompt to Gemini and return the natural-language answer.

    Catches and formats API errors gracefully without exposing sensitive credentials.
    Automatically tries fallback models if the requested model returns 404/NOT_FOUND.
    """
    active_key = api_key or get_gemini_api_key()
    if not active_key:
        raise GeminiAPIKeyError(
            "Gemini API key is not configured. Please add GEMINI_API_KEY to .env."
        )

    models_to_try = [model_name] + [m for m in FALLBACK_MODELS if m != model_name]
    last_error = None

    for candidate_model in models_to_try:
        try:
            llm = get_llm(model_name=candidate_model, api_key=active_key)
            response = llm.invoke(prompt_text)
            content = getattr(response, "content", response)
            return extract_content_text(content)
        except Exception as e:
            last_error = e
            err_msg = str(e).lower()
            # If model is not found, try next candidate model
            if "not found" in err_msg or "404" in err_msg or "not available" in err_msg:
                continue
            else:
                # If it's a quota or auth error, do not keep looping needlessly
                break

    clean_err = str(last_error).replace(active_key, "[REDACTED_API_KEY]")
    raise RuntimeError(f"Gemini API error: {clean_err}")
