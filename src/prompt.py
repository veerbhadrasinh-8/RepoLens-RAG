"""Prompt Construction Module for RepoLens.

Responsible for:
- Formatting retrieved code chunks into structured context blocks with explicit source labels.
- Constructing the final prompt with strict anti-hallucination and grounding rules.
- Ensuring the LLM is strictly constrained to the supplied context.

Grounding & Anti-Hallucination Philosophy:
1. "Closed Domain" Answering: The LLM is explicitly instructed to act on ONLY the provided
   repository snippets, refusing to assume external code exists.
2. Explicit Source Grounding: Each snippet is prefixed by "SOURCE: <file_path>" so the LLM
   can refer to exact filenames accurately.
3. Graceful Negative Response: If the context is insufficient or silent regarding the query,
   the LLM is instructed to state that the repository does not contain enough information.
"""

from typing import List, Dict, Any

SYSTEM_INSTRUCTIONS = """You are RepoLens, an AI assistant that answers questions about a software repository.

Answer the user's question using ONLY the provided repository context.

Rules:
1. Do not invent code, files, functions, dependencies, or behavior.
2. If the answer is not present in or cannot be directly deduced from the repository context, clearly state: "The retrieved repository context does not contain enough information to answer this question."
3. Explain the answer clearly, concisely, and accurately based on the code provided.
4. When possible, mention the relevant source file(s).
5. If multiple files are involved, explain how they relate.
6. Prefer concrete evidence from the retrieved code over general assumptions."""

PROMPT_TEMPLATE = """{system_instructions}

REPOSITORY CONTEXT:
{retrieved_context}

USER QUESTION:
{question}
"""


def format_retrieved_context(retrieved_chunks: List[Dict[str, Any]]) -> str:
    """Format retrieved chunks into a standardized context block with source headers.

    Example output:
    ---
    SOURCE: backend/auth.js
    function authenticateUser(email, password) { ... }
    ---
    SOURCE: backend/middleware.js
    function verifyAuthToken(req, res, next) { ... }
    """
    if not retrieved_chunks:
        return "No relevant repository context found."

    context_blocks = []
    for i, chunk in enumerate(retrieved_chunks, start=1):
        source = chunk.get("source", "unknown")
        content = chunk.get("content", "").strip()
        block = f"[Snippet {i} | SOURCE: {source}]\n{content}"
        context_blocks.append(block)

    return "\n\n".join(context_blocks)


def build_prompt(
    question: str,
    retrieved_chunks: List[Dict[str, Any]],
    system_instructions: str = SYSTEM_INSTRUCTIONS,
) -> str:
    """Construct the complete prompt to be sent to the LLM.

    Args:
        question: The user's query.
        retrieved_chunks: List of retrieved chunk dictionaries from the retriever.
        system_instructions: System grounding rules.

    Returns:
        Formatted prompt string ready for Gemini.
    """
    cleaned_question = question.strip()
    formatted_context = format_retrieved_context(retrieved_chunks)

    return PROMPT_TEMPLATE.format(
        system_instructions=system_instructions,
        retrieved_context=formatted_context,
        question=cleaned_question,
    )
