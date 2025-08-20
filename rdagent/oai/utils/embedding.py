"""
Embedding utilities for handling token limits and text truncation.
"""

from typing import Optional

from litellm import decode, encode, get_max_tokens, token_counter

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS

# Common embedding model token limits
EMBEDDING_MODEL_LIMITS = {
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    "Qwen3-Embedding-8B": 32000,
    "Qwen3-Embedding-4B": 32000,
    "Qwen3-Embedding-0.6B": 32000,
    "bge-m3": 8191,
    "bce-embedding-base_v1": 511,
    "bge-large-zh-v1.5": 511,
    "bge-large-en-v1.5": 511,
}


def get_embedding_max_tokens(model: str) -> int:
    """
    Get maximum token limit for embedding model.

    Three-level fallback strategy:
    1. Use litellm.get_max_tokens()
    2. Query EMBEDDING_MODEL_LIMITS mapping
    3. Use default value 8192

    Args:
        model: Model name

    Returns:
        Maximum token limit
    """
    # Remove prefix (e.g., "provider/model" -> "model")
    model_name = model.split("/")[-1] if "/" in model else model

    # Level 1: Try litellm
    try:
        max_tokens = get_max_tokens(model_name)
        if max_tokens and max_tokens > 0:
            return max_tokens
    except Exception as e:
        logger.warning(f"Failed to get max tokens for {model_name}: {e}")

    # Level 2: Query mapping table
    if model_name in EMBEDDING_MODEL_LIMITS:
        return EMBEDDING_MODEL_LIMITS[model_name]

    # Level 3: fallback to LLM_SETTINGS.embedding_max_length
    default_max_tokens = LLM_SETTINGS.embedding_max_length
    logger.warning(f"Unknown embedding model {model}, using default max_tokens={default_max_tokens}")
    return default_max_tokens


def trim_text_for_embedding(text: str, model: str, max_tokens: Optional[int] = None) -> str:
    """
    Truncate text for embedding model using encode/decode approach.

    Args:
        text: Input text
        model: Model name
        max_tokens: Maximum token limit, auto-detected if None. If still exceeds limit,
                   raises error directing user to set LLM_SETTINGS.embedding_max_length

    Returns:
        Truncated text
    """
    if not text:
        return ""

    # Get model's maximum token limit
    if max_tokens is None:
        max_tokens = get_embedding_max_tokens(model)

    # Apply safety margin
    safe_max_tokens = int(max_tokens * 0.9)

    # Calculate current token count
    current_tokens = token_counter(model=model, text=text)

    if current_tokens <= safe_max_tokens:
        return text

    logger.warning(
        f"Text too long for embedding model {model}: "
        f"{current_tokens} tokens > {safe_max_tokens} limit (with safety margin). "
        f"Truncating using encode/decode approach."
    )

    try:
        # Use encode/decode approach for precise truncation
        enc_ids = encode(model=model, text=text)
        enc_ids_trunc = enc_ids[:safe_max_tokens]
        text_trunc = decode(model=model, tokens=enc_ids_trunc)
        # Ensure we return a string type (mypy type safety)
        text_trunc = str(text_trunc) if text_trunc is not None else ""

        final_tokens = token_counter(model=model, text=text_trunc)
        logger.warning(f"Truncation completed: {current_tokens} -> {final_tokens} tokens")

        return text_trunc
    except Exception as e:
        raise RuntimeError(
            f"Failed to truncate text for embedding model {model}. "
            f"Please set LLM_SETTINGS.embedding_max_length to a smaller value. "
            f"Original error: {e}"
        ) from e


def truncate_content_list(content_list: list[str], model: str) -> list[str]:
    """
    Truncate a list of content strings.

    Args:
        content_list: List of content strings to truncate
        model: Model name

    Returns:
        List of truncated content strings
    """
    truncated_list = []
    for content in content_list:
        truncated_content = trim_text_for_embedding(content, model)
        truncated_list.append(truncated_content)

    return truncated_list
