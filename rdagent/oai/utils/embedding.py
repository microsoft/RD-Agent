"""
Embedding utilities for handling token limits and text truncation with retry strategies.
"""

from typing import Optional

from litellm import get_max_tokens, token_counter

from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS

# Common embedding model token limits
EMBEDDING_MODEL_LIMITS = {
    "text-embedding-ada-002": 8191,
    "text-embedding-3-small": 8191,
    "text-embedding-3-large": 8191,
    # Add more models as needed...
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
    # Level 1: Try litellm
    try:
        max_tokens = get_max_tokens(model)
        if max_tokens and max_tokens > 0:
            return max_tokens
    except Exception as e:
        logger.warning(f"Failed to get max tokens for {model}: {e}")

    # Level 2: Query mapping table
    # Remove prefix (e.g., "provider/model" -> "model")
    model_name = model.split("/")[-1] if "/" in model else model
    if model_name in EMBEDDING_MODEL_LIMITS:
        return EMBEDDING_MODEL_LIMITS[model_name]

    # Level 3: fallback to LLM_SETTINGS.embedding_max_length
    default_max_tokens = LLM_SETTINGS.embedding_max_length
    logger.warning(f"Unknown embedding model {model}, using default max_tokens={default_max_tokens}")
    return default_max_tokens


def estimate_token_count(text: str, model: str = "text-embedding-3-small") -> int:
    """
    Calculate token count using litellm's token_counter.

    Args:
        text: Input text
        model: Model name

    Returns:
        Actual token count
    """
    try:
        return token_counter(model=model, text=text)
    except Exception as e:
        logger.warning(f"Failed to count tokens for {model}: {e}. Using fallback estimation.")
        # Simple fallback: ~0.5 tokens per character
        return len(text) // 2


def trim_text_for_embedding(
    text: str, model: str, max_tokens: Optional[int] = None, retry_count: int = 0, apply_safety_margin: bool = True
) -> str:
    """
    Truncate text for embedding model using binary search with retry support.

    Args:
        text: Input text
        model: Model name
        max_tokens: Maximum token limit, auto-detected if None
        retry_count: Current retry attempt (0 = first attempt)
        apply_safety_margin: Whether to apply safety margin (can be disabled for forced limits)

    Returns:
        Truncated text
    """
    if not text:
        return text

    # Get model's maximum token limit
    if max_tokens is None:
        base_max_tokens = get_embedding_max_tokens(model)
        # Apply retry strategy: reduce by 50% on each retry
        if retry_count > 3:
            # Ultimate fallback after 3 failed attempts
            max_tokens = 512
        elif retry_count >= 1:
            # Reduce by 50% on each retry for first 3 attempts
            # retry_count=1 -> use base_max_tokens (first actual retry)
            # retry_count=2 -> use base_max_tokens // 2
            # retry_count=3 -> use base_max_tokens // 4
            max_tokens = base_max_tokens // (2 ** (retry_count - 1))
        else:
            max_tokens = base_max_tokens

    # Apply safety margin only if requested (default) and not in ultimate fallback mode
    if apply_safety_margin and retry_count <= 3:
        safe_max_tokens = int(max_tokens * 0.9)
    else:
        # For forced limits or ultimate fallback, use the limit directly
        safe_max_tokens = max_tokens

    # Calculate current token count
    current_tokens = estimate_token_count(text, model)

    if current_tokens <= safe_max_tokens:
        return text

    logger.warning(
        f"Text too long for embedding model {model} (retry #{retry_count}): "
        f"{current_tokens} tokens > {safe_max_tokens} limit "
        f"{'(with safety margin)' if apply_safety_margin and retry_count <= 3 else '(forced limit)'}. "
        f"Using binary search to truncate."
    )

    # Binary search by words
    words = text.split()
    if not words:
        # Edge case: empty after split
        return ""

    left, right = 0, len(words)

    while left < right:
        mid = (left + right + 1) // 2
        truncated_text = " ".join(words[:mid])
        tokens = estimate_token_count(truncated_text, model)

        if tokens <= safe_max_tokens:
            left = mid
        else:
            right = mid - 1

    # Check if we found a valid word-level truncation
    if left > 0:
        result_text = " ".join(words[:left])
        final_tokens = estimate_token_count(result_text, model)

        logger.warning(
            f"Binary search truncation completed (retry #{retry_count}): "
            f"{len(words)} -> {left} words, {current_tokens} -> {final_tokens} tokens"
        )

        return result_text

    # Fallback: character-level truncation if even first word is too long
    logger.warning(f"Even first word exceeds token limit. Using character-level fallback.")

    # Binary search by characters
    left, right = 0, len(text)

    while left < right:
        mid = (left + right + 1) // 2
        truncated_text = text[:mid]
        tokens = estimate_token_count(truncated_text, model)

        if tokens <= safe_max_tokens:
            left = mid
        else:
            right = mid - 1

    result_text = text[:left]
    final_tokens = estimate_token_count(result_text, model)

    logger.warning(
        f"Character-level truncation completed (retry #{retry_count}): "
        f"{len(text)} -> {left} chars, {current_tokens} -> {final_tokens} tokens"
    )

    return result_text


def truncate_content_list_with_retry(content_list: list[str], model: str, retry_count: int = 0) -> list[str]:
    """
    Truncate a list of content strings with retry strategy.

    Args:
        content_list: List of content strings to truncate
        model: Model name
        retry_count: Current retry attempt

    Returns:
        List of truncated content strings
    """
    truncated_list = []
    for content in content_list:
        truncated_content = trim_text_for_embedding(
            content,
            model,
            max_tokens=None,  # Let the function handle retry logic internally
            retry_count=retry_count,
            apply_safety_margin=(retry_count <= 3),  # Disable safety margin for ultimate fallback
        )
        truncated_list.append(truncated_content)

    if retry_count > 0:
        logger.warning(
            f"Applied retry truncation #{retry_count} to {len(content_list)} content items for model {model}"
        )

    return truncated_list
