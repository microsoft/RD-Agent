"""
Simple embedding utilities for handling token limits and text truncation.
"""

import re
from typing import List, Optional

from rdagent.log import rdagent_logger as logger


def get_embedding_max_length(model_name: str) -> int:
    """
    Get max token length for embedding model, with simple fallback.

    Args:
        model_name: Name of the embedding model

    Returns:
        Maximum token length (defaults to 8192)
    """
    try:
        from litellm import get_max_tokens

        max_tokens = get_max_tokens(model_name)
        if max_tokens and max_tokens > 0:
            return max_tokens
    except Exception as e:
        logger.warning(f"Failed to get max tokens for {model_name}: {e}")

    # Default fallback
    return 8192


def estimate_token_count(text: str) -> int:
    """
    Rough estimation of token count for text.
    Based on conservative heuristic: 1 token ≈ 2 characters (for embedding models).

    Args:
        text: Input text

    Returns:
        Estimated token count
    """
    # Conservative heuristic: ~2 characters per token for embedding models
    return len(text) // 2


def estimate_max_string_length(max_tokens: int) -> int:
    """
    Estimate maximum string length based on token limit.

    Args:
        max_tokens: Maximum token limit

    Returns:
        Estimated maximum string length
    """
    # Conservative estimation: 1.8 characters per token for embedding models
    return int(max_tokens * 1.8)


def truncate_text_gradually(text: str, max_tokens: int, steps: int = 5) -> str:
    """
    Gradually truncate text using step-wise approach.

    Args:
        text: Input text to truncate
        max_tokens: Maximum token limit
        steps: Number of truncation steps to try

    Returns:
        Truncated text that should be within token limits
    """
    if not text:
        return text

    estimated_tokens = estimate_token_count(text)
    if estimated_tokens <= max_tokens:
        return text

    # Log warning about truncation
    estimated_max_length = estimate_max_string_length(max_tokens)
    logger.warning(
        f"Text too long for embedding model: "
        f"estimated {estimated_tokens} tokens > {max_tokens} limit. "
        f"Estimated max string length: {estimated_max_length} chars. "
        f"Will truncate gradually in {steps} steps."
    )

    # Calculate truncation steps
    reduction_per_step = 0.8  # Reduce by 20% each step

    current_text = text
    for step in range(steps):
        # Try different truncation strategies
        if step == 0:
            # First try: truncate to estimated safe length
            target_length = estimate_max_string_length(max_tokens)
            current_text = text[:target_length]
        else:
            # Subsequent tries: gradually reduce further
            current_length = len(current_text)
            new_length = int(current_length * reduction_per_step)
            current_text = current_text[:new_length]

        # Check if current text is likely within limits
        estimated_tokens = estimate_token_count(current_text)
        if estimated_tokens <= max_tokens:
            if step > 0:
                logger.info(
                    f"Truncation successful: Step {step + 1} -> {len(current_text)} chars (estimated {estimated_tokens} tokens)"
                )
                logger.info(f"Truncated text start: {current_text[:50]}...")
                logger.info(f"Truncated text end: ...{current_text[-50:]}")
            return current_text

    # If we still haven't succeeded, do a final aggressive truncation
    final_length = estimate_max_string_length(max_tokens // 2)  # Very conservative
    current_text = text[:final_length]
    logger.warning(f" Final aggressive truncation: {final_length} chars (target {max_tokens//2} tokens)")
    logger.info(f" Final truncated text start: {current_text[:50]}...")
    logger.info(f" Final truncated text end: ...{current_text[-50:]}")

    return current_text


def smart_text_truncate(text: str, max_tokens: int) -> str:
    """
    Smart text truncation that tries to preserve meaning.

    Args:
        text: Input text to truncate
        max_tokens: Maximum token limit

    Returns:
        Truncated text
    """
    if not text:
        return text

    estimated_tokens = estimate_token_count(text)
    if estimated_tokens <= max_tokens:
        return text

    # Use gradual truncation
    truncated = truncate_text_gradually(text, max_tokens)

    # Try to end at a sentence boundary if possible
    sentences = re.split(r"[.!?]+", truncated)
    if len(sentences) > 1:
        # Remove the last potentially incomplete sentence
        truncated = ".".join(sentences[:-1]) + "."

    return truncated.strip()
