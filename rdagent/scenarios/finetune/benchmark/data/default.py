"""
Error sample extraction from OpenCompass benchmark results.

This module provides a unified approach to extract error samples from various
OpenCompass evaluator formats using both results and predictions directories.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, List

from rdagent.log import rdagent_logger as logger

# ============================================================================
# Helper Functions
# ============================================================================


def _to_bool(value: Any) -> bool:
    """
    Unified boolean conversion supporting multiple types.

    Handles: list, str, bool, None, and other types.
    Key: [False] -> False, [True] -> True
    """
    if value is None:
        return False
    if isinstance(value, list):
        return all(_to_bool(v) for v in value) if value else False
    if isinstance(value, str):
        return value.strip().upper() in ("A", "CORRECT", "TRUE", "YES", "1")
    return bool(value)


def _is_correct(sample: Dict) -> bool:
    """
    Unified correctness check - returns True if sample is correct (should be skipped).

    Checks fields in priority order from results directory.
    """
    # Direct fields
    for field in ["cascade_correct", "correct", "is_correct", "exact_match"]:
        if field in sample:
            return _to_bool(sample[field])

    # Nested llm_evaluation
    llm_eval = sample.get("llm_evaluation")
    if llm_eval and isinstance(llm_eval, list) and llm_eval:
        return _to_bool(llm_eval[0].get("llm_correct"))

    # Nested rule_evaluation
    rule_eval = sample.get("rule_evaluation")
    if rule_eval and isinstance(rule_eval, list) and rule_eval:
        return _to_bool(rule_eval[0].get("correct"))

    return False


def _format_value(value: Any) -> str:
    """Format value to string, handling list/dict/None."""
    if value is None:
        return "N/A"
    if isinstance(value, list):
        return str(value[0]) if value else "N/A"
    return str(value)


def _format_prompt(prompt: Any) -> str:
    """
    Format prompt to readable string (matches model input format).

    Handles:
    - Simple string: return as-is
    - Single message dict: extract prompt field
    - Single-turn list [{'role': 'HUMAN', 'prompt': '...'}]: return prompt directly (no prefix)
    - Multi-turn few-shot: format with ChatML-style role markers
    """
    if isinstance(prompt, str):
        return prompt
    if isinstance(prompt, dict):
        return prompt.get("prompt", str(prompt))
    if isinstance(prompt, list) and prompt:
        first = prompt[0]
        # Check if it's conversation format
        if isinstance(first, dict) and "role" in first:
            # Single-turn: return prompt directly without prefix
            if len(prompt) == 1:
                return first.get("prompt", str(first))
            # Multi-turn few-shot: format with ChatML-style markers
            parts = []
            for msg in prompt:
                if isinstance(msg, dict):
                    role = msg.get("role", "UNKNOWN")
                    content = msg.get("prompt", str(msg))
                    # Map HUMAN/BOT to user/assistant
                    role_name = "user" if role == "HUMAN" else "assistant"
                    parts.append(f"<|im_start|>{role_name}\n{content}<|im_end|>")
                else:
                    parts.append(str(msg))
            return "\n".join(parts)
        # Single item list (not conversation format)
        if isinstance(first, dict):
            return first.get("prompt", str(first))
        return str(first)
    return "N/A"


def _extract_tag_content(prompt: Any, tag_name: str) -> str:
    """
    Extract content from <tag_name Begin>...<tag_name End> in prompt.

    Used for extracting Original Question and Predicted Answer from LLM Judge prompts.
    """
    if isinstance(prompt, list):
        prompt = str(prompt)
    prompt_str = str(prompt)

    start_tag = f"<{tag_name} Begin>"
    end_tag = f"<{tag_name} End>"

    start = prompt_str.find(start_tag)
    end = prompt_str.find(end_tag)

    if start != -1 and end > start:
        content = prompt_str[start + len(start_tag) : end].strip()
        # Clean up formatting artifacts
        if content.startswith(": \\n"):
            content = content[4:]
        return content.strip()

    return "N/A"


def _get_question(sample: Dict, pred_entry: Dict) -> str:
    """Extract question - prioritize predictions for complete content."""
    # 1. Priority: predictions directory origin_prompt
    if pred_entry.get("origin_prompt"):
        return _format_prompt(pred_entry["origin_prompt"])

    # 2. Results directory direct fields
    for field in ["origin_prompt", "prompt", "source"]:
        if field in sample and sample[field]:
            return _format_prompt(sample[field])

    # 3. Nested llm_evaluation (extract from <Original Question> tag)
    llm_eval = sample.get("llm_evaluation")
    if llm_eval and isinstance(llm_eval, list) and llm_eval:
        prompt = llm_eval[0].get("origin_prompt")
        if prompt:
            content = _extract_tag_content(prompt, "Original Question")
            if content != "N/A":
                return content

    return sample.get("example_abbr", "N/A")


def _get_gold(sample: Dict, pred_entry: Dict) -> str:
    """Extract gold/reference answer - prioritize predictions."""
    # 1. Priority: predictions directory
    if pred_entry.get("gold") is not None:
        return _format_value(pred_entry["gold"])

    # 2. Results directory direct fields
    for field in ["gold", "answer", "reference", "references"]:
        if field in sample and sample[field] is not None:
            return _format_value(sample[field])

    # 3. Nested structures
    for nested in ["llm_evaluation", "rule_evaluation"]:
        eval_data = sample.get(nested)
        if eval_data and isinstance(eval_data, list) and eval_data:
            gold = eval_data[0].get("gold") or eval_data[0].get("answer")
            if gold is not None:
                return _format_value(gold)

    return "N/A"


def _get_prediction(sample: Dict, pred_entry: Dict) -> str:
    """Extract model prediction/output - prioritize predictions."""
    # 1. Priority: predictions directory
    if pred_entry.get("prediction") is not None:
        return _format_value(pred_entry["prediction"])

    # 2. Results directory direct fields (PANORAMA and similar formats)
    for field in ["pred_raw", "pred", "origin_prediction"]:
        if field in sample:
            return _format_value(sample[field])

    # 3. Nested rule_evaluation.pred (CascadeEvaluator extracted answer)
    rule_eval = sample.get("rule_evaluation")
    if rule_eval and isinstance(rule_eval, list) and rule_eval:
        pred = rule_eval[0].get("pred")
        if pred is not None:
            return _format_value(pred)

    return "N/A"


# ============================================================================
# Main Entry Point
# ============================================================================


def extract_error_samples(
    results_base: Path,
    max_samples: int = 10,
) -> List[Dict[str, Any]]:
    """
    Extract error samples from OpenCompass benchmark results.

    Uses both results and predictions directories:
    - results: correctness judgment
    - predictions: complete question/gold/prediction content

    Args:
        results_base: Path to benchmark_results/{timestamp} directory
        max_samples: Maximum number of error samples to return

    Returns:
        List of error samples, each containing:
        - question: The original prompt/question
        - gold: The expected/ground truth answer
        - model_output: The model's actual output
        - silver_answers (optional): For PANORAMA evaluator
        - custom_score (optional): For PANORAMA evaluator
    """
    errors: List[Dict[str, Any]] = []
    results_dir = results_base / "results"
    predictions_dir = results_base / "predictions"

    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return errors

    for result_file in results_dir.rglob("*.json"):
        with open(result_file) as f:
            results_data = json.load(f)

        # Load corresponding predictions file
        rel_path = result_file.relative_to(results_dir)
        pred_file = predictions_dir / rel_path
        predictions: Dict[str, Any] = {}
        if pred_file.exists():
            with open(pred_file) as f:
                predictions = json.load(f)

        details = results_data.get("details", [])
        if not details:
            continue

        # Handle both list and dict formats
        if isinstance(details, list):
            iterator = enumerate(details)
        else:
            iterator = details.items()

        for idx, sample in iterator:
            if not isinstance(sample, dict):
                continue

            # Skip correct samples (from results)
            if _is_correct(sample):
                continue

            # Get predictions entry (complete content)
            pred_entry = predictions.get(str(idx), {})

            # Build error sample with core fields
            error = {
                "question": _get_question(sample, pred_entry),
                "gold": _get_gold(sample, pred_entry),
                "model_output": _get_prediction(sample, pred_entry),
            }

            # Add PANORAMA extra fields if present
            if "silver" in sample:
                error["silver_answers"] = sample.get("silver", [])
            if "custom_score" in sample:
                error["custom_score"] = sample.get("custom_score", 0.0)

            errors.append(error)

    # Random sample if we have more than max_samples
    if len(errors) > max_samples:
        errors = random.sample(errors, max_samples)

    logger.info(f"Extracted {len(errors)} error samples from benchmark results")
    return errors
