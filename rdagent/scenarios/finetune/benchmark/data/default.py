from __future__ import annotations

from pathlib import Path
import json
import random
from typing import Any, Dict, List

from rdagent.log import rdagent_logger as logger


def extract_error_samples(results_base: Path, max_samples: int = 10) -> List[Dict[str, Any]]:
    """
    Default implementation to extract error samples from OpenCompass detailed results.

    When dump_details=True is set in OpenCompass config, detailed prediction results
    are saved in the results directory. This function extracts samples where the model
    answered incorrectly for feedback analysis.

    This implementation expects the OpenCompass "cascade eval" layout with:
        results_base/
          results/<...>.json      # contains `details` with llm_evaluation
          predictions/<...>.json  # contains origin_prompt / gold / prediction

    Args:
        results_base: Path to benchmark_results/{timestamp} directory
        max_samples: Maximum number of error samples to return

    Returns:
        List of error samples, each containing:
        - question: The original prompt/question
        - gold: The expected/ground truth answer
        - model_output: The model's actual output
    """
    error_samples: List[Dict[str, Any]] = []
    results_dir = results_base / "results"
    predictions_dir = results_base / "predictions"

    if not results_dir.exists():
        logger.warning(f"Results directory not found: {results_dir}")
        return error_samples

    # Iterate through all result JSON files
    for result_file in results_dir.rglob("*.json"):
        try:
            with open(result_file) as f:
                data = json.load(f)

            details = data.get("details", [])
            if not details:
                continue

            # Find corresponding predictions file
            rel_path = result_file.relative_to(results_dir)
            pred_file = predictions_dir / rel_path

            predictions: Dict[str, Any] = {}
            if pred_file.exists():
                with open(pred_file) as f:
                    predictions = json.load(f)

            # Extract error samples from OpenCompass detailed results
            # Structure: details[i].llm_evaluation[0] contains judgment info
            for idx, detail in enumerate(details):
                # Get llm_evaluation result (OpenCompass cascade eval format)
                llm_eval = detail.get("llm_evaluation", [])
                if not llm_eval:
                    continue

                eval_result = llm_eval[0]
                # Check if incorrect: prediction=="B" or llm_correct==False
                is_error = eval_result.get("prediction") == "B" or not eval_result.get("llm_correct", True)

                if is_error:
                    pred_entry = predictions.get(str(idx), {})

                    # Extract question - origin_prompt is a list of role/prompt dicts
                    origin_prompt = pred_entry.get("origin_prompt", [])
                    if isinstance(origin_prompt, list) and origin_prompt:
                        question = origin_prompt[0].get("prompt", "N/A")
                    elif isinstance(origin_prompt, str):
                        question = origin_prompt
                    else:
                        question = "N/A"

                    sample = {
                        "question": question,
                        "gold": pred_entry.get("gold", eval_result.get("gold", "N/A")),
                        "model_output": pred_entry.get("prediction", "N/A"),
                    }
                    error_samples.append(sample)

        except (json.JSONDecodeError, OSError) as e:
            logger.warning(f"Failed to parse result file {result_file}: {e}")
            continue

    # Random sampling if too many error samples
    if len(error_samples) > max_samples:
        error_samples = random.sample(error_samples, max_samples)

    logger.info(f"Extracted {len(error_samples)} error samples from benchmark results")
    return error_samples
