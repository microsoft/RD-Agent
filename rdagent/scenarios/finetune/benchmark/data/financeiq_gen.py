from __future__ import annotations

import json
import random
import shutil
import subprocess
from pathlib import Path
from typing import Any, Dict, List

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.finetune.datasets.financeiq.split import split_financeiq_dataset


def download_financeiq_dataset() -> None:
    """
    Download and arrange the FinanceIQ dataset for OpenCompass.

    This downloads from `Duxiaoman-DI/FinanceIQ` into:
        <FT_RD_SETTING.file_path>/benchmarks/opencompass_data/data/FinanceIQ

    The repo structure includes a `data` subdirectory; we move `dev` and `test`
    up one level to match the expected OpenCompass layout.
    """
    target_dir = FT_RD_SETTING.file_path / "benchmarks" / "opencompass_data" / "data" / "FinanceIQ"
    if target_dir.exists():
        logger.info(f"FinanceIQ dataset already exists at {target_dir}")
        return

    logger.info(f"Downloading FinanceIQ dataset to {target_dir}")
    target_dir.parent.mkdir(parents=True, exist_ok=True)

    subprocess.check_call(
        [
            "git",
            "clone",
            "https://huggingface.co/datasets/Duxiaoman-DI/FinanceIQ",
            str(target_dir),
        ]
    )

    # Move dev and test folders to upper level (opencompass_data/data/FinanceIQ)
    data_subdir = target_dir / "data"
    if data_subdir.exists():
        for folder in ("dev", "test"):
            src = data_subdir / folder
            if src.exists():
                shutil.move(str(src), str(target_dir / folder))
        shutil.rmtree(data_subdir)

    # Apply split for benchmark (keep test set only)
    split_financeiq_dataset(str(target_dir), split="test")


def extract_error_samples(results_base: Path, max_samples: int = 10) -> List[Dict[str, Any]]:
    """
    (Deprecated, processed by unified logic now)
    Extract error samples specifically for FinanceIQ_gen benchmark.

    FinanceIQ_gen result files (per subject) look like:

        {
            "accuracy": 60.0,
            "details": {
                "type": "GEN",
                "0": {
                    "prompt": [...],
                    "origin_prediction": "...",
                    "predictions": "D",
                    "references": "B"
                },
                "1": { ... },
                ...
            }
        }

    We treat a sample as error when predictions != references.
    The question text is taken from the last HUMAN prompt in the prompt list.

    Args:
        results_base: Path to benchmark_results/{timestamp} directory
        max_samples: Maximum number of error samples to return

    Returns:
        List of error samples, each containing:
        - question: The original prompt/question
        - gold: The expected/ground truth answer (references)
        - model_output: The model's actual output (predictions)
    """
    error_samples: List[Dict[str, Any]] = []
    results_dir = results_base / "results" / "ft-FinanceIQ_gen"

    if not results_dir.exists():
        logger.warning(f"FinanceIQ_gen results directory not found: {results_dir}")
        return error_samples

    # Iterate through all FinanceIQ subject JSON files
    for result_file in sorted(results_dir.glob("*.json")):
        with open(result_file) as f:
            data = json.load(f)

        details = data.get("details", {})
        if not isinstance(details, dict):
            continue

        # Each key in details except "type" is a sample index
        for key, sample in details.items():
            if key == "type" or not isinstance(sample, dict):
                continue

            pred = sample.get("predictions")
            gold = sample.get("references")

            # Skip if either is missing
            if pred is None or gold is None:
                continue

            # Only keep incorrect predictions
            if str(pred) == str(gold):
                continue

            prompt_list = sample.get("prompt", [])
            question = "N/A"
            if isinstance(prompt_list, list) and prompt_list:
                # Take the last HUMAN message as the question
                for msg in reversed(prompt_list):
                    if isinstance(msg, dict) and msg.get("role") == "HUMAN":
                        question = msg.get("prompt", "N/A")
                        break

            error_samples.append(
                {
                    "question": question,
                    "gold": str(gold),
                    "model_output": str(pred),
                }
            )

    if not error_samples:
        logger.info("No FinanceIQ_gen error samples found")
        return error_samples

    # Random sampling if too many error samples
    if len(error_samples) > max_samples:
        error_samples = random.sample(error_samples, max_samples)

    logger.info(f"Extracted {len(error_samples)} FinanceIQ_gen error samples from {results_dir}")
    return error_samples
