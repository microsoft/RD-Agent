"""
AlpacaEval training data preparation

Use independent instruction data as the training set (to avoid leakage with the evaluation set).
By default, the first N samples from tatsu-lab/alpaca are used.
"""

import json
import os
from pathlib import Path

from datasets import load_dataset
from loguru import logger

DATASET_REPO = "tatsu-lab/alpaca"
TRAIN_SAMPLES = int(os.getenv("ALPACAEVAL_TRAIN_SAMPLES", "2000"))


def _convert_row(row: dict) -> dict:
    instruction = row.get("instruction", "")
    user_input = row.get("input", "")
    output = row.get("output", "")
    question = instruction if not user_input else f"{instruction}\n\n{user_input}"
    return {
        "instruction": instruction,
        "input": user_input,
        "output": output,
        "question": question,
        "answer": output,
    }


def download_train_data(target_dir: Path) -> None:
"""Download command data (visible to agent)."""
    output_file = target_dir / "train.jsonl"

    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        if line_count == TRAIN_SAMPLES:
            logger.info(f"AlpacaEval train data exists: {output_file} ({line_count} samples)")
            return
        logger.warning(f"AlpacaEval train data has {line_count} samples, expected {TRAIN_SAMPLES}. Rebuilding...")

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Downloading {DATASET_REPO} (first {TRAIN_SAMPLES} samples)...")
    dataset = load_dataset(DATASET_REPO, split=f"train[:{TRAIN_SAMPLES}]")

    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(_convert_row(item), ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(dataset)} samples to {output_file}")
