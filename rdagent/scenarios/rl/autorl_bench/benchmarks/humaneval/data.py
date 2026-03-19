"""
HumanEval data download

HumanEval official data only has test split, which is fixed at 1:1:
- The first half (82 items) is exported to train.jsonl for agent training.
- The second half (82 items) is reserved for evaluation (controlled by evaluator through test_range)
"""

import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger

_TOTAL_SAMPLES = 164
_TRAIN_SAMPLES = _TOTAL_SAMPLES // 2


def _convert_row(row: dict) -> dict:
"""Unify openai/openai_humaneval into autorl_bench common fields."""
    return {
        "question": row.get("prompt", ""),
        "answer": row.get("canonical_solution", ""),
        "task_id": row.get("task_id", ""),
        "entry_point": row.get("entry_point", ""),
        "test": row.get("test", ""),
    }


def download_train_data(target_dir: Path) -> None:
"""Download HumanEval data (visible to agent)."""
    output_file = target_dir / "train.jsonl"
    if output_file.exists():
        with open(output_file, "r", encoding="utf-8") as f:
            line_count = sum(1 for _ in f)
        if line_count == _TRAIN_SAMPLES:
            logger.info(f"HumanEval train data exists: {output_file} ({line_count} samples)")
            return
        logger.warning(f"HumanEval train data has {line_count} samples, expected {_TRAIN_SAMPLES}. Rebuilding...")

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading HumanEval split...")
    dataset = load_dataset("openai/openai_humaneval", split="test")
    train_split = dataset.select(range(_TRAIN_SAMPLES))

    with open(output_file, "w", encoding="utf-8") as f:
        for item in train_split:
            f.write(json.dumps(_convert_row(item), ensure_ascii=False) + "\n")

    logger.info(f"Saved {len(train_split)} train samples to {output_file}")
