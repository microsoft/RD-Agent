"""
GSM8K 数据下载

Agent 只能看到 train split。
评估（OpenCompass）用 test split，由 OpenCompass 自己内部加载。
"""

import json
from pathlib import Path

from datasets import load_dataset
from loguru import logger


def download_train_data(target_dir: Path) -> None:
    """下载 GSM8K 训练数据（agent 可见）"""
    output_file = target_dir / "train.jsonl"
    if output_file.exists():
        logger.info(f"GSM8K train data exists: {output_file}")
        return

    target_dir.mkdir(parents=True, exist_ok=True)
    logger.info("Downloading GSM8K train split...")
    dataset = load_dataset("openai/gsm8k", "main", split="train")
    with open(output_file, "w", encoding="utf-8") as f:
        for item in dataset:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")
    logger.info(f"Saved {len(dataset)} samples to {output_file}")
