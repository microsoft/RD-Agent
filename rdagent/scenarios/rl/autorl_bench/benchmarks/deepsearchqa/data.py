# benchmarks/deepsearchqa/data.py
from pathlib import Path

from datasets import load_dataset


def download_train_data(target_dir: Path):
    """下载 deepsearchqa 数据到本地"""
    target_dir.mkdir(parents=True, exist_ok=True)

    # 只下载 eval split（DeepSearchQA 只有 eval split）
    dataset = load_dataset("google/deepsearchqa", split="eval")
    dataset.save_to_disk(str(target_dir / "deepsearchqa"))
    print(f"DeepSearchQA saved to {target_dir}")
