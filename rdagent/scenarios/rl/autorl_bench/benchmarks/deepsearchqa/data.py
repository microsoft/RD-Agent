# benchmarks/deepsearchqa/data.py
import json
import shutil
from pathlib import Path

from datasets import Dataset, load_dataset

DATASET_NAME = "google/deepsearchqa"
SOURCE_SPLIT = "eval"
SPLIT_SEED = 42
TRAIN_SIZE = 100
TOTAL_SIZE = 900
EVAL_SIZE = TOTAL_SIZE - TRAIN_SIZE


def load_source_dataset() -> Dataset:
    """Load the single official DeepSearchQA split."""
    return load_dataset(DATASET_NAME, split=SOURCE_SPLIT)


def split_dataset(dataset: Dataset) -> tuple[Dataset, Dataset]:
    """Create a deterministic 100/800 train/eval split from the 900-item eval set."""
    shuffled = dataset.shuffle(seed=SPLIT_SEED)
    train = shuffled.select(range(min(TRAIN_SIZE, len(shuffled))))
    eval_set = shuffled.select(range(min(TRAIN_SIZE, len(shuffled)), len(shuffled)))
    return train, eval_set


def download_train_data(target_dir: Path):
    """Download and persist the held-in 100-sample training split for agents."""
    target_dir.mkdir(parents=True, exist_ok=True)

    dataset = load_source_dataset()
    train, eval_set = split_dataset(dataset)

    output_dir = target_dir / "deepsearchqa"
    if output_dir.exists():
        shutil.rmtree(output_dir)
    train.save_to_disk(str(output_dir))

    split_meta = {
        "dataset": DATASET_NAME,
        "source_split": SOURCE_SPLIT,
        "shuffle_seed": SPLIT_SEED,
        "train_size": len(train),
        "eval_size": len(eval_set),
        "total_size": len(dataset),
    }
    (target_dir / "split_meta.json").write_text(json.dumps(split_meta, indent=2), encoding="utf-8")
    print(f"DeepSearchQA train split saved to {output_dir} ({len(train)} train / {len(eval_set)} eval)")
