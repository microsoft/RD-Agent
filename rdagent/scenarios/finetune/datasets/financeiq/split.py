import csv
import math
from pathlib import Path
from typing import Literal


def get_split_indices(
    total_count: int, split: Literal["train", "test"], test_limit: int = 100, test_ratio: float = 0.5
) -> slice:
    """
    Calculate the slice for train/test split.

    Rule:
    - Test set size = min(total_count * test_ratio, test_limit)
    - Test set takes from the END of the data.
    - Train set takes the rest (from the START).
    """
    test_count = min(int(math.ceil(total_count * test_ratio)), test_limit)

    if split == "test":
        return slice(total_count - test_count, total_count)
    else:
        return slice(0, total_count - test_count)


def split_financeiq_dataset(data_dir: str, split: Literal["train", "test"]) -> None:
    """
    Iterate over CSV files in the directory and apply the split in-place.
    """
    path = Path(data_dir)

    # Process CSV files
    for f in list(path.rglob("*.csv")):
        # HACK:
        # FinanceIQ specific: 'dev' folder is small and used for few-shot.
        # We preserve it for benchmarking (split='test') but remove for training (split='train') to avoid leakage.
        # Some times, the training in debug mode of llama factory will only check few samples. Which may results in failures
        rel_parts = f.relative_to(path).parts
        if "dev" in rel_parts:
            if split == "train":
                f.unlink()
            continue

        rows = []
        header = None
        # Use 'utf-8-sig' to handle potential BOM in Excel-saved CSVs, or just 'utf-8'
        # Assuming 'utf-8' for now as it's standard for HF datasets
        with open(f, "r", encoding="utf-8", newline="") as fp:
            reader = csv.reader(fp)
            try:
                header = next(reader)
                rows = list(reader)
            except StopIteration:
                # Empty file
                continue

        indices = get_split_indices(len(rows), split)
        new_rows = rows[indices]

        with open(f, "w", encoding="utf-8", newline="") as fp:
            writer = csv.writer(fp)
            if header:
                writer.writerow(header)
            writer.writerows(new_rows)
