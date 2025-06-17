import shutil
from pathlib import Path

import numpy as np
import pandas as pd
import sparse
from tqdm import tqdm


def sample_and_copy_subfolder(
    input_dir: Path,
    output_dir: Path,
    min_frac: float,
    min_num: int,
    seed: int = 42,
):
    np.random.seed(seed)

    feature_path = input_dir / "X.npz"
    label_path = input_dir / "ARF_12h.csv"

    X_sparse = sparse.load_npz(feature_path)
    df_label = pd.read_csv(label_path)

    N = X_sparse.shape[0]
    n_keep = max(int(N * min_frac), min_num)
    idx = np.random.choice(N, n_keep, replace=False)

    X_sample = X_sparse[idx]
    df_sample = df_label.iloc[idx].reset_index(drop=True)

    output_dir.mkdir(parents=True, exist_ok=True)
    sparse.save_npz(output_dir / "X.npz", X_sample)
    df_sample.to_csv(output_dir / "ARF_12h.csv", index=False)

    print(f"[INFO] Sampled {n_keep} of {N} from {input_dir.name}")


def create_debug_data(
    dataset_path: str,
    output_path: str,
    min_frac: float = 0.02,
    min_num: int = 10,
):
    dataset_root = Path(dataset_path) / "arf-12-hour-prediction-task"
    output_root = Path(output_path)

    for sub in ["train", "test"]:
        input_dir = dataset_root / sub
        output_dir = output_root / sub
        print(f"\n[PROCESS] {sub} subset")
        sample_and_copy_subfolder(
            input_dir=input_dir,
            output_dir=output_dir,
            min_frac=min_frac,
            min_num=min_num,
            seed=42 if sub == "train" else 123,
        )

    print(f"\n[INFO] Sampling complete → Output in: {output_root}")


if __name__ == "__main__" or globals().get("__name__") == "<run_path>":
    dataset_path = globals().get("dataset_path", "./")
    output_path = globals().get("output_path", "./sample")
    create_debug_data(
        dataset_path=dataset_path,
        output_path=output_path,
        min_frac=0.02,
        min_num=10,
    )
