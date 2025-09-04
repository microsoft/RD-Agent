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

    # Load sparse features and label
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

    # Copy additional files
    for f in input_dir.glob("*"):
        if f.name not in {"X.npz", "ARF_12h.csv"} and f.is_file():
            shutil.copy(f, output_dir / f.name)
            print(f"[COPY] Extra file: {f.name}")


def copy_other_file(source: Path, target: Path):
    for item in source.iterdir():
        if item.name in {"train", "test"}:
            continue

        relative_path = item.relative_to(source)
        target_path = target / relative_path

        if item.is_dir():
            shutil.copytree(item, target_path, dirs_exist_ok=True)
            print(f"[COPY DIR] {item} -> {target_path}")
        elif item.is_file():
            target_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(item, target_path)
            print(f"[COPY FILE] {item} -> {target_path}")


def create_debug_data(
    dataset_path: str,
    output_path: str,
    min_frac: float = 0.02,
    min_num: int = 10,
):
    dataset_root = Path(dataset_path) / "arf-12-hours-prediction-task"
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
    print(dataset_root.resolve())
    print(output_root.resolve())
    copy_other_file(source=dataset_root, target=output_root)

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
