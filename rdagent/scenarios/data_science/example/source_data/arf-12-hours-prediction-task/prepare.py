import random
from pathlib import Path

import numpy as np
import pandas as pd
import sparse

CURRENT_DIR = Path(__file__).resolve().parent
ROOT_DIR = CURRENT_DIR.parent.parent

raw_feature_path = CURRENT_DIR / "X.npz"
raw_label_path = CURRENT_DIR / "ARF_12h.csv"

public = ROOT_DIR / "arf-12-hours-prediction-task"
private = ROOT_DIR / "eval" / "arf-12-hours-prediction-task"

if not (public / "test").exists():
    (public / "test").mkdir(parents=True, exist_ok=True)

if not (public / "train").exists():
    (public / "train").mkdir(parents=True, exist_ok=True)

if not private.exists():
    private.mkdir(parents=True, exist_ok=True)

SEED = 42
random.seed(SEED)
np.random.seed(SEED)

X_sparse = sparse.load_npz(raw_feature_path)  # COO matrix, shape: [N, D, T]
df_label = pd.read_csv(raw_label_path)  # Contains column 'ARF_LABEL'
N = X_sparse.shape[0]

indices = np.arange(N)
np.random.shuffle(indices)
split = int(0.7 * N)
train_idx, test_idx = indices[:split], indices[split:]

X_train = X_sparse[train_idx]
X_test = X_sparse[test_idx]

df_train = df_label.iloc[train_idx].reset_index(drop=True)
df_test = df_label.iloc[test_idx].reset_index(drop=True)

submission_df = df_test.copy()
submission_df["ARF_LABEL"] = 0
submission_df.drop(submission_df.columns.difference(["ID", "ARF_LABEL"]), axis=1, inplace=True)
submission_df.to_csv(public / "sample_submission.csv", index=False)

df_test.to_csv(private / "submission_test.csv", index=False)

df_test.drop(["ARF_LABEL"], axis=1, inplace=True)
df_test.to_csv(public / "test" / "ARF_12h.csv", index=False)
sparse.save_npz(public / "test" / "X.npz", X_test)

sparse.save_npz(public / "train" / "X.npz", X_train)
df_train.to_csv(public / "train" / "ARF_12h.csv", index=False)

assert (
    X_train.shape[0] == df_train.shape[0]
), f"Mismatch: X_train rows ({X_train.shape[0]}) != df_train rows ({df_train.shape[0]})"
assert (
    X_test.shape[0] == df_test.shape[0]
), f"Mismatch: X_test rows ({X_test.shape[0]}) != df_test rows ({df_test.shape[0]})"
assert df_test.shape[1] == 2, "Public test set should have 2 columns"
assert df_train.shape[1] == 3, "Public train set should have 3 columns"
assert len(df_train) + len(df_test) == len(
    df_label
), "Length of new_train and new_test should equal length of old_train"
