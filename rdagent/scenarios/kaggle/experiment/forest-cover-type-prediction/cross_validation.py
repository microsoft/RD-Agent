import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import KFold

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1) Preprocess the data
data_df = pd.read_csv(
    "/data/userdata/v-haoranpan/RD-Agent/git_ignore_folder/data/forest-cover-type-prediction/train.csv"
)
data_df = data_df.drop(["Id"], axis=1)

X_train = data_df.drop(["Cover_Type"], axis=1)
y_train = data_df["Cover_Type"] - 1

# Set up KFold
kf = KFold(n_splits=5, shuffle=True, random_state=SEED)

# Store results
accuracies = []

# 3) Train and evaluate using KFold
fold_number = 1
for train_index, valid_index in kf.split(X_train):
    print(f"Starting fold {fold_number}...")

    X_train_l, X_valid_l = [], []  # Reset feature lists for each fold
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]

    # Feature engineering
    for f in DIRNAME.glob("feature/feat*.py"):
        cls = import_module_from_path(f.stem, f).feature_engineering_cls()
        cls.fit(X_tr)
        X_train_f = cls.transform(X_tr)
        X_valid_f = cls.transform(X_val)

        X_train_l.append(X_train_f)
        X_valid_l.append(X_valid_f)

    X_tr = pd.concat(X_train_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_train_l))])
    X_val = pd.concat(X_valid_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_valid_l))])

    print("Shape of X_tr: ", X_tr.shape, " Shape of X_val: ", X_val.shape)

    # Replace inf and -inf with NaN
    X_tr.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_tr = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)

    # Remove duplicate columns
    X_tr = X_tr.loc[:, ~X_tr.columns.duplicated()]
    X_val = X_val.loc[:, ~X_val.columns.duplicated()]

    # Train the model
    model_l = []  # list[tuple[model, predict_func]]
    for f in DIRNAME.glob("model/model*.py"):
        m = import_module_from_path(f.stem, f)
        model_l.append((m.fit(X_tr, y_tr, X_val, y_val), m.predict))

    # Evaluate the model on the validation set
    y_valid_pred_l = []
    for model, predict_func in model_l:
        y_valid_pred = predict_func(model, X_val)
        y_valid_pred_l.append(y_valid_pred)

    # Majority vote ensemble
    y_valid_pred_ensemble = stats.mode(y_valid_pred_l, axis=0)[0].flatten()

    # Compute metrics
    accuracy = accuracy_score(y_val, y_valid_pred_ensemble)
    accuracies.append(accuracy)
    print(f"Fold {fold_number} accuracy: {accuracy}")

    fold_number += 1

# Print average accuracy
print(f"Average accuracy across folds: {np.mean(accuracies)}")
