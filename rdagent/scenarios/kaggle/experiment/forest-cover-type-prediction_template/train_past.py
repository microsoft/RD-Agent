import importlib.util
import random
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler

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
print("-1")
data_df = pd.read_csv("/kaggle/input/train.csv")
data_df = data_df.drop(["Id"], axis=1)
print("0")
X_train = data_df.drop(["Cover_Type"], axis=1)
y_train = data_df["Cover_Type"] - 1
print("81")
submission_df = pd.read_csv("/kaggle/input/test.csv")
ids = submission_df["Id"]
X_test = submission_df.drop(["Id"], axis=1)


# Store results
accuracies = []
y_test_pred_l = []
scaler = StandardScaler()

print("12")
# 3) Train and evaluate using KFold
fold_number = 1
model_count = defaultdict(int)
print("123")
for train_index, valid_index in kf.split(X_train):
    print(f"Starting fold {fold_number}...")

    X_train_l, X_valid_l, X_test_l = [], [], []  # Reset feature lists for each fold
    X_tr, X_val = X_train.iloc[train_index], X_train.iloc[valid_index]
    y_tr, y_val = y_train.iloc[train_index], y_train.iloc[valid_index]
    X_te = X_test

    # Feature engineering
    for f in DIRNAME.glob("feature/feat*.py"):
        cls = import_module_from_path(f.stem, f).feature_engineering_cls()
        cls.fit(X_tr)
        X_train_f = cls.transform(X_tr)
        X_valid_f = cls.transform(X_val)
        X_test_f = cls.transform(X_te)

        X_train_l.append(X_train_f)
        X_valid_l.append(X_valid_f)
        X_test_l.append(X_test_f)

    X_tr = pd.concat(X_train_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_train_l))])
    X_val = pd.concat(X_valid_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_valid_l))])
    X_te = pd.concat(X_test_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_test_l))])

    print("Shape of X_tr: ", X_tr.shape, " Shape of X_val: ", X_val.shape, " Shape of X_te: ", X_te.shape)

    # Replace inf and -inf with NaN
    X_tr.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_te.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_tr = pd.DataFrame(imputer.fit_transform(X_tr), columns=X_tr.columns)
    X_val = pd.DataFrame(imputer.transform(X_val), columns=X_val.columns)
    X_te = pd.DataFrame(imputer.transform(X_te), columns=X_te.columns)

    # Standardize the data
    X_tr = pd.DataFrame(scaler.fit_transform(X_tr), columns=X_tr.columns)
    X_val = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
    X_te = pd.DataFrame(scaler.transform(X_te), columns=X_te.columns)

    # Remove duplicate columns
    X_tr = X_tr.loc[:, ~X_tr.columns.duplicated()]
    X_val = X_val.loc[:, ~X_val.columns.duplicated()]
    X_te = X_te.loc[:, ~X_te.columns.duplicated()]

    model_l = []  # list[tuple[model, predict_func]]
    for f in DIRNAME.glob("model/model*.py"):
        select_python_path = f.with_name(f.stem.replace("model", "select") + f.suffix)
        select_m = import_module_from_path(select_python_path.stem, select_python_path)
        X_train_selected = select_m.select(X_tr.copy())
        X_valid_selected = select_m.select(X_val.copy())

        m = import_module_from_path(f.stem, f)
        model_l.append((m.fit(X_train_selected, y_tr, X_valid_selected, y_val), m.predict))

    # 4) Evaluate the models on the validation set and choose the best one
    best_accuracy = -1
    best = None
    for model, predict_func in model_l:
        X_valid_selected = select_m.select(X_val.copy())
        y_valid_pred = predict_func(model, X_valid_selected)
        accuracy = accuracy_score(y_val, y_valid_pred)
        print(f"Accuracy on valid set: {accuracy}")

        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best = (model, predict_func)

    model_count[best] += 1
    fold_number += 1

# 5) Save the validation accuracy
final_model = max(model_count, key=model_count.get)
pd.Series(data=best_accuracy, index=["multi-class accuracy"]).to_csv("submission_score.csv")

# 6) Make predictions on the test set and save them
X_test_selected = select_m.select(X_te.copy())
y_test_pred = final_model[1](final_model[0], X_test_selected).flatten() + 1

submission_result = pd.DataFrame(y_test_pred, columns=["Cover_Type"])
submission_result.insert(0, "Id", ids)

submission_result.to_csv("submission.csv", index=False)
