import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script
from sklearn.metrics import accuracy_score

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent


# support various method for metrics calculation
def compute_metrics_for_classification(y_true, y_pred):
    """Compute accuracy metric for classification."""
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1) Preprocess the data
X_train, X_valid, y_train, y_valid, X_test, passenger_ids = preprocess_script()

# 2) Auto feature engineering
X_train_l, X_valid_l = [], []
X_test_l = []

for f in DIRNAME.glob("feature/feat*.py"):
    cls = import_module_from_path(f.stem, f).feature_engineering_cls()
    cls.fit(X_train)
    X_train_f = cls.transform(X_train.copy())
    X_valid_f = cls.transform(X_valid.copy())
    X_test_f = cls.transform(X_test.copy())

    if X_train_f.shape[-1] == X_valid_f.shape[-1] and X_train_f.shape[-1] == X_test_f.shape[-1]:
        X_train_l.append(X_train_f)
        X_valid_l.append(X_valid_f)
        X_test_l.append(X_test_f)

X_train = pd.concat(X_train_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_train_l))])
X_valid = pd.concat(X_valid_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_valid_l))])
X_test = pd.concat(X_test_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_test_l))])

print(X_train.shape, X_valid.shape, X_test.shape)

# Handle inf and -inf values
X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
X_valid.replace([np.inf, -np.inf], np.nan, inplace=True)
X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

from sklearn.impute import SimpleImputer

imputer = SimpleImputer(strategy="mean")

X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns)
X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

# Remove duplicate columns
X_train = X_train.loc[:, ~X_train.columns.duplicated()]
X_valid = X_valid.loc[:, ~X_valid.columns.duplicated()]
X_test = X_test.loc[:, ~X_test.columns.duplicated()]


# 3) Train the model
model_l = []  # list[tuple[model, predict_func,]]
for f in DIRNAME.glob("model/model*.py"):
    select_python_path = f.with_name(f.stem.replace("model", "select") + f.suffix)
    select_m = import_module_from_path(select_python_path.stem, select_python_path)
    X_train_selected = select_m.select(X_train.copy())
    X_valid_selected = select_m.select(X_valid.copy())

    m = import_module_from_path(f.stem, f)
    model_l.append((m.fit(X_train_selected, y_train, X_valid_selected, y_valid), m.predict, select_m))

# 4) Evaluate the model on the validation set
# metrics_all = []
# for model, predict_func, select_m in model_l:
#     X_valid_selected = select_m.select(X_valid.copy())
#     y_valid_pred = predict_func(model, X_valid_selected)
#     y_valid_pred = (y_valid_pred > 0.5).astype(int)
#     metrics = compute_metrics_for_classification(y_valid, y_valid_pred)
#     print(f"Accuracy on valid set: {metrics}")
#     metrics_all.append(metrics)

# 4) Use grid search to find the best ensemble model
valid_pred_list = []
for model, predict_func, select_m in model_l:
    X_valid_selected = select_m.select(X_valid.copy())
    y_valid_pred = predict_func(model, X_valid_selected)
    valid_pred_list.append(y_valid_pred)

metrics_all = []
weight_list = []
searched_set = set()
for i in range(1000):
    weight = np.random.randint(0, high=10, size=(len(valid_pred_list),), dtype="i")
    if str(weight.tolist()) in searched_set or weight.sum() == 0:
        continue
    weight = weight / weight.sum()
    searched_set.add(str(weight.tolist()))
    y_valid_pred = np.zeros_like(valid_pred_list[0])
    for j in range(len(valid_pred_list)):
        y_valid_pred += valid_pred_list[j] * weight[j]
    y_valid_pred = (y_valid_pred > 0.5).astype(int)
    metrics = compute_metrics_for_classification(y_valid, y_valid_pred)
    metrics_all.append(metrics)
    weight_list.append(weight)


# 5) Save the validation accuracy
max_index = np.argmax(metrics_all)
pd.Series(data=[metrics_all[max_index]], index=["MCC"]).to_csv("submission_score.csv")
print(f"Accuracy on valid set: {metrics_all[max_index]}")

# 6) Make predictions on the test set and save them
test_pred_list = []
for model, predict_func, select_m in model_l:
    X_test_selected = select_m.select(X_test.copy())
    y_test_pred = predict_func(model, X_test_selected)
    test_pred_list.append(y_test_pred)
y_test_pred = np.zeros_like(test_pred_list[0])
for j in range(len(test_pred_list)):
    y_test_pred += test_pred_list[j] * weight_list[max_index][j]
y_test_pred = (y_test_pred > 0.5).astype(bool)
y_test_pred = y_test_pred.ravel()

submission_result = pd.DataFrame({"PassengerId": passenger_ids, "Transported": y_test_pred})

# 8) Submit predictions for the test set
submission_result.to_csv("submission.csv", index=False)
