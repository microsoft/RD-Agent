import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script

DIRNAME = Path(__file__).absolute().resolve().parent


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def MCRMSE(y_true, y_pred):
    return np.mean(np.sqrt(np.mean((y_true - y_pred) ** 2, axis=0)))


# 1) Preprocess the data
X_train, X_valid, y_train, y_valid, X_test = preprocess_script()

# 2) Auto feature engineering
X_train_l, X_valid_l = [], []
X_test_l = []

for f in DIRNAME.glob("feature/feat*.py"):
    cls = import_module_from_path(f.stem, f).feature_engineering_cls()
    print(X_train.head())
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

# 3) Train the model
model_l = []  # list[tuple[model, predict_func,]]
for f in DIRNAME.glob("model/model*.py"):
    select_python_path = f.with_name(f.stem.replace("model", "select") + f.suffix)
    select_m = import_module_from_path(select_python_path.stem, select_python_path)
    X_train_selected = select_m.select(X_train.copy())
    X_valid_selected = select_m.select(X_valid.copy())

    m = import_module_from_path(f.stem, f)
    model_l.append((m.fit(X_train_selected, y_train, X_valid_selected, y_valid), m.predict))

# 4) Evaluate the model on the validation set
metrics_all = []
for model, predict_func in model_l:
    X_valid_selected = select_m.select(X_valid.copy())
    y_valid_pred = predict_func(model, X_valid_selected)
    metrics = MCRMSE(y_valid, y_valid_pred)
    print(f"MCRMSE on valid set: {metrics}")
    metrics_all.append(metrics)

# 5) Save the validation accuracy
min_index = np.argmin(metrics_all)
pd.Series(data=[metrics_all[min_index]], index=["MCRMSE"]).to_csv("submission_score.csv")

# 6) Make predictions on the test set and save them
X_test_selected = select_m.select(X_test.copy())
y_test_pred = model_l[min_index][1](model_l[min_index][0], X_test_selected)

# 7) Submit predictions for the test set
submission_result = pd.read_csv("/kaggle/input/sample_submission.csv")
submission_result["cohesion"] = y_test_pred[:, 0]
submission_result["syntax"] = y_test_pred[:, 1]
submission_result["vocabulary"] = y_test_pred[:, 2]
submission_result["phraseology"] = y_test_pred[:, 3]
submission_result["grammar"] = y_test_pred[:, 4]
submission_result["conventions"] = y_test_pred[:, 5]

submission_result.to_csv("submission.csv", index=False)
