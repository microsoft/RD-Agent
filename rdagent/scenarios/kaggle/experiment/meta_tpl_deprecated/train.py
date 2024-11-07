import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script
from sklearn.metrics import accuracy_score, matthews_corrcoef

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


def compute_metrics_for_classification(y_true, y_pred):
    """Compute MCC for classification."""
    mcc = matthews_corrcoef(y_true, y_pred)
    return mcc


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1) Preprocess the data
# TODO 如果已经做过数据预处理了，不需要再做了
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

    X_train_l.append(X_train_f)
    X_valid_l.append(X_valid_f)
    X_test_l.append(X_test_f)

X_train = pd.concat(X_train_l, axis=1)
X_valid = pd.concat(X_valid_l, axis=1)
X_test = pd.concat(X_test_l, axis=1)

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
    m = import_module_from_path(f.stem, f)
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict))

# 4) Evaluate the model on the validation set
y_valid_pred_l = []
for model, predict_func in model_l:
    y_valid_pred_l.append(predict_func(model, X_valid))

# 5) Ensemble
# TODO: ensemble method in a script
# Average the predictions and apply a threshold to determine class labels
y_valid_pred = np.mean(y_valid_pred_l, axis=0)
y_valid_pred = (y_valid_pred > 0.5).astype(int)

mcc = compute_metrics_for_classification(y_valid, y_valid_pred)
print("Final on validation set: ", mcc)

# 6) Save the validation accuracy
pd.Series(data=[mcc], index=["MCC"]).to_csv("submission_score.csv")

# 7) Make predictions on the test set and save them
y_test_pred_l = []
for m, m_pred in model_l:
    y_test_pred_l.append(m_pred(m, X_test))  # TODO Make this an ensemble. Currently it uses the last prediction

y_test_pred = np.mean(y_test_pred_l, axis=0)
y_test_pred = (y_test_pred > 0.5).astype(int)

y_test_pred_labels = np.where(y_test_pred == 1, "p", "e")  # 将整数转换回 'e' 或 'p'

submission_result = pd.DataFrame({"id": passenger_ids, "class": y_test_pred_labels})

# 8) Submit predictions for the test set
submission_result.to_csv("submission.csv", index=False)
