import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
from fea_share_preprocess import clean_and_impute_data, preprocess_script
from scipy import stats
from sklearn.metrics import accuracy_score, matthews_corrcoef
from sklearn.preprocessing import LabelEncoder

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
X_train, X_valid, y_train, y_valid, X_test, ids = preprocess_script()

# 2) Auto feature engineering
X_train_l, X_valid_l = [], []
X_test_l = []

for f in DIRNAME.glob("feature/feat*.py"):
    cls = import_module_from_path(f.stem, f).feature_engineering_cls()
    cls.fit(X_train)
    X_train_f = cls.transform(X_train)
    X_valid_f = cls.transform(X_valid)
    X_test_f = cls.transform(X_test)

    X_train_l.append(X_train_f)
    X_valid_l.append(X_valid_f)
    X_test_l.append(X_test_f)

X_train = pd.concat(X_train_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_train_l))])
X_valid = pd.concat(X_valid_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_valid_l))])
X_test = pd.concat(X_test_l, axis=1, keys=[f"feature_{i}" for i in range(len(X_test_l))])

print(X_train.shape, X_valid.shape, X_test.shape)

# Handle inf and -inf values
X_train, X_valid, X_test = clean_and_impute_data(X_train, X_valid, X_test)

# 3) Train the model
model_l = []  # list[tuple[model, predict_func]]
for f in DIRNAME.glob("model/model*.py"):
    m = import_module_from_path(f.stem, f)
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict))

# 4) Evaluate the model on the validation set
y_valid_pred_l = []
for model, predict_func in model_l:
    y_valid_pred = predict_func(model, X_valid)
    y_valid_pred_l.append(y_valid_pred)
    print(y_valid_pred)
    print(y_valid_pred.shape)

# 5) Ensemble
# Majority vote ensemble
y_valid_pred_ensemble = stats.mode(y_valid_pred_l, axis=0)[0].flatten()

# Compute metrics
accuracy = accuracy_score(y_valid, y_valid_pred_ensemble)
print(f"final accuracy on valid set: {accuracy}")

# 6) Save the validation metrics
pd.Series(data=[accuracy], index=["multi-class accuracy"]).to_csv("submission_score.csv")

# 7) Make predictions on the test set and save them
y_test_pred_l = []
for model, predict_func in model_l:
    y_test_pred_l.append(predict_func(model, X_test))

# For multiclass classification, use the mode of the predictions
y_test_pred = stats.mode(y_test_pred_l, axis=0)[0].flatten() + 1


submission_result = pd.DataFrame(y_test_pred, columns=["Cover_Type"])
submission_result.insert(0, "Id", ids)

submission_result.to_csv("submission.csv", index=False)
