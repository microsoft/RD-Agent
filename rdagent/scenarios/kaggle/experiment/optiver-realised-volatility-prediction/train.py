import importlib.util
import random
from pathlib import Path

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import TimeSeriesSplit
from sklearn.preprocessing import LabelEncoder

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
DIRNAME = Path(__file__).absolute().resolve().parent


def compute_rmse(y_true, y_pred):
    """Compute RMSE for regression."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    return rmse


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


print("begin preprocess")
# 1) Preprocess the data
X_train, X_valid, y_train, y_valid, X_test, ids = preprocess_script()
print("preprocess done")

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
def flatten_columns(df: pd.DataFrame) -> pd.DataFrame:
    """
    Flatten the columns of a DataFrame with MultiIndex columns,
    for (feature_0, a), (feature_0, b) -> feature_0_a, feature_0_b
    """
    if df.columns.nlevels == 1:
        return df
    df.columns = ["_".join(col).strip() for col in df.columns.values]
    return df


X_train = flatten_columns(X_train)
X_valid = flatten_columns(X_valid)
X_test = flatten_columns(X_test)

model_l = []  # list[tuple[model, predict_func,]]
for f in DIRNAME.glob("model/model*.py"):
    m = import_module_from_path(f.stem, f)
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict))

# 4) Evaluate the model on the validation set
y_valid_pred_l = []
for model, predict_func in model_l:
    y_valid_pred_l.append(predict_func(model, X_valid))
    print(predict_func(model, X_valid).shape)

# 5) Ensemble
y_valid_pred = np.mean(y_valid_pred_l, axis=0)

rmse = compute_rmse(y_valid, y_valid_pred)
print("Final RMSE on validation set: ", rmse)

# 6) Save the validation RMSE
pd.Series(data=[rmse], index=["RMSE"]).to_csv("submission_score.csv")

# 7) Make predictions on the test set and save them
y_test_pred_l = []
for m, m_pred in model_l:
    y_test_pred_l.append(m_pred(m, X_test))

y_test_pred = np.mean(y_test_pred_l, axis=0).ravel()

# 8) Submit predictions for the test set
submission_result = pd.DataFrame({"id": ids, "price": y_test_pred})
submission_result.to_csv("submission.csv", index=False)
