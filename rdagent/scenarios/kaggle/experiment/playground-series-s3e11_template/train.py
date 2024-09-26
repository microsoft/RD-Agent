import importlib.util
from pathlib import Path

import numpy as np
import pandas as pd
from fea_share_preprocess import preprocess_script
from sklearn.metrics import mean_squared_error

DIRNAME = Path(__file__).absolute().resolve().parent


def import_module_from_path(module_name, module_path):
    spec = importlib.util.spec_from_file_location(module_name, module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


# 1) Preprocess the data
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

model_l = []  # list[tuple[model, predict_func]]
for f in DIRNAME.glob("model/model*.py"):
    m = import_module_from_path(f.stem, f)
    model_l.append((m.fit(X_train, y_train, X_valid, y_valid), m.predict))

# 4) Evaluate the model on the validation set
y_valid_pred_l = []
for model, predict_func in model_l:
    y_valid_pred = predict_func(model, X_valid)
    y_valid_pred_l.append(y_valid_pred)
    # print(y_valid_pred)
    # print(y_valid_pred.shape)

# 5) Ensemble
# Majority vote ensemble
y_valid_pred_ensemble = np.mean(y_valid_pred_l, axis=0)


# 6) Save the validation metrics
metrics = mean_squared_error(y_valid, y_valid_pred_ensemble, squared=False)
print(f"RMLSE on valid set: {metrics}")
pd.Series(data=[metrics], index=["RMLSE"]).to_csv("submission_score.csv")

# 7) Make predictions on the test set and save them
y_test_pred_l = []
for model, predict_func in model_l:
    y_test_pred_l.append(predict_func(model, X_test))


# For multiclass classification, use the mode of the predictions
y_test_pred = np.mean(y_test_pred_l, axis=0).ravel()


submission_result = pd.DataFrame(np.expm1(y_test_pred), columns=["cost"])
submission_result.insert(0, "id", ids)

submission_result.to_csv("submission.csv", index=False)
