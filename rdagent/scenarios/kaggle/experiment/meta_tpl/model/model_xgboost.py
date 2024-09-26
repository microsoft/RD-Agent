"""
motivation  of the model
"""

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_squared_error


def select(X: pd.DataFrame) -> pd.DataFrame:
    # Ignore feature selection logic
    return X


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    X_train = select(X_train)
    X_valid = select(X_valid)

    params = {
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "nthread": -1,
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    num_round = 100

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=50, verbose_eval=100)

    y_valid_pred = bst.predict(dvalid)
    rmse = np.sqrt(mean_squared_error(y_valid, y_valid_pred))
    print(f"Validation RMSE: {rmse:.4f}")

    return bst


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    X = select(X)
    dtest = xgb.DMatrix(X)
    y_pred_prob = model.predict(dtest)
    return y_pred_prob
