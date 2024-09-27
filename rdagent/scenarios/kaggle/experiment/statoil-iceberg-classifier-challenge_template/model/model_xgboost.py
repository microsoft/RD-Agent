"""
motivation  of the model
"""

import numpy as np
import pandas as pd
import xgboost as xgb


def fit(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    """Define and train the model. Merge feature_select"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "binary:logistic",
        "eval_metric": "logloss",
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "nthread": -1,
    }
    num_round = 200  # Increase number of rounds

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=50)

    return bst


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred_prob = model.predict(dtest)
    return y_pred_prob.reshape(-1, 1)
