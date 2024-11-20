"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model. Merge feature_select"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "multi:softmax",  # Use softmax for multi-class classification
        "num_class": len(set(y_train)),  # Number of classes
        "nthread": -1,
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "merror",
    }
    num_round = 100

    evallist = [(dtrain, "train"), (dvalid, "valid")]
    bst = xgb.train(params, dtrain, num_round, evallist, verbose_eval=10)

    return bst


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)
    return y_pred.astype(int).reshape(-1, 1)
