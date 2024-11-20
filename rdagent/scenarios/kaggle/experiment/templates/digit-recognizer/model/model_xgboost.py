"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb


def fit(X_train, y_train, X_valid, y_valid):
    """Define and train the model. Merge feature_select"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "multi:softmax",
        "eval_metric": "mlogloss",
        "num_class": 10,
        "nthread": -1,
        "tree_method": "gpu_hist",
        "device": "cuda",
    }
    num_round = 100

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    model = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)

    return model


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    return model.predict(dtest).astype(int)
