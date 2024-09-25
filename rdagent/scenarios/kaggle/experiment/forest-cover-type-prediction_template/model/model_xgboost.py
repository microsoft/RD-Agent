"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb


def select(X: pd.DataFrame) -> pd.DataFrame:
    # Ignore feature selection logic
    return X


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model. Merge feature_select"""
    X_train = select(X_train)
    X_valid = select(X_valid)
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "objective": "multi:softmax",  # Use softmax for multi-class classification
        "num_class": len(set(y_train)),  # Number of classes
        "nthread": -1,
    }
    num_round = 100

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(params, dtrain, num_round, evallist)

    return bst


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    X = select(X)
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)
    return y_pred.astype(int).reshape(-1, 1)
