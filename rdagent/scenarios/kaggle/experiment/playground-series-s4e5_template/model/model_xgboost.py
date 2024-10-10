import pandas as pd
import xgboost as xgb


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model. Merge feature_select"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # Parameters for regression
    params = {
        "objective": "reg:squarederror",  # Use squared error for regression
        "nthread": -1,
        "n_estimators": 8000,
        "tree_method": "gpu_hist",
        "device": "cuda",
        "max_depth": 10,
        "learning_rate": 0.01,
    }
    num_round = 5000

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(params, dtrain, num_round, evallist)

    return bst


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)
    return y_pred.reshape(-1, 1)
