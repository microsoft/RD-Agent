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
        "objective": "reg:squarederror",
        "eval_metric": "rmse",
        "tree_method": "gpu_hist",  # Use GPU
        "gpu_id": 0,  # Specify the GPU device to use
        "eta": 0.1,
        "max_depth": 6,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
    }
    num_round = 200

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
    return y_pred.reshape(-1, 1)
