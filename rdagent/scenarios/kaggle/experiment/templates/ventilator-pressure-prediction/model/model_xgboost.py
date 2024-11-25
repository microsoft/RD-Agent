"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame) -> xgb.Booster:
    """Define and train the model. Merge feature_select"""
    # 将数据转换为 DMatrix 并指定设备
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = {
        "learning_rate": 0.1,
        "subsample": 0.95,
        "colsample_bytree": 0.11,
        "max_depth": 2,
        "booster": "gbtree",
        "reg_lambda": 66.1,
        "reg_alpha": 15.9,
        "random_state": 42,
        "tree_method": "hist",
        "device": "cuda",
        "eval_metric": "mae",
    }
    num_boost_round = 1000

    model = xgb.train(params, dtrain, num_boost_round=num_boost_round, evals=[(dvalid, "validation")], verbose_eval=100)
    return model


def predict(model: xgb.Booster, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred = model.predict(dtest)
    return y_pred
