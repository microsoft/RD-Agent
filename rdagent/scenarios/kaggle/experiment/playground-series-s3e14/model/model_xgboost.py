"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model. Merge feature_select"""
    xgb_params = {
        "n_estimators": 280,
        "learning_rate": 0.05,
        "max_depth": 10,
        "subsample": 1.0,
        "colsample_bytree": 1.0,
        "tree_method": "hist",
        "enable_categorical": True,
        "verbosity": 1,
        "min_child_weight": 3,
        "base_score": 4.6,
        "random_state": 2023,
    }
    model = xgb.XGBRegressor(**xgb_params)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Keep feature select's consistency.
    """
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1)
