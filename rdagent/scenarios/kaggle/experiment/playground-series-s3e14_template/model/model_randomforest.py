import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the Random Forest model. Merge feature_select"""
    rf_params = {
        "n_estimators": 100,
        "max_depth": 10,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
        "max_features": "sqrt",
        "random_state": 2023,
        "n_jobs": -1,
        "verbose": 1,
    }
    model = RandomForestRegressor(**rf_params)
    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Keep feature select's consistency.
    """
    y_pred = model.predict(X_test)
    return y_pred.reshape(-1, 1)
