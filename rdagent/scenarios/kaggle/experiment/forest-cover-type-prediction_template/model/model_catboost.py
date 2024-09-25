import pandas as pd
from catboost import CatBoostClassifier


def fit(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    # Define CatBoost parameters
    cat_params = {
        "iterations": 5000,
        "learning_rate": 0.03,
        "od_wait": 1000,
        "depth": 7,
        "task_type": "GPU",
        "l2_leaf_reg": 3,
        "eval_metric": "Accuracy",
        "devices": "0",
        "verbose": 1000,
    }

    # Initialize and train the CatBoost model
    model = CatBoostClassifier(**cat_params)
    model.fit(X_train, y_train, eval_set=(X_valid, y_valid))

    return model


def predict(model, X: pd.DataFrame):
    # Predict using the trained model
    y_pred = model.predict(X)
    return y_pred.reshape(-1, 1)
