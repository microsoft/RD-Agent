import pandas as pd
import lightgbm as lgb

def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model."""
    # Create datasets for LightGBM
    dtrain = lgb.Dataset(X_train, label=y_train)
    dvalid = lgb.Dataset(X_valid, label=y_valid, reference=dtrain)

    # Parameters for regression
    params = {
        "objective": "regression",
        "metric": "rmse",
        "num_threads": -1,
        "device_type": "gpu",
    }
    num_round = 100

    # Train the model
    bst = lgb.train(params, dtrain, num_boost_round=num_round, valid_sets=[dtrain, dvalid], valid_names=["train", "eval"])

    return bst

def predict(model, X):
    """Make predictions with the trained model."""
    y_pred = model.predict(X, num_iteration=model.best_iteration)
    return y_pred.reshape(-1, 1)