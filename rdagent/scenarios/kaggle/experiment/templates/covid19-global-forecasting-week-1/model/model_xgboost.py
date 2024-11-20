import pandas as pd
import xgboost as xgb


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model for both ConfirmedCases and Fatalities."""
    models = {}
    for target in ["ConfirmedCases", "Fatalities"]:
        dtrain = xgb.DMatrix(X_train, label=y_train[target])
        dvalid = xgb.DMatrix(X_valid, label=y_valid[target])

        params = {
            "objective": "reg:squarederror",
            "eval_metric": "rmse",
            "nthread": -1,
            "tree_method": "gpu_hist",
            "device": "cuda",
        }
        num_round = 1000

        evallist = [(dtrain, "train"), (dvalid, "eval")]
        models[target] = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=50)

    return models


def predict(models, X):
    """Make predictions for both ConfirmedCases and Fatalities."""
    dtest = xgb.DMatrix(X)
    predictions = {}
    for target, model in models.items():
        predictions[target] = model.predict(dtest)
    return pd.DataFrame(predictions)
