"""
motivation  of the model
"""
import pandas as pd
import xgboost as xgb


def select(X):
    """
    Select relevant features. To be used in fit & predict function
    """
    return X


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model. Merge feature_select"""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    # TODO: for quick running....
    params = {
        'objective': 'binary:logistic',
        'max_depth': 8,
        'learning_rate': 0.1,
        'colsample_bytree': 0.6,
        'min_child_weight': 2,
        'subsample': 0.9,
        'ensemble_method': 'boosting'
    }
    num_round = 100

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(params, dtrain, num_round, evallist)

    return bst


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred_prob = model.predict(dtest)
    return y_pred_prob > 0.5  # Apply threshold to get boolean predictions
