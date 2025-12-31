import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder


def fit(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    """Define and train the model. Merge feature_select"""

    # Combine train and valid labels to get all unique labels
    all_labels = np.unique(np.concatenate([y_train, y_valid]))
    le = LabelEncoder().fit(all_labels)

    # Encode labels
    y_train_encoded = le.transform(y_train)
    y_valid_encoded = le.transform(y_valid)

    dtrain = xgb.DMatrix(X_train, label=y_train_encoded)
    dvalid = xgb.DMatrix(X_valid, label=y_valid_encoded)
    num_classes = len(le.classes_)

    params = {
        "objective": "multi:softprob",
        "num_class": num_classes,
        "max_depth": 6,
        "eta": 0.3,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "min_child_weight": 1,
        "nthread": -1,
    }
    num_round = 100

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(params, dtrain, num_round, evallist, early_stopping_rounds=10)

    # Store the LabelEncoder in the model for later use in prediction
    bst.le = le

    return bst


def predict(model, X):
    """
    Keep feature select's consistency.
    """
    dtest = xgb.DMatrix(X)
    y_pred_prob = model.predict(dtest)
    # Convert probabilities back to original labels if needed
    # y_pred_labels = model.le.inverse_transform(y_pred_prob.argmax(axis=1))
    return y_pred_prob
