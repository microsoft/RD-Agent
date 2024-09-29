import os

import numpy as np  # linear algebra
import pandas as pd  # data processing, CSV file I/O (e.g. pd.read_csv)
from sklearn.model_selection import train_test_split


def preprocess_script():
    """
    This method applies the preprocessing steps to the training, validation, and test datasets.
    """
    if os.path.exists("/kaggle/input/X_train.pkl"):
        X_train = pd.read_pickle("/kaggle/input/X_train.pkl")
        X_valid = pd.read_pickle("/kaggle/input/X_valid.pkl")
        y_train = pd.read_pickle("/kaggle/input/y_train.pkl")
        y_valid = pd.read_pickle("/kaggle/input/y_valid.pkl")
        X_test = pd.read_pickle("/kaggle/input/X_test.pkl")
        others = pd.read_pickle("/kaggle/input/others.pkl")
        y_train = pd.Series(y_train).reset_index(drop=True)
        y_valid = pd.Series(y_valid).reset_index(drop=True)

        return X_train, X_valid, y_train, y_valid, X_test, *others

    # train
    train = pd.read_csv("/kaggle/input/train.csv")
    train = train.drop(["id"], axis=1)
    train["store_sqft"] = train["store_sqft"].astype("category")
    train["salad"] = (train["salad_bar"] + train["prepared_food"]) / 2
    train["log_cost"] = np.log1p(train["cost"])
    most_important_features = [
        "total_children",
        "num_children_at_home",
        "avg_cars_at home(approx).1",
        "store_sqft",
        "coffee_bar",
        "video_store",
        "salad",
        "florist",
    ]

    X_train, X_valid, y_train, y_valid = train_test_split(
        train[most_important_features], train["log_cost"], test_size=0.2, random_state=2023
    )
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_valid = pd.Series(y_valid).reset_index(drop=True)

    # test
    test = pd.read_csv("/kaggle/input/test.csv")
    test["store_sqft"] = test["store_sqft"].astype("category")
    test["salad"] = (test["salad_bar"] + test["prepared_food"]) / 2

    ids = test["id"]
    X_test = test.drop(["id"], axis=1)
    X_test = X_test[most_important_features]

    return X_train, X_valid, y_train, y_valid, X_test, ids
