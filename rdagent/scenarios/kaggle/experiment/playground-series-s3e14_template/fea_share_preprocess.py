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
    X_train, X_valid, y_train, y_valid = train_test_split(
        train.drop(["yield", "id"], axis=1), train["yield"], test_size=0.2, random_state=2023
    )
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_valid = pd.Series(y_valid).reset_index(drop=True)

    # test
    test = pd.read_csv("/kaggle/input/test.csv")

    ids = test["id"]
    X_test = test.drop(["id"], axis=1)

    return X_train, X_valid, y_train, y_valid, X_test, ids
