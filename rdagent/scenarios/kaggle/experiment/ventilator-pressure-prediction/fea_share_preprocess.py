import os

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler


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

        return X_train, X_valid, y_train, y_valid, X_test, *others

    train_df = pd.read_csv("/kaggle/input/train.csv")
    test_df = pd.read_csv("/kaggle/input/test.csv")

    X = train_df.drop(["pressure", "id"], axis=1)
    y = train_df["pressure"]

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.3, random_state=0)

    # Load and preprocess the test data
    ids = test_df["id"]
    X_test = test_df.drop(["id"], axis=1)

    return X_train, X_valid, y_train, y_valid, X_test, ids
