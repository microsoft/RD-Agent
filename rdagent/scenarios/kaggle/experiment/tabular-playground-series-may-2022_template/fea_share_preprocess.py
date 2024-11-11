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

    x = train_df.drop(columns=["target", "id", "f_27"])
    y = train_df["target"]
    scaler = MinMaxScaler()
    x_scaled = pd.DataFrame(scaler.fit_transform(x))

    X_train, X_valid, y_train, y_valid = train_test_split(x_scaled, y, test_size=0.20, random_state=101)

    # Load and preprocess the test data
    ids = test_df["id"]
    X_test = test_df.drop(["id", "f_27"], axis=1)
    X_test = pd.DataFrame(scaler.transform(X_test))

    return X_train, X_valid, y_train, y_valid, X_test, ids
