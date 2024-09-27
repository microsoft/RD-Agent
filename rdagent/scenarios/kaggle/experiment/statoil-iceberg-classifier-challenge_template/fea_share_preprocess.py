import os

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split


def prepreprocess():
    """
    This method loads the data, processes it, and splits it into train and validation sets.
    """
    # Load the data
    train = pd.read_json("/kaggle/input/train.json")
    train = train.drop(columns=["id"])
    test = pd.read_json("/kaggle/input/test.json")
    test_ids = test["id"]
    test = test.drop(columns=["id"])

    # Process the data
    def process_data(df):
        X = df.copy()
        X["band_1"] = X["band_1"].apply(lambda x: np.array(x).reshape(75, 75))
        X["band_2"] = X["band_2"].apply(lambda x: np.array(x).reshape(75, 75))
        X["band_3"] = (X["band_1"] + X["band_2"]) / 2

        # Extract features
        X["band_1_mean"] = X["band_1"].apply(np.mean)
        X["band_2_mean"] = X["band_2"].apply(np.mean)
        X["band_3_mean"] = X["band_3"].apply(np.mean)
        X["band_1_max"] = X["band_1"].apply(np.max)
        X["band_2_max"] = X["band_2"].apply(np.max)
        X["band_3_max"] = X["band_3"].apply(np.max)

        # Handle missing incidence angles
        X["inc_angle"] = X["inc_angle"].replace("na", np.nan).astype(float)
        X["inc_angle"].fillna(X["inc_angle"].mean(), inplace=True)

        return X

    X_train = process_data(train)
    X_test = process_data(test)

    y_train = X_train["is_iceberg"]
    X_train = X_train.drop(["is_iceberg", "band_1", "band_2", "band_3"], axis=1)
    X_test = X_test.drop(["band_1", "band_2", "band_3"], axis=1)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    return X_train, X_valid, y_train, y_valid, X_test, test_ids


def preprocess_script():
    """
    This method applies the preprocessing steps to the training, validation, and test datasets.
    """
    if os.path.exists("X_train.pkl"):
        X_train = pd.read_pickle("X_train.pkl")
        X_valid = pd.read_pickle("X_valid.pkl")
        y_train = pd.read_pickle("y_train.pkl")
        y_valid = pd.read_pickle("y_valid.pkl")
        X_test = pd.read_pickle("X_test.pkl")
        test_ids = pd.read_pickle("test_ids.pkl")
        return X_train, X_valid, y_train, y_valid, X_test, test_ids

    X_train, X_valid, y_train, y_valid, X_test, test_ids = prepreprocess()

    # Save preprocessed data
    X_train.to_pickle("X_train.pkl")
    X_valid.to_pickle("X_valid.pkl")
    y_train.to_pickle("y_train.pkl")
    y_valid.to_pickle("y_valid.pkl")
    X_test.to_pickle("X_test.pkl")
    test_ids.to_pickle("test_ids.pkl")

    return X_train, X_valid, y_train, y_valid, X_test, test_ids
