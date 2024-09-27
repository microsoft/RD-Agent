import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split


def prepreprocess():
    """
    This method loads the data, drops the unnecessary columns, and splits it into train and validation sets.
    """
    # Load and preprocess the data
    data_df = pd.read_csv("/kaggle/input/train.csv")
    data_df = data_df.drop(["Id"], axis=1)

    X = data_df.drop(["Cover_Type"], axis=1)
    y = data_df["Cover_Type"]

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)

    return X_train, X_valid, y_train, y_valid


def preprocess_script():
    """
    This method applies the preprocessing steps to the training, validation, and test datasets.
    """
    # Load the data
    train = pd.read_csv("/kaggle/input/train.csv")
    test = pd.read_csv("/kaggle/input/test.csv")

    # Separate features and target
    X = train.drop("label", axis=1)
    y = train["label"]

    # Normalize pixel values
    X = X / 255.0
    test = test / 255.0

    # Split into train and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    return X_train, X_valid, y_train, y_valid, test, test.index + 1


def clean_and_impute_data(X_train, X_valid, X_test):
    """
    Handles inf and -inf values by replacing them with NaN,
    then imputes missing values using the mean strategy.
    Also removes duplicate columns.
    """
    # Replace inf and -inf with NaN
    X_train.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_valid.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_test.replace([np.inf, -np.inf], np.nan, inplace=True)

    # Impute missing values
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Remove duplicate columns
    X_train = X_train.loc[:, ~X_train.columns.duplicated()]
    X_valid = X_valid.loc[:, ~X_valid.columns.duplicated()]
    X_test = X_test.loc[:, ~X_test.columns.duplicated()]

    return X_train, X_valid, X_test