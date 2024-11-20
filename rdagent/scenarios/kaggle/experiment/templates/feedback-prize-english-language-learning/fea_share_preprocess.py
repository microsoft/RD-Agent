import os
import re

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

        return X_train, X_valid, y_train, y_valid, X_test, *others

    def data_cleaner(text):
        text = text.strip()
        text = re.sub(r"\n", "", text)
        text = text.lower()
        return text

    # train
    train = pd.read_csv("/kaggle/input/train.csv")
    test = pd.read_csv("/kaggle/input/test.csv")

    train["full_text"] = train["full_text"].apply(data_cleaner)
    test["full_text"] = test["full_text"].apply(data_cleaner)

    y_train = train[["cohesion", "syntax", "vocabulary", "phraseology", "grammar", "conventions"]]

    X_train = train[["full_text"]]
    X_test = test[["full_text"]]

    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

    return X_train, X_valid, y_train, y_valid, X_test
