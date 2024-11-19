import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

def process_data(df):
    imgs = []
    for i, row in df.iterrows():
        band_1 = np.array(row['band_1']).reshape(75, 75)
        band_2 = np.array(row['band_2']).reshape(75, 75)
        band_3 = band_1 + band_2

        # Standardize each band
        a = (band_1 - band_1.mean()) / (band_1.max() - band_1.min())
        b = (band_2 - band_2.mean()) / (band_2.max() - band_2.min())
        c = (band_3 - band_3.mean()) / (band_3.max() - band_3.min())

        imgs.append(np.dstack((a, b, c)))

    return np.array(imgs)

def prepreprocess():
    # Load the data
    train = pd.read_json("/kaggle/input/train.json")
    test = pd.read_json("/kaggle/input/test.json")
    test_ids = test["id"]
    test = test.drop(columns=["id"])

    # Extract target
    y_train = train["is_iceberg"].values
    train = train.drop(columns=["is_iceberg"])

    # Process the data
    X_train = process_data(train)
    X_test = process_data(test)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.20, random_state=42)

    return X_train, X_valid, y_train, y_valid, X_test, test_ids

def preprocess_script():
    if os.path.exists("X_train.npy"):
        X_train = np.load("X_train.npy")
        X_valid = np.load("X_valid.npy")
        y_train = np.load("y_train.npy")
        y_valid = np.load("y_valid.npy")
        X_test = np.load("X_test.npy")
        test_ids = pd.read_pickle("test_ids.pkl")
        return X_train, X_valid, y_train, y_valid, X_test, test_ids

    X_train, X_valid, y_train, y_valid, X_test, test_ids = prepreprocess()

    # Save preprocessed data
    np.save("X_train.npy", X_train)
    np.save("X_valid.npy", X_valid)
    np.save("y_train.npy", y_train)
    np.save("y_valid.npy", y_valid)
    np.save("X_test.npy", X_test)
    test_ids.to_pickle("test_ids.pkl")

    return X_train, X_valid, y_train, y_valid, X_test, test_ids