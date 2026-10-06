import os

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def preprocess_data(df):
    """Preprocess the data with feature engineering."""
    # Convert time to more useful features
    df["hour"] = df["time"] % 24
    df["day"] = (df["time"] // 24) % 7
    df["week"] = df["time"] // (24 * 7)

    # Create distance from center feature
    df["dist_from_center"] = np.sqrt(df["x"] ** 2 + df["y"] ** 2)

    # Create accuracy bins
    df["accuracy_bins"] = pd.cut(df["accuracy"], bins=5, labels=False)

    # Create interaction features
    df["xy"] = df["x"] * df["y"]
    df["x_accuracy"] = df["x"] * df["accuracy"]
    df["y_accuracy"] = df["y"] * df["accuracy"]

    return df


def preprocess_script():
    """Main preprocessing function."""
    if os.path.exists("/kaggle/input/X_train.pkl"):
        X_train = pd.read_pickle("/kaggle/input/X_train.pkl")
        X_valid = pd.read_pickle("/kaggle/input/X_valid.pkl")
        y_train = pd.read_pickle("/kaggle/input/y_train.pkl")
        y_valid = pd.read_pickle("/kaggle/input/y_valid.pkl")
        X_test = pd.read_pickle("/kaggle/input/X_test.pkl")
        others = pd.read_pickle("/kaggle/input/others.pkl")
        return X_train, X_valid, y_train, y_valid, X_test, *others

    # Load the training data
    train_df = pd.read_csv("/kaggle/input/train.csv").head(1000)
    test_df = pd.read_csv("/kaggle/input/test.csv").head(1000)

    # Preprocess the data
    train_df = preprocess_data(train_df)
    test_df = preprocess_data(test_df)

    # Encode place_ids
    place_id_encoder = LabelEncoder()
    place_id_encoder.fit(train_df["place_id"])
    train_df["place_id"] = place_id_encoder.transform(train_df["place_id"])

    # Split features and target for training data
    X = train_df.drop(["place_id"], axis=1)
    y = train_df["place_id"]

    # Prepare test data
    test_row_ids = test_df["row_id"]
    X_test = test_df.drop(["row_id"], axis=1)

    # Ensure X_test has the same columns as X
    for col in X.columns:
        if col not in X_test.columns:
            X_test[col] = 0  # or some other appropriate default value

    X_test = X_test[X.columns]  # Reorder columns to match X

    # Attempt stratified split, fall back to random split if necessary
    try:
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    except ValueError:
        print("Warning: Stratified split not possible. Falling back to random split.")
        X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    # Handle missing values
    imputer = SimpleImputer(strategy="mean")
    X_train = pd.DataFrame(imputer.fit_transform(X_train), columns=X_train.columns)
    X_valid = pd.DataFrame(imputer.transform(X_valid), columns=X_valid.columns)
    X_test = pd.DataFrame(imputer.transform(X_test), columns=X_test.columns)

    # Count the number of unique classes
    n_classes = len(place_id_encoder.classes_)

    return X_train, X_valid, y_train, y_valid, X_test, place_id_encoder, test_row_ids, n_classes
