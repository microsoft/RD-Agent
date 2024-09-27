import os

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OrdinalEncoder


def prepreprocess():
    # Load the training data
    train_df = pd.read_csv("/kaggle/input/train.csv")

    # Load book and trade data
    book_train = pd.read_parquet("/kaggle/input/book_train.parquet")
    trade_train = pd.read_parquet("/kaggle/input/trade_train.parquet")

    # Merge book and trade data with train_df
    merged_df = pd.merge(train_df, book_train, on=["stock_id", "time_id"], how="left")
    merged_df = pd.merge(merged_df, trade_train, on=["stock_id", "time_id"], how="left")

    # Split the data
    X = merged_df.drop(["target"], axis=1)
    y = merged_df["target"]

    print(X.columns.to_list())

    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.2, random_state=42)

    print(X_train.columns.to_list())

    return X_train, X_valid, y_train, y_valid


def preprocess_fit(X_train: pd.DataFrame):
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ["int64", "float64"]]
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("ordinal", OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1)),
        ]
    )

    numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
            ("cat", categorical_transformer, categorical_cols),
        ]
    )

    preprocessor.fit(X_train)

    return preprocessor, numerical_cols, categorical_cols


def preprocess_transform(X: pd.DataFrame, preprocessor, numerical_cols, categorical_cols):
    X_transformed = preprocessor.transform(X)

    X_transformed = pd.DataFrame(X_transformed, columns=numerical_cols + categorical_cols, index=X.index)

    return X_transformed


def preprocess_script():
    if os.path.exists("/kaggle/input/X_train.pkl"):
        X_train = pd.read_pickle("/kaggle/input/X_train.pkl")
        X_valid = pd.read_pickle("/kaggle/input/X_valid.pkl")
        y_train = pd.read_pickle("/kaggle/input/y_train.pkl")
        y_valid = pd.read_pickle("/kaggle/input/y_valid.pkl")
        X_test = pd.read_pickle("/kaggle/input/X_test.pkl")
        others = pd.read_pickle("/kaggle/input/others.pkl")

        return X_train, X_valid, y_train, y_valid, X_test, *others

    X_train, X_valid, y_train, y_valid = prepreprocess()

    submission_df = pd.read_csv("/kaggle/input/test.csv")

    ids = submission_df["row_id"]
    submission_df = submission_df.drop(["row_id"], axis=1)

    # Add missing columns to submission_df
    for col in X_train.columns:
        if col not in submission_df.columns:
            submission_df[col] = 0  # Fill with 0 or another appropriate value

    # Handle missing values
    for df in [X_train, X_valid, submission_df]:
        df.fillna(df.mean(), inplace=True)

    return X_train, X_valid, y_train, y_valid, submission_df, ids
