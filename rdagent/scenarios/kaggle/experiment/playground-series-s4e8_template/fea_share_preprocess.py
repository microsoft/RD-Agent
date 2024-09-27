import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OrdinalEncoder


def prepreprocess():
    """
    This method loads the data, drops the unnecessary columns, and splits it into train and validation sets.
    """
    # Load and preprocess the data
    data_df = pd.read_csv("/kaggle/input/train.csv")
    data_df = data_df.drop(["id"], axis=1)

    X = data_df.drop(["class"], axis=1)
    y = data_df[["class"]]

    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)  # Convert class labels to numeric

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=42)

    return X_train, X_valid, y_train, y_valid


def preprocess_fit(X_train: pd.DataFrame):
    """
    Fits the preprocessor on the training data and returns the fitted preprocessor.
    """
    # Identify numerical and categorical features
    numerical_cols = [cname for cname in X_train.columns if X_train[cname].dtype in ["int64", "float64"]]
    categorical_cols = [cname for cname in X_train.columns if X_train[cname].dtype == "object"]

    # Define preprocessors for numerical and categorical features
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

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    return preprocessor, numerical_cols, categorical_cols


def preprocess_transform(X: pd.DataFrame, preprocessor, numerical_cols, categorical_cols):
    X_transformed = preprocessor.transform(X)

    # Convert arrays back to DataFrames
    X_transformed = pd.DataFrame(X_transformed, columns=numerical_cols + categorical_cols, index=X.index)

    return X_transformed


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
    X_train, X_valid, y_train, y_valid = prepreprocess()

    # Fit the preprocessor on the training data
    preprocessor, numerical_cols, categorical_cols = preprocess_fit(X_train)
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_valid = pd.Series(y_valid).reset_index(drop=True)

    # Preprocess the train, validation, and test data
    X_train = preprocess_transform(X_train, preprocessor, numerical_cols, categorical_cols)
    X_valid = preprocess_transform(X_valid, preprocessor, numerical_cols, categorical_cols)

    # Load and preprocess the test data
    submission_df = pd.read_csv("/kaggle/input/test.csv")
    ids = submission_df["id"]
    submission_df = submission_df.drop(["id"], axis=1)
    X_test = preprocess_transform(submission_df, preprocessor, numerical_cols, categorical_cols)

    return X_train, X_valid, y_train, y_valid, X_test, ids
