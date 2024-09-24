import os

import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder


def prepreprocess():
    """
    This method loads the data, drops the unnecessary columns, and splits it into train and validation sets.
    """
    # Load and preprocess the data
    train = pd.read_csv("/kaggle/input/train.csv")
    # train = train.drop(["Descript", "Resolution", "Address"], axis=1)

    test = pd.read_csv("/kaggle/input/test.csv")
    test_ids = test["id"]
    # test = test.drop(["Address"], axis=1)

    # Encoding 'PdDistrict'
    categorical_cols = ["Drug", "Sex", "Ascites", "Hepatomegaly", "Spiders", "Edema"]
    encoders = {col: LabelEncoder().fit(train[col]) for col in categorical_cols}

    for col, encoder in encoders.items():
        train[col] = encoder.transform(train[col])
        test[col] = encoder.transform(test[col])

    # Encoding 'Stage' in train set
    status_encoder = LabelEncoder()
    train["StatusEncoded"] = status_encoder.fit_transform(train["Status"])

    # Selecting feature columns for modeling
    x_cols = train.columns.drop(["id", "Status", "StatusEncoded"])
    X = train[x_cols]
    y = train["StatusEncoded"]
    X_test = test.drop(["id"], axis=1)

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
    print(X.shape, y.shape, X_test.shape)

    return X_train, X_valid, y_train, y_valid, X_test, status_encoder, test_ids


def preprocess_fit(X_train: pd.DataFrame):
    """
    Fits the preprocessor on the training data and returns the fitted preprocessor.
    """
    # Identify numerical features
    numerical_cols = X_train.columns  # All columns are numerical

    # Define preprocessor for numerical features
    numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(transformers=[("num", numerical_transformer, numerical_cols)])

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    return preprocessor


def preprocess_transform(X: pd.DataFrame, preprocessor):
    """
    Transforms the given DataFrame using the fitted preprocessor.
    """
    # Transform the data using the fitted preprocessor
    X_array = preprocessor.transform(X)

    # Convert arrays back to DataFrames
    X_transformed = pd.DataFrame(X_array, columns=X.columns, index=X.index)

    return X_transformed


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
        return X_train, X_valid, y_train, y_valid, X_test

    X_train, X_valid, y_train, y_valid, test, status_encoder, test_ids = prepreprocess()

    # Fit the preprocessor on the training data
    preprocessor = preprocess_fit(X_train)

    # Preprocess the train and validation data
    X_train = preprocess_transform(X_train, preprocessor)
    X_valid = preprocess_transform(X_valid, preprocessor)

    # Preprocess the test data
    X_test = preprocess_transform(test, preprocessor)

    return X_train, X_valid, y_train, y_valid, X_test, status_encoder, test_ids
