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
    data_df = pd.read_csv("/kaggle/input/train.csv")
    data_df = data_df.drop(["PassengerId"], axis=1)

    X = data_df.drop(["Transported"], axis=1)
    y = data_df["Transported"]

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
    label_encoders = {col: LabelEncoder().fit(X_train[col]) for col in categorical_cols}

    numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numerical_transformer, numerical_cols),
        ],
        remainder="passthrough",
    )

    # Fit the preprocessor on the training data
    preprocessor.fit(X_train)

    return preprocessor, label_encoders


def preprocess_transform(X: pd.DataFrame, preprocessor, label_encoders):
    """
    Transforms the given DataFrame using the fitted preprocessor.
    Ensures the processed data has consistent features across train, validation, and test sets.
    """
    # Encode categorical features
    for col, le in label_encoders.items():
        # Handle unseen labels by setting them to a default value (e.g., -1)
        X[col] = X[col].apply(lambda x: le.transform([x])[0] if x in le.classes_ else -1)

    # Transform the data using the fitted preprocessor
    X_array = preprocessor.transform(X)

    # Convert arrays back to DataFrames
    X_transformed = pd.DataFrame(X_array, columns=X.columns, index=X.index)

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
    y_train = pd.Series(y_train).reset_index(drop=True)
    y_valid = pd.Series(y_valid).reset_index(drop=True)

    # Fit the preprocessor on the training data
    preprocessor, label_encoders = preprocess_fit(X_train)

    # Preprocess the train, validation, and test data
    X_train = preprocess_transform(X_train, preprocessor, label_encoders)
    X_valid = preprocess_transform(X_valid, preprocessor, label_encoders)

    # Load and preprocess the test data
    submission_df = pd.read_csv("/kaggle/input/test.csv")
    passenger_ids = submission_df["PassengerId"]
    submission_df = submission_df.drop(["PassengerId"], axis=1)
    X_test = preprocess_transform(submission_df, preprocessor, label_encoders)

    return X_train, X_valid, y_train, y_valid, X_test, passenger_ids
