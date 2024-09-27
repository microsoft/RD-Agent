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
    train = pd.read_csv(
        "/kaggle/input/train.csv",
        parse_dates=["Dates"],
        index_col=False,
    )
    train = train.drop(["Descript", "Resolution", "Address"], axis=1)

    test = pd.read_csv(
        "/kaggle/input/test.csv",
        parse_dates=["Dates"],
        index_col=False,
    )
    test_ids = test["Id"]
    test = test.drop(["Address"], axis=1)

    # Feature engineering
    def feature_engineering(data):
        data["Day"] = data["Dates"].dt.day
        data["Month"] = data["Dates"].dt.month
        data["Year"] = data["Dates"].dt.year
        data["Hour"] = data["Dates"].dt.hour
        data["Minute"] = data["Dates"].dt.minute
        data["DayOfWeek"] = data["Dates"].dt.dayofweek
        data["WeekOfYear"] = data["Dates"].dt.isocalendar().week
        return data

    train = feature_engineering(train)
    test = feature_engineering(test)

    # Encoding 'PdDistrict'
    enc = LabelEncoder()
    train["PdDistrict"] = enc.fit_transform(train["PdDistrict"])
    test["PdDistrict"] = enc.transform(test["PdDistrict"])

    # Encoding 'Category' in train set
    category_encoder = LabelEncoder()
    category_encoder.fit(train["Category"])
    train["CategoryEncoded"] = category_encoder.transform(train["Category"])

    # Selecting feature columns for modeling
    x_cols = list(train.columns[2:12].values)
    x_cols.remove("Minute")  # Exclude the 'Minute' column
    X = train[x_cols]
    y = train["CategoryEncoded"]
    X_test = test[x_cols]

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42)
    print(X.shape, y.shape, X_test.shape)

    return X_train, X_valid, y_train, y_valid, X_test, category_encoder, test_ids


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
    if os.path.exists("/kaggle/input/X_train.pkl"):
        X_train = pd.read_pickle("/kaggle/input/X_train.pkl")
        X_valid = pd.read_pickle("/kaggle/input/X_valid.pkl")
        y_train = pd.read_pickle("/kaggle/input/y_train.pkl")
        y_valid = pd.read_pickle("/kaggle/input/y_valid.pkl")
        X_test = pd.read_pickle("/kaggle/input/X_test.pkl")
        others = pd.read_pickle("/kaggle/input/others.pkl")
        return X_train, X_valid, y_train, y_valid, X_test, *others

    X_train, X_valid, y_train, y_valid, test, category_encoder, test_ids = prepreprocess()

    # Fit the preprocessor on the training data
    preprocessor = preprocess_fit(X_train)

    # Preprocess the train and validation data
    X_train = preprocess_transform(X_train, preprocessor)
    X_valid = preprocess_transform(X_valid, preprocessor)

    # Preprocess the test data
    X_test = preprocess_transform(test, preprocessor)

    return X_train, X_valid, y_train, y_valid, X_test, category_encoder, test_ids
