import os
import pandas as pd
import numpy as np
import scipy.sparse
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler, OneHotEncoder


def prepreprocess():
    """
    Load the data, preprocess it, and split into train and validation sets.
    """
    # Load the data
    data_df = pd.read_csv("/kaggle/input/train.csv")
    
    # Check if 'id' is actually the index
    if 'id' not in data_df.columns and data_df.index.name == 'id':
        data_df = data_df.reset_index()
    
    # Now we can safely drop the 'id' column
    data_df = data_df.drop(["id"], axis=1) 

    print(data_df.head(20))

    X = data_df.drop(["Target"], axis=1)
    y = data_df["Target"]

    # Encode target variable
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)

    # Split the data
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.20, random_state=42, stratify=y)

    return X_train, X_valid, y_train, y_valid


def preprocess_fit(X_train: pd.DataFrame):
    """
    Fit the preprocessor on the training data.
    """
    numerical_cols = X_train.select_dtypes(include=['int64', 'float64']).columns
    categorical_cols = X_train.select_dtypes(include=['object']).columns

    numerical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_cols)
        ])

    if len(categorical_cols) > 0:
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('onehot', OneHotEncoder(handle_unknown='ignore'))
        ])
        preprocessor.transformers.append(('cat', categorical_transformer, categorical_cols))

    preprocessor.fit(X_train)

    return preprocessor, numerical_cols, categorical_cols


def preprocess_transform(X: pd.DataFrame, preprocessor, numerical_cols, categorical_cols):
    X_transformed = preprocessor.transform(X)
    
    # If X_transformed is a sparse matrix, convert it to a dense array
    if scipy.sparse.issparse(X_transformed):
        X_transformed = X_transformed.toarray()
    
    # Get feature names from the preprocessor
    feature_names = preprocessor.get_feature_names_out()

    # Convert arrays back to DataFrames
    X_transformed = pd.DataFrame(X_transformed, columns=feature_names, index=X.index)

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
