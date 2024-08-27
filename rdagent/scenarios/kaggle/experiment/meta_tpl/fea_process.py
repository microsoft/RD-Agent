import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

def preprocess(X: pd.DataFrame, X_train: pd.DataFrame, X_valid: pd.DataFrame):
    # Identify numerical and categorical features
    numerical_cols = [cname for cname in X.columns if X[cname].dtype in ["int64", "float64"]]
    categorical_cols = [cname for cname in X.columns if X[cname].dtype == "object"]

    # Define preprocessors for numerical and categorical features
    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    numerical_transformer = Pipeline(steps=[("imputer", SimpleImputer(strategy="mean"))])

    # Combine preprocessing steps
    preprocessor = ColumnTransformer(
        transformers=[
            ("cat", categorical_transformer, categorical_cols),
            ("num", numerical_transformer, numerical_cols),
        ]
    )

    # Fit the preprocessor on the training data and transform both training and validation data
    preprocessor.fit(X_train)

    # Transform the data and convert to dense arrays
    X_train_array = preprocessor.transform(X_train).toarray()
    X_valid_array = preprocessor.transform(X_valid).toarray()

    # Get feature names for the columns in the transformed data
    feature_names = (
        preprocessor.named_transformers_['cat']['onehot']
        .get_feature_names_out(categorical_cols)
        .tolist() + numerical_cols
    )

    # Convert arrays back to DataFrames
    X_train = pd.DataFrame(X_train_array, columns=feature_names, index=X_train.index)

    return X_train, X_valid