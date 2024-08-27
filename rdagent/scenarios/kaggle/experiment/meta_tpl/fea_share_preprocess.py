import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def preprocess(X: pd.DataFrame):
    """
    We want the X_train & X_test & X_valid to contain the same number of columns & maintain feature consistency.
    """
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

    # Fit the preprocessor on the data and transform it
    preprocessor.fit(X)  # TODO depend on its input shape
    X_array = preprocessor.transform(X).toarray()

    # Get feature names for the columns in the transformed data
    feature_names = (
        preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols).tolist()
        + numerical_cols
    )

    # Convert arrays back to DataFrames
    X_transformed = pd.DataFrame(X_array, columns=feature_names, index=X.index)

    return X_transformed
