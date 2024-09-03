import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def prepreprocess(self):
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

def preprocess(self, X: pd.DataFrame):
    """
    Preprocesses the given DataFrame by transforming categorical and numerical features.
    Ensures the processed data has consistent features across train, validation, and test sets.
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
    preprocessor.fit(X)
    X_array = preprocessor.transform(X).toarray()

    # Get feature names for the columns in the transformed data
    feature_names = (
        preprocessor.named_transformers_["cat"]["onehot"].get_feature_names_out(categorical_cols).tolist()
        + numerical_cols
    )

    # Convert arrays back to DataFrames
    X_transformed = pd.DataFrame(X_array, columns=feature_names, index=X.index)

    return X_transformed

def preprocess_script(self):
    """
    This method applies the preprocessing steps to the training, validation, and test datasets.
    """
    X_train, X_valid, y_train, y_valid = self.prepreprocess()

    # Preprocess the train and validation data
    X_train = self.preprocess(X_train)
    X_valid = self.preprocess(X_valid)

    # Load and preprocess the test data
    submission_df = pd.read_csv("/kaggle/input/test.csv")
    passenger_ids = submission_df["id"]
    submission_df = submission_df.drop(["id"], axis=1)
    X_test = self.preprocess(submission_df)

    return X_train, X_valid, y_train, y_valid, X_test, passenger_ids