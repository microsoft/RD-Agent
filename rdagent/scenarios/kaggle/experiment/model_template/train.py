import random

import numpy as np
import pandas as pd
import xgboost as xgb
from model import get_num_round, get_params
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_classification(y_true, y_pred):
    """Compute accuracy metric for classification."""
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def train_model(X_train, y_train, X_valid, y_valid):
    """Define and train the model."""
    dtrain = xgb.DMatrix(X_train, label=y_train)
    dvalid = xgb.DMatrix(X_valid, label=y_valid)

    params = get_params()
    num_round = get_num_round()

    evallist = [(dtrain, "train"), (dvalid, "eval")]
    bst = xgb.train(params, dtrain, num_round, evallist)

    return bst


def predict(model, X):
    dtest = xgb.DMatrix(X)
    y_pred_prob = model.predict(dtest)
    return y_pred_prob > 0.5  # Apply threshold to get boolean predictions


if __name__ == "__main__":
    # Load and preprocess the data
    data_df = pd.read_csv("/root/.data/train.csv")
    data_df = data_df.drop(["PassengerId", "Name"], axis=1)

    X = data_df.drop(["Transported"], axis=1)
    y = data_df.Transported.to_numpy()

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

    # Split the data into training and validation sets
    X_train, X_valid, y_train, y_valid = train_test_split(X, y, test_size=0.10, random_state=SEED)

    # Fit the preprocessor on the training data and transform both training and validation data
    preprocessor.fit(X_train)
    X_train = preprocessor.transform(X_train)
    X_valid = preprocessor.transform(X_valid)

    # Train the model
    model = train_model(X_train, y_train, X_valid, y_valid)

    # Evaluate the model on the validation set
    y_valid_pred = predict(model, X_valid)
    accuracy = compute_metrics_for_classification(y_valid, y_valid_pred)
    print("Final Accuracy on validation set: ", accuracy)

    # Save the validation accuracy
    pd.Series(data=[accuracy], index=["ACC"]).to_csv("./submission_score.csv")

    # Load and preprocess the test set
    submission_df = pd.read_csv("/root/.data/test.csv")
    passenger_ids = submission_df["PassengerId"]
    submission_df = submission_df.drop(["PassengerId", "Name"], axis=1)
    X_test = preprocessor.transform(submission_df)

    # Make predictions on the test set and save them
    y_test_pred = predict(model, X_test)
    submission_result = pd.DataFrame({"PassengerId": passenger_ids, "Transported": y_test_pred})
    # submit predictions for the test set
    submission_result.to_csv("./submission.csv", index=False)
