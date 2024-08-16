import random

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from model import model_cls
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

# Set random seed for reproducibility
SEED = 42
random.seed(SEED)
torch.manual_seed(SEED)
np.random.seed(SEED)


def compute_metrics_for_classification(y_true, y_pred):
    """Compute accuracy metric for classification."""
    accuracy = accuracy_score(y_true, y_pred)
    return accuracy


def train_model(X_train, y_train, X_valid, y_valid):
    """Define and train the model."""
    X_train_dense = X_train.toarray() if hasattr(X_train, "toarray") else X_train
    X_valid_dense = X_valid.toarray() if hasattr(X_valid, "toarray") else X_valid

    X_train_tensor = torch.tensor(X_train_dense, dtype=torch.float32)
    y_train_tensor = torch.tensor(y_train, dtype=torch.float32).unsqueeze(1)
    X_valid_tensor = torch.tensor(X_valid_dense, dtype=torch.float32)
    y_valid_tensor = torch.tensor(y_valid, dtype=torch.float32).unsqueeze(1)

    # Define the model
    model = model_cls(num_features=X_train.shape[1])

    # Define loss function and optimizer
    criterion = nn.BCELoss()  # Binary cross entropy loss
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Train the model
    num_epochs = 150  # Number of epochs
    for epoch in range(num_epochs):
        model.train()
        optimizer.zero_grad()
        y_train_pred = model(X_train_tensor)
        loss = criterion(y_train_pred, y_train_tensor)
        loss.backward()
        optimizer.step()

        # Evaluate model on validation set after each epoch
        model.eval()
        with torch.no_grad():
            y_valid_pred = model(X_valid_tensor)
            valid_loss = criterion(y_valid_pred, y_valid_tensor)
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {loss.item()}, Validation Loss: {valid_loss.item()}")

    return model


def predict(model, X):
    """Make predictions using the trained model."""
    X_dense = X.toarray() if hasattr(X, "toarray") else X
    X_tensor = torch.tensor(X_dense, dtype=torch.float32)
    model.eval()

    with torch.no_grad():
        y_pred = model(X_tensor)
    y_pred = y_pred.numpy().flatten()
    return y_pred > 0.5  # Apply threshold to get boolean predictions


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
        steps=[("imputer", SimpleImputer(strategy="most_frequent")), ("onehot", OneHotEncoder(handle_unknown="ignore"))]
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
    pd.Series(data=[accuracy], index=["ACC"]).to_csv("./submission.csv")

    # Load and preprocess the test set
    submission_df = pd.read_csv("/root/.data/test.csv")
    submission_df = submission_df.drop(["PassengerId", "Name"], axis=1)
    X_test = preprocessor.transform(submission_df)

    # Make predictions on the test set and save them
    y_test_pred = predict(model, X_test)
    pd.Series(y_test_pred).to_csv("./submission_update.csv", index=False)

    # submit predictions for the test set
    submission_df = pd.read_csv("/root/.data/test.csv")
    submission_df = submission_df.drop(["PassengerId", "Name"], axis=1)
    X_test = preprocessor.transform(submission_df)
    y_test_pred = predict(model, X_test)
    y_test_pred.to_csv("./submission_update.csv")
