"""
Motivation of the model:
The Linear Regression model is chosen for its simplicity and interpretability. It is a good starting point for regression tasks
and provides a baseline to compare more complex models against. Linear Regression assumes a linear relationship between the 
features and the target variable, which can be a reasonable assumption for many problems.
"""

import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error


def fit(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    """
    Define and train the Linear Regression model. Merge feature selection into the pipeline.
    """
    # Initialize the Linear Regression model
    model = LinearRegression()

    # Fit the model
    model.fit(X_train, y_train)

    # Validate the model
    y_valid_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_valid_pred)
    print(f"Validation Mean Squared Error: {mse:.4f}")

    return model


def predict(model, X):
    """
    Keep feature selection's consistency and make predictions.
    """
    # Predict using the trained model
    y_pred = model.predict(X)

    return y_pred.reshape(-1, 1)
