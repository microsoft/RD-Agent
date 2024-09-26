import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


def select(X: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features. To be used in fit & predict function.
    """
    # For now, we assume all features are relevant. This can be expanded to feature selection logic.
    return X


def fit(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    """
    Define and train the Random Forest model. Merge feature selection into the pipeline.
    """
    # Initialize the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=32, n_jobs=-1)

    # Select features (if any feature selection is needed)
    X_train_selected = select(X_train)

    # Fit the model
    model.fit(X_train_selected, y_train)

    return model


def predict(model, X):
    """
    Keep feature selection's consistency and make predictions.
    """
    # Select features (if any feature selection is needed)
    X_selected = select(X)

    # Predict using the trained model
    y_pred = model.predict(X_selected)

    return y_pred
