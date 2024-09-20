"""
Motivation of the model:
The Random Forest model is chosen for its robustness and ability to handle large datasets with higher dimensionality.
It reduces overfitting by averaging multiple decision trees and typically performs well out of the box, making it a good
baseline model for many classification tasks.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


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
    model = RandomForestClassifier(n_estimators=100, random_state=32, n_jobs=-1)

    # Select features (if any feature selection is needed)
    X_train_selected = select(X_train)
    X_valid_selected = select(X_valid)

    # Fit the model
    model.fit(X_train_selected, y_train)

    # Validate the model
    y_valid_pred = model.predict(X_valid_selected)
    accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    return model


def predict(model, X):
    """
    Keep feature selection's consistency and make predictions.
    """
    # Select features (if any feature selection is needed)
    X_selected = select(X)

    # Predict using the trained model
    y_pred_prob = model.predict_proba(X_selected)[:, 1]

    # Apply threshold to get boolean predictions
    return y_pred_prob
