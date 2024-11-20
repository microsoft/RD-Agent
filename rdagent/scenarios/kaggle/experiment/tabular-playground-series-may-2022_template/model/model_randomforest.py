"""
Motivation of the model:
The Random Forest model is chosen for its robustness and ability to handle large datasets with higher dimensionality.
It reduces overfitting by averaging multiple decision trees and typically performs well out of the box, making it a good
baseline model for many classification tasks.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score


def fit(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    """
    Define and train the Random Forest model. Merge feature selection into the pipeline.
    """
    # Initialize the Random Forest model
    model = RandomForestClassifier(n_estimators=200, random_state=32, n_jobs=-1)

    # Fit the model
    model.fit(X_train, y_train)

    # Validate the model
    y_valid_pred = model.predict(X_valid)
    auroc = roc_auc_score(y_valid, y_valid_pred)
    print(f"Validation AUROC: {auroc:.4f}")

    return model


def predict(model, X):
    """
    Keep feature selection's consistency and make predictions.
    """
    # Predict using the trained model
    y_pred = model.predict(X)

    return y_pred.reshape(-1, 1)
