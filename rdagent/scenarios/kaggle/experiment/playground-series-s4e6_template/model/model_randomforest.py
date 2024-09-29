"""
Random Forest model for academic success classification.
"""

import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np

def fit(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    """
    Train the Random Forest model.
    """
    model = RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Validate the model
    y_valid_pred = model.predict(X_valid)
    accuracy = accuracy_score(y_valid, y_valid_pred)
    print(f"Validation Accuracy: {accuracy:.4f}")

    return model

def predict(model, X):
    """
    Make predictions using the trained model.
    """
    probas = model.predict_proba(X)
    return np.column_stack([proba[:, 1] for proba in probas])
