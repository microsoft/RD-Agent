import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error


def fit(X_train: pd.DataFrame, y_train: pd.Series, X_valid: pd.DataFrame, y_valid: pd.Series):
    """
    Define and train the Random Forest model. Merge feature selection into the pipeline.
    """
    # Initialize the Random Forest model
    model = RandomForestRegressor(n_estimators=100, random_state=32, n_jobs=-1)

    # Fit the model
    model.fit(X_train, y_train)

    # Validate the model
    y_valid_pred = model.predict(X_valid)
    mse = mean_squared_error(y_valid, y_valid_pred)
    rmse = np.sqrt(mse)
    print(f"Validation RMSE: {rmse:.4f}")

    return model


def predict(model, X):
    """
    Keep feature selection's consistency and make predictions.
    """
    # Predict using the trained model
    y_pred = model.predict(X)

    return y_pred.reshape(-1, 1)
