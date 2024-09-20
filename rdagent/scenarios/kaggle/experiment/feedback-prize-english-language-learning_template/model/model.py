import pandas as pd
from sklearn.multioutput import MultiOutputRegressor
from sklearn.svm import SVR


def select(X: pd.DataFrame) -> pd.DataFrame:
    return X


def fit(X_train: pd.DataFrame, y_train: pd.Series):
    model = MultiOutputRegressor(SVR())
    model.fit(X_train, y_train)
    return model


def predict(model: MultiOutputRegressor, X_test: pd.DataFrame):
    X_test_selected = select(X_test)
    return model.predict(X_test_selected)
