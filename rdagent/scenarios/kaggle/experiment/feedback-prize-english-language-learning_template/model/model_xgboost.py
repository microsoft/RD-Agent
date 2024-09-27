"""
motivation  of the model
"""

import pandas as pd
import xgboost as xgb
from sklearn.multioutput import MultiOutputRegressor


def select(X: pd.DataFrame) -> pd.DataFrame:
    # Ignore feature selection logic
    return X


def is_sparse_df(df: pd.DataFrame) -> bool:
    # 检查 DataFrame 中的每一列是否为稀疏类型
    return any(isinstance(dtype, pd.SparseDtype) for dtype in df.dtypes)


def fit(X_train: pd.DataFrame, y_train: pd.DataFrame, X_valid: pd.DataFrame, y_valid: pd.DataFrame):
    """Define and train the model. Merge feature_select"""
    X_train = select(X_train)

    xgb_estimator = xgb.XGBRegressor(
        n_estimators=500, random_state=0, objective="reg:squarederror", tree_method="gpu_hist", device="cuda"
    )

    model = MultiOutputRegressor(xgb_estimator, n_jobs=-1)

    if is_sparse_df(X_train):
        X_train = X_train.sparse.to_coo()

    model.fit(X_train, y_train)
    return model


def predict(model, X_test):
    """
    Keep feature select's consistency.
    """
    X_test = select(X_test)
    if is_sparse_df(X_test):
        X_test = X_test.sparse.to_coo()
    y_pred = model.predict(X_test)
    return y_pred
