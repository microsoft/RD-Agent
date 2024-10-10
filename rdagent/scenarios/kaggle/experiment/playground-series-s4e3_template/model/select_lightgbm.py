import pandas as pd


def select(X: pd.DataFrame) -> pd.DataFrame:
    """
    Select relevant features. To be used in fit & predict function.
    """
    # For now, we assume all features are relevant. This can be expanded to feature selection logic.
    if X.columns.nlevels == 1:
        return X
    X.columns = ["_".join(str(i) for i in col).strip() for col in X.columns.values]
    return X
