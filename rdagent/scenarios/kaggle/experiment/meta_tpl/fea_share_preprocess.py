import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


"""
限制llm一定输出一个函数，名字为preprocess，作用是Preprocess the data by imputing missing values and encoding categorical features.
"""
def preprocess(X: pd.DataFrame) -> pd.DataFrame:
    X_preprocessed = X.copy()
    # Identify numerical and categorical features

    return X_preprocessed