import pandas as pd

"""
Here is the feature engineering code for each task, with a class that has a fit and transform method.
Remember
"""


class IdentityFeature:
    def fit(self, train_df: pd.DataFrame):
        """
        Fit the feature engineering model to the training data.
        """
        pass

    def transform(self, X: pd.DataFrame):
        """
        Transform the input data.
        """
        return X


feature_engineering_cls = IdentityFeature
