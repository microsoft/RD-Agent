import pandas as pd

"""
Here is the feature engineering code for each task, with a class that has a fit and transform method.
Remember
"""


class DatetimeFeature:
    def fit(self, train_df: pd.DataFrame):
        """
        Fit the feature engineering model to the training data.
        """
        pass

    def transform(self, X: pd.DataFrame):
        """
        Transform the input data.
        """
        X["pickup_datetime"] = pd.to_datetime(X["pickup_datetime"], format="%Y-%m-%d %H:%M:%S UTC")
        X["hour"] = X.pickup_datetime.dt.hour
        X["day"] = X.pickup_datetime.dt.day
        X["month"] = X.pickup_datetime.dt.month
        X["weekday"] = X.pickup_datetime.dt.weekday
        X["year"] = X.pickup_datetime.dt.year
        X.drop(columns=["pickup_datetime"], inplace=True)
        return X


feature_engineering_cls = DatetimeFeature
