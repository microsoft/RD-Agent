import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

"""
Here is the feature engineering code for each task, with a class that has a fit and transform method.
Remember
"""


class IdentityFeature:
    def fit(self, train_df: pd.DataFrame):
        """
        Fit the feature engineering model to the training data.
        """
        self.vectorizer = TfidfVectorizer()
        self.vectorizer.fit(train_df["full_text"])

    def transform(self, X: pd.DataFrame):
        """
        Transform the input data.
        """
        X = self.vectorizer.transform(X["full_text"])
        X = pd.DataFrame.sparse.from_spmatrix(X)
        return X


feature_engineering_cls = IdentityFeature
