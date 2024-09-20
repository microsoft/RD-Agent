import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


class TfidfFeature:
    def fit(self, train_df: pd.DataFrame):
        train_df = np.array(train_df).tolist()
        train_X = list(map("".join, train_df))
        self.model = TfidfVectorizer(stop_words="english", max_df=0.5, min_df=0.01).fit(train_X)
        # print(self.model.get_feature_names_out()[:5])

    def transform(self, X: pd.DataFrame):
        X = np.array(X).tolist()
        X = list(map("".join, X))
        return self.model.transform(X)
