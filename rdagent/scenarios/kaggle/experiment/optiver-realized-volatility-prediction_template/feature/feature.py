import pandas as pd


class MidPriceFeature:
    def fit(self, train_df: pd.DataFrame):
        return self

    def transform(self, X: pd.DataFrame):
        # Check if the required columns exist in the DataFrame
        if "bid_price1" not in X.columns or "ask_price1" not in X.columns:
            print("Warning: Required columns bid_price1 and ask_price1 are missing from the DataFrame")
            return pd.DataFrame(index=X.index)
        X["mid_price"] = (X["bid_price1"] + X["ask_price1"]) / 2
        return X[["mid_price"]]


feature_engineering_cls = MidPriceFeature
