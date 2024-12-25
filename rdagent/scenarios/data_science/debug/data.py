from pathlib import Path

import pandas as pd

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING


class DataHandler:

    def load(self, path) -> pd.DataFrame:
        ...

    def dump(self, df: pd.DataFrame, path):
        ...


class CSVDataHandler(DataHandler):

    def load(self, path) -> pd.DataFrame:
        return pd.read_csv(path)

    def dump(self, df: pd.DataFrame, path):
        df.to_csv(path, index=False)


class DataReducer:

    def reduce(self, df) -> pd.DataFrame:
        ...


class RandDataReducer(DataReducer):

    def __init__(self, min_frac=0.05, min_num=100):
        self.min_frac = min_frac
        self.min_num = min_num

    def reduce(self, df) -> pd.DataFrame:
        # Calculate the fraction to sample
        frac = max(self.min_frac, self.min_num / len(df))
        # Sample the data
        return df.sample(frac=frac, random_state=1)


def create_debug_data(
    competition,
    original_file_name,
    dh_cls: type[DataHandler],
    dr_cls: type[DataReducer],
    dr_cls_kwargs={},
    dataset_path=KAGGLE_IMPLEMENT_SETTING.local_data_path,
):
    # Define the path to the original data file
    data_path = Path(dataset_path) / competition / original_file_name

    # Automatically generate full and sampled file names based on the original file name
    original_suffix = Path(original_file_name).suffix
    full_file_name = original_file_name.replace(original_suffix, f'.full{original_suffix}')
    sampled_file_name = original_file_name.replace(original_suffix, f'.sampled{original_suffix}')

    # Define the path to the .full data file
    full_data_path = data_path.with_name(full_file_name)

    # Check if the .full file exists
    if not full_data_path.exists():
        # Initialize handlers
        data_handler = dh_cls()
        data_reducer = dr_cls(**dr_cls_kwargs)

        # Load the data file
        df = data_handler.load(data_path)

        # Reduce the data
        df_sampled = data_reducer.reduce(df)

        # Save the sampled data to a new data file
        sampled_data_path = data_path.with_name(sampled_file_name)
        data_handler.dump(df_sampled, sampled_data_path)

        # Rename the original file with .full
        data_path.rename(full_data_path)

        # Move the sampled data to replace the original one
        sampled_data_path.rename(data_path)


class PickleDataHandler(DataHandler):

    def load(self, path) -> pd.DataFrame:
        return pd.read_pickle(path)

    def dump(self, df: pd.DataFrame, path):
        df.to_pickle(path)


class ColumnReducer(DataReducer):

    def reduce(self, df) -> pd.DataFrame:
        return df.iloc[:, :5]


def new_york_city_taxi_fare_prediction_creator():
    create_debug_data(competition="new-york-city-taxi-fare-prediction",
                      original_file_name="train.csv",
                      dh_cls=CSVDataHandler,
                      dr_cls=RandDataReducer,
                      dr_cls_kwargs=dict(min_frac=0.05, min_num=100))


def amc_debug_data_creator():
    create_debug_data(
        competition="amc",
        original_file_name="train_feature_with_label.pkl",
        dh_cls=PickleDataHandler,
        dr_cls=ColumnReducer,
    )

    create_debug_data(
        competition="amc",
        original_file_name="test_feature.pkl",
        dh_cls=PickleDataHandler,
        dr_cls=ColumnReducer,
    )


# competition to data handler & Reducer mapping
# find a place to store reduced data.
# - <local_data_path>, <local_data_path>.debug

import fire
if __name__ == "__main__":
    # fire.Fire(create_debug_data)
    fire.Fire(amc_debug_data_creator)
