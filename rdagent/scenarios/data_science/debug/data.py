import os
import platform
import shutil
from pathlib import Path

import pandas as pd

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING


class DataHandler:
    """Base DataHandler interface."""

    def load(self, path) -> pd.DataFrame:
        raise NotImplementedError

    def dump(self, df: pd.DataFrame, path):
        raise NotImplementedError


class GenericDataHandler(DataHandler):
    """
    A generic data handler that automatically detects file type based on suffix
    and uses the correct pandas method for load/dump.
    """

    def load(self, path) -> pd.DataFrame:
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            return pd.read_csv(path)
        elif suffix == ".pkl":
            return pd.read_pickle(path)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        elif suffix in [".h5", ".hdf", ".hdf5"]:
            # Note: for HDF, you need a 'key' in read_hdf. If you expect a single key,
            # you might do: pd.read_hdf(path, key='df') or something similar.
            # Adjust as needed based on your HDF structure.
            return pd.read_hdf(path, key="data")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def dump(self, df: pd.DataFrame, path):
        path = Path(path)
        suffix = path.suffix.lower()

        if suffix == ".csv":
            df.to_csv(path, index=False)
        elif suffix == ".pkl":
            df.to_pickle(path)
        elif suffix == ".parquet":
            df.to_parquet(path, index=True)
        elif suffix in [".h5", ".hdf", ".hdf5"]:
            # Similarly, you need a key for HDF.
            df.to_hdf(path, key="data", mode="w")
        else:
            raise ValueError(f"Unsupported file type: {suffix}")


class DataReducer:
    """Base DataReducer interface."""

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RandDataReducer(DataReducer):
    """
    Example random sampler: ensures at least `min_num` rows
    or at least `min_frac` fraction of the data (whichever is larger).
    """

    def __init__(self, min_frac=0.05, min_num=100):
        self.min_frac = min_frac
        self.min_num = min_num

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        frac = max(self.min_frac, self.min_num / len(df))
        if frac >= 1:
            return df
        return df.sample(frac=frac, random_state=1)


class ColumnReducer(DataReducer):
    """
    Example column reducer: keep only the first 5 columns.
    """

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.iloc[:, :5]


class RowReducer(DataReducer):
    """
    Example row reducer: keep only the first 10% rows.
    """

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        ten_percent = int(max(len(df) * 0.1, 100))
        return df.iloc[:ten_percent]


def create_debug_data(
    competition: str,
    dr_cls: type[DataReducer] = RandDataReducer,
    dr_cls_kwargs=None,
    dataset_path=None,
    sample_path=None,
):
    """
    Reads the original data file, creates a reduced sample,
    and renames/moves files for easier debugging.
    Automatically detects file type (csv, pkl, parquet, hdf, etc.).
    """
    if dr_cls_kwargs is None:
        dr_cls_kwargs = {}

    if dataset_path is None:
        dataset_path = KAGGLE_IMPLEMENT_SETTING.local_data_path  # FIXME: don't hardcode this KAGGLE_IMPLEMENT_SETTING

    if sample_path is None:
        # Create a sample folder under the dataset folder, which should be available in docker container
        sample_path = Path(dataset_path) / "sample"

    data_folder = Path(dataset_path) / competition
    sample_folder = Path(sample_path) / competition

    # Traverse the folder and exclude specific file types
    included_extensions = {".csv", ".pkl", ".parquet", ".h5", ".hdf", ".hdf5"}
    files_to_process = [file for file in data_folder.rglob("*") if file.is_file()]

    for file_path in files_to_process:
        sampled_file_path = sample_folder / file_path.relative_to(data_folder)
        if sampled_file_path.exists():
            continue

        sampled_file_path.parent.mkdir(parents=True, exist_ok=True)
        if file_path.suffix not in included_extensions:
            shutil.copy(file_path, sampled_file_path)
            continue

        # Initialize the generic data handler
        data_handler = GenericDataHandler()

        # Initialize the data reducer (e.g., RandDataReducer or ColumnReducer)
        data_reducer = dr_cls(**dr_cls_kwargs)

        # Load the original data
        df = data_handler.load(file_path)

        # Create a sampled subset
        df_sampled = data_reducer.reduce(df)

        # Dump the sampled data
        try:
            data_handler.dump(df_sampled, sampled_file_path)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue
