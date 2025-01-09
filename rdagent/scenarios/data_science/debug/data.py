import os
import platform
import shutil
from pathlib import Path

import pandas as pd

try:
    import bson  # pip install pymongo
except:
    pass

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
        elif suffix == ".jsonl":
            # Read JSON Lines file
            return pd.read_json(path, lines=True)
        elif suffix == ".bson":
            data = bson.decode_file_iter(open(path, "rb"))
            df = pd.DataFrame(data)
            return df
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
        elif suffix == ".jsonl":
            # Save DataFrame to JSON Lines file
            df.to_json(path, orient="records", lines=True)
        elif suffix == ".bson":
            data = df.to_dict(orient="records")
            with open(path, "wb") as file:
                # Write each record in the list to the BSON file
                for record in data:
                    file.write(bson.BSON.encode(record))
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

    def __init__(self, min_frac=0.02, min_num=5):
        self.min_frac = min_frac
        self.min_num = min_num

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        frac = max(self.min_frac, self.min_num / len(df))
        print(f"Sampling {frac * 100:.2f}% of the data ({len(df)} rows)")
        if frac >= 1:
            return df
        return df.sample(frac=frac, random_state=1)


def count_files_in_folder(folder: Path) -> int:
    """
    Count the total number of files in a folder, including files in subfolders.
    """
    return sum(1 for _ in folder.rglob("*") if _.is_file())


def create_debug_data(
    competition: str,
    dr_cls: type[DataReducer] = RandDataReducer,
    min_frac=0.002,
    min_num=5,
    dataset_path=None,
    sample_path=None,
):
    """
    Reads the original data file, creates a reduced sample,
    and renames/moves files for easier debugging.
    Automatically detects file type (csv, pkl, parquet, hdf, etc.).
    """
    if dataset_path is None:
        dataset_path = KAGGLE_IMPLEMENT_SETTING.local_data_path  # FIXME: don't hardcode this KAGGLE_IMPLEMENT_SETTING

    if sample_path is None:
        sample_path = Path(dataset_path) / "sample"

    data_folder = Path(dataset_path) / competition
    sample_folder = Path(sample_path) / competition

    # Traverse the folder and exclude specific file types
    included_extensions = {".csv", ".pkl", ".parquet", ".h5", ".hdf", ".hdf5", ".jsonl", ".bson"}
    files_to_process = [file for file in data_folder.rglob("*") if file.is_file()]

    total_files_count = len(files_to_process)
    print(
        f"[INFO] Original dataset folder `{data_folder}` has {total_files_count} files in total (including subfolders)."
    )

    # This set will store filenames or paths that appear in the sampled data
    sample_used_file_names = set()

    # Prepare data handler and reducer
    data_handler = GenericDataHandler()
    data_reducer = dr_cls(min_frac=min_frac, min_num=min_num)

    for file_path in files_to_process:
        sampled_file_path = sample_folder / file_path.relative_to(data_folder)
        if sampled_file_path.exists():
            continue

        if file_path.suffix.lower() not in included_extensions:
            continue

        sampled_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the original data
        df = data_handler.load(file_path)

        # Create a sampled subset
        df_sampled = data_reducer.reduce(df)

        # Dump the sampled data
        try:
            data_handler.dump(df_sampled, sampled_file_path)
            # Extract possible file references from the sampled data
            for col in df_sampled.columns:
                unique_vals = df_sampled[col].astype(str).unique()
                for val in unique_vals:
                    # Add the entire string to the set;
                    # in real usage, might want to parse or extract basename, etc.
                    sample_used_file_names.add(val)
        except Exception as e:
            print(f"Error processing {file_path}: {e}")
            continue

    # Process non-data files
    subfolder_dict = {}
    for file_path in files_to_process:
        if file_path.suffix.lower() in included_extensions:
            continue  # Already handled above
        rel_dir = file_path.relative_to(data_folder).parts[0]
        subfolder_dict.setdefault(rel_dir, []).append(file_path)

    # For each subfolder, decide which files to copy
    for rel_dir, file_list in subfolder_dict.items():
        used_files = []
        not_used_files = []

        # Check if each file is in the "used" list
        for fp in file_list:
            if str(fp.name) in sample_used_file_names or str(fp.stem) in sample_used_file_names:
                used_files.append(fp)
            else:
                not_used_files.append(fp)

        # Directly copy used files
        for uf in used_files:
            sampled_file_path = sample_folder / uf.relative_to(data_folder)
            if sampled_file_path.exists():
                continue
            sampled_file_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy(uf, sampled_file_path)

        # If no files are used, randomly sample files to keep the folder from being empty
        if len(used_files) == 0:
            if len(file_list) <= min_num:
                num_to_keep = len(file_list)
            else:
                num_to_keep = max(int(len(file_list) * min_frac), min_num)
            print(f"Sampling {num_to_keep} files without label from {len(file_list)} files in {rel_dir}")
            sampled_not_used = pd.Series(not_used_files).sample(n=num_to_keep, random_state=1)
            for nf in sampled_not_used:
                sampled_file_path = sample_folder / nf.relative_to(data_folder)
                if sampled_file_path.exists():
                    continue
                sampled_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(nf, sampled_file_path)

    final_files_count = count_files_in_folder(sample_folder)
    print(f"[INFO] After sampling, the sample folder `{sample_folder}` contains {final_files_count} files in total.")
