import shutil
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

try:
    import bson  # pip install pymongo
except:
    pass


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
            return pd.read_csv(path, encoding="utf-8")
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
            df.to_csv(path, index=False, encoding="utf-8")
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

    def reduce(self, df: pd.DataFrame, frac: float = None) -> pd.DataFrame:
        frac = max(self.min_frac, self.min_num / len(df)) if frac is None else frac
        # print(f"Sampling {frac * 100:.2f}% of the data ({len(df)} rows)")
        if frac >= 1:
            return df
        return df.sample(frac=frac, random_state=1)


class UniqueIDDataReducer(DataReducer):
    def __init__(self, min_frac=0.02, min_num=5):
        self.min_frac = min_frac
        self.min_num = min_num
        self.random_reducer = RandDataReducer(min_frac, min_num)

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        if not len(df):
            return df

        if not isinstance(df, pd.DataFrame):
            return self.random_reducer.reduce(df)

        def is_valid_label(column):
            if not isinstance(column.iloc[0], (int, float, str, tuple, frozenset, bytes, complex, type(None))):
                return False

            if not (0 < column.nunique() < df.shape[0] * 0.5):
                return False

            if pd.api.types.is_numeric_dtype(column) and all(isinstance(x, float) for x in column.dropna()):
                return False

            return True

        label_col = df.iloc[:, -1]
        if not is_valid_label(label_col) and df.shape[1] > 2:
            label_col = df.iloc[:, 1]

        if not is_valid_label(label_col):
            return self.random_reducer.reduce(df)

        unique_labels = label_col.unique()
        unique_count = len(unique_labels)
        print(f"Unique labels: {unique_count} / {df.shape[0]}")

        sampled_rows = df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(n=1, random_state=1))
        frac = max(self.min_frac, self.min_num / len(df))

        if int(len(df) * frac) < unique_count:
            return sampled_rows.reset_index(drop=True)

        remain_df = df.drop(index=sampled_rows.index)
        remaining_frac = frac - unique_count / len(df)

        remaining_sampled = self.random_reducer.reduce(remain_df, remaining_frac)
        result_df = pd.concat([sampled_rows, remaining_sampled]).sort_index()
        return result_df


def count_files_in_folder(folder: Path) -> int:
    """
    Count the total number of files in a folder, including files in subfolders.
    """
    return sum(1 for _ in folder.rglob("*") if _.is_file())


def copy_file(src_fp, target_folder, data_folder):
    """
    Construct the target file path based on the file's relative location from data_folder,
    then copy the file if it doesn't already exist.
    """
    target_fp = target_folder / src_fp.relative_to(data_folder)
    if not target_fp.exists():
        target_fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_fp, target_fp)


def create_debug_data(
    competition: str,
    dataset_path: str | Path,
    dr_cls: type[DataReducer] = UniqueIDDataReducer,
    min_frac=0.01,
    min_num=5,
    sample_path=None,
):
    """
    Reads the original data file, creates a reduced sample,
    and renames/moves files for easier debugging.
    Automatically detects file type (csv, pkl, parquet, hdf, etc.).
    """
    dataset_path = Path(dataset_path)
    if sample_path is None:
        sample_path = dataset_path / "sample"

    data_folder = dataset_path / competition
    sample_folder = sample_path / competition

    # Traverse the folder and exclude specific file types
    included_extensions = {".csv", ".pkl", ".parquet", ".h5", ".hdf", ".hdf5", ".jsonl", ".bson"}
    files_to_process = [file for file in data_folder.rglob("*") if file.is_file()]
    total_files_count = len(files_to_process)
    print(
        f"[INFO] Original dataset folder `{data_folder}` has {total_files_count} files in total (including subfolders)."
    )
    file_types_count = Counter(file.suffix.lower() for file in files_to_process)
    print("File type counts:")
    for file_type, count in file_types_count.items():
        print(f"{file_type}: {count}")

    # This set will store filenames or paths that appear in the sampled data
    sample_used_file_names = set()

    # Prepare data handler and reducer
    data_handler = GenericDataHandler()
    data_reducer = dr_cls(min_frac=min_frac, min_num=min_num)

    skip_subfolder_data = any(
        f.is_file() and f.suffix in included_extensions
        for f in data_folder.iterdir()
        if f.name.startswith(("train", "test"))
    )
    processed_files = []

    for file_path in tqdm(files_to_process, desc="Processing data", unit="file"):
        sampled_file_path = sample_folder / file_path.relative_to(data_folder)
        if sampled_file_path.exists():
            continue

        if file_path.suffix.lower() not in included_extensions:
            continue

        if skip_subfolder_data and file_path.parent != data_folder:
            continue  # bypass files in subfolders

        sampled_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Load the original data
        df = data_handler.load(file_path)

        # Create a sampled subset
        df_sampled = data_reducer.reduce(df)
        processed_files.append(file_path)
        # Dump the sampled data
        try:
            data_handler.dump(df_sampled, sampled_file_path)
            # Extract possible file references from the sampled data
            if "submission" in file_path.stem:
                continue  # Skip submission files
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
    global_groups = defaultdict(list)
    for file_path in files_to_process:
        if file_path in processed_files:
            continue  # Already handled above
        rel_dir = file_path.relative_to(data_folder).parts[0]
        subfolder_dict.setdefault(rel_dir, []).append(file_path)
        global_groups[file_path.stem].append(Path(file_path))

    # For each subfolder, decide which files to copy
    selected_groups = []
    for rel_dir, file_list in tqdm(subfolder_dict.items(), desc="Processing files", unit="file"):
        used_files = []
        not_used_files = []
        extra_files = []

        # Check if each file is in the "used" list
        for fp in file_list:
            if str(fp.name) in sample_used_file_names or str(fp.stem) in sample_used_file_names:
                used_files.append(fp)
            else:
                if file_types_count.get(".txt", 1000) < 100 and fp.suffix.lower() == ".txt":
                    extra_files.append(fp)
                not_used_files.append(fp)

        # Directly copy used files
        for uf in used_files:
            copy_file(uf, sample_folder, data_folder)

        # If no files are used, randomly sample files to keep the folder from being empty
        if len(used_files) == 0:
            if len(file_list) <= min_num:
                num_to_keep = len(file_list)
            else:
                num_to_keep = max(int(len(file_list) * min_frac), min_num)

            # Use a greedy strategy to select groups so that the total number of files is as close as possible to num_to_keep
            total_files = 0
            np.random.shuffle(not_used_files)
            for nf in not_used_files:
                if total_files > num_to_keep:
                    break
                if nf.stem in selected_groups:
                    total_files += 1
                else:
                    selected_groups.append(nf.stem)
                    total_files += 1

            print(f"Sampling {num_to_keep} files without label from {total_files} files in {rel_dir}")

            # Flatten the selected groups into a single list of files
            sampled_not_used = [
                nf for group, value in global_groups.items() if group in selected_groups for nf in value
            ]

            # Copy the selected files to the target directory (all files with the same base name will be copied)
            for nf in sampled_not_used:
                # Construct the target path based on the relative path of nf from data_folder
                sampled_file_path = sample_folder / nf.relative_to(data_folder)
                if sampled_file_path.exists():
                    continue
                sampled_file_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy(nf, sampled_file_path)

        # Copy extra files
        print(f"Copying {len(extra_files)} extra files")
        for uf in extra_files:
            copy_file(uf, sample_folder, data_folder)

    final_files_count = count_files_in_folder(sample_folder)
    print(f"[INFO] After sampling, the sample folder `{sample_folder}` contains {final_files_count} files in total.")
