import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

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
        elif suffix == ".json":
            # Not each json file is able to be converted to a DataFrame
            try:
                return pd.read_json(path, lines=False)
            except:
                return None
        elif suffix == ".bson":
            data = bson.decode_file_iter(open(path, "rb"))
            df = pd.DataFrame(data)
            return df
        else:
            raise ValueError(f"Unsupported file type: {suffix}")

    def dump(self, df: pd.DataFrame | dict, path):
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
        elif suffix == ".json":
            df.to_json(path, orient="records", lines=False)
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

    def __init__(self, min_frac=0.02, min_num=5):
        self.min_frac = min_frac
        self.min_num = min_num
        self.sampled_files = []

    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError


class RandDataReducer(DataReducer):
    """
    Example random sampler: ensures at least `min_num` rows
    or at least `min_frac` fraction of the data (whichever is larger).
    """

    def reduce(self, df: pd.DataFrame, frac: float = None) -> pd.DataFrame:
        frac = max(self.min_frac, self.min_num / len(df)) if frac is None else frac
        # print(f"Sampling {frac * 100:.2f}% of the data ({len(df)} rows)")
        if frac >= 1:
            return df
        return df.sample(frac=frac, random_state=1)


class FolderReducer(DataReducer):
    """
    Sample folder from a large number of folders.
    """

    def reduce(self, array: list, frac: float = None) -> list:
        frac = max(self.min_frac, self.min_num / len(array)) if frac is None else frac
        if frac >= 1:
            return array
        train_items = [x for x in array if "train" in str(x)]
        test_items = [x for x in array if "test" in str(x)]

        # 至少保留一个 train 和一个 test
        mandatory = []
        if train_items:
            mandatory.append(np.random.choice(train_items, size=1, replace=False)[0])
        if test_items:
            mandatory.append(np.random.choice(test_items, size=1, replace=False)[0])
        mandatory.extend(np.random.choice(array, size=int(len(array) * frac) - len(mandatory), replace=False))
        return mandatory


class FileReducer(DataReducer):
    """
    Sample file from a large number of files, keep min_num of files for each folder.
    """

    def reduce(self, files: list[Path]) -> list:
        folder_dict = defaultdict(list)
        for file in files:
            folder_dict[file.parent].append(file)

        sampled_files = []
        for folder, folder_files in folder_dict.items():
            n = min(max(int(len(folder_files) * self.min_frac), self.min_num), len(folder_files))
            sampled_files.extend(np.random.choice(folder_files, size=n, replace=False))
        return sampled_files


class FileKeepReducer(DataReducer):
    """
    Sample file from a large number of files, keep min_num of files for each folder.
    """

    def reduce(self, files: list[Path]) -> list:
        folder_dict = defaultdict(list)
        for file in files:
            folder_dict[file.parent].append(file)

        sampled_files = []
        max_num = max(len(folder_files) for folder_files in folder_dict.values())
        for folder, folder_files in folder_dict.items():
            print(f"[INFO] Folder {folder} contains {len(folder_files)} files.")
            if len(folder_files) < max_num * self.min_frac:
                print(f"[INFO] Folder {folder} less than {max_num * self.min_frac} files.")
                sampled_files.extend(folder_files)
                continue
            n = min(max(int(len(folder_files) * self.min_frac), self.min_num), len(folder_files))
            sampled_files.extend(np.random.choice(folder_files, size=n, replace=False))
        return sampled_files


class SingleFileReducer(DataReducer):
    """
    Sample file from a large number of files, keep at least 1 file.
    """

    def reduce(self, files: list[Path]) -> list:
        n = min(max(int(len(files) * self.min_frac), 1), len(files))
        return np.random.choice(files, size=n, replace=False)


class UniqueIDDataReducer(DataReducer):
    def reduce(self, df: pd.DataFrame) -> pd.DataFrame:
        if not len(df):
            return df

        random_reducer = RandDataReducer(self.min_frac, self.min_num)
        if not isinstance(df, pd.DataFrame):
            return random_reducer.reduce(df)

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
            return random_reducer.reduce(df)

        unique_labels = label_col.unique()
        unique_count = len(unique_labels)
        print(f"Unique labels: {unique_count} / {df.shape[0]}")

        sampled_rows = df.groupby(label_col, group_keys=False).apply(lambda x: x.sample(n=1, random_state=1))
        frac = max(self.min_frac, self.min_num / len(df))

        if int(len(df) * frac) < unique_count:
            return sampled_rows.reset_index(drop=True)

        remain_df = df.drop(index=sampled_rows.index)
        remaining_frac = frac - unique_count / len(df)

        remaining_sampled = random_reducer.reduce(remain_df, remaining_frac)
        result_df = pd.concat([sampled_rows, remaining_sampled]).sort_index()
        return result_df


class JsonReducer(DataReducer):

    def extract_filename(self, item: Any) -> Optional[str]:
        if isinstance(item, str):
            return item

        if isinstance(item, dict):
            for key in ("file_name", "filename", "path", "file", "url"):
                if key in item and isinstance(item[key], str):
                    return item[key]

            for v in item.values():
                if isinstance(v, str):
                    if "/" in v or re.search(r"\.\w{2,4}$", v):
                        return v

        return None

    def reduce(self, data: dict) -> dict:
        """
        1. 找到最大列表
        2. 随机采样并替换
        """
        candidates: List[Tuple[Union[Dict, str, int, List], Union[str, int], List[Any]]] = []
        self._find_all_lists(data, None, None, candidates)

        for parent, key, lst in sorted(candidates, key=lambda x: len(x[2]), reverse=True):
            sampled = self._sample_list(lst)
            if isinstance(parent, dict):
                parent[key] = sampled  # type: ignore
            else:
                parent[key] = sampled  # type: ignore  # parent 是 list，key 是 index, list.__setitem__(key, sampled)
            self.sampled_files.extend([self.extract_filename(i) for i in sampled])
            break
        assert len(self.sampled_files) > 0
        return data

    def _find_all_lists(
        self,
        current: Any,
        parent: Union[Dict, List, None],
        key: Union[str, int, None],
        out: List[Tuple[Union[Dict, List], Union[str, int], List[Any]]],
    ) -> None:
        """
        out => (parent_container, key_or_index, the_list)。
        """
        if isinstance(current, dict):
            for k, v in current.items():
                if isinstance(v, list):
                    out.append((current, k, v))
                    self._find_all_lists(v, current, k, out)
                elif isinstance(v, (dict, list)):
                    self._find_all_lists(v, current, k, out)

        elif isinstance(current, list):
            if parent is not None and key is not None:
                out.append((parent, key, current))
            for idx, item in enumerate(current):
                if isinstance(item, (dict, list)):
                    self._find_all_lists(item, current, idx, out)

    def _sample_list(self, lst: List[Any]) -> List[Any]:
        target = max(self.min_num, int(len(lst) * self.min_frac))
        if target >= len(lst):
            return lst[:]
        return np.random.choice(lst, size=target, replace=False)


class DataSampler:
    """Base DataSampler interface."""

    def __init__(self, data_folder, sample_folder, reducer):
        self.data_folder = data_folder
        self.sample_folder = sample_folder
        self.data_reducer = reducer
        self.included_extensions = {".csv", ".pkl", ".parquet", ".h5", ".hdf", ".hdf5", ".jsonl", ".bson"}
        self.data_handler = GenericDataHandler()

    def sample(self) -> None:
        raise NotImplementedError


class DefaultSampler(DataSampler):
    def sample(self) -> None:
        # Traverse the folder and exclude specific file types, without json currently

        files_to_process = [file for file in self.data_folder.rglob("*") if file.is_file()]
        file_types_count = count_files_in_folder(files_to_process)
        sample_json = False
        if isinstance(self.data_reducer, JsonReducer):
            self.included_extensions.add(".json")
            sample_json = True

        skip_subfolder_data = any(
            f.is_file() and f.suffix in self.included_extensions
            for f in self.data_folder.iterdir()
            if f.name.startswith(("train", "test"))
        )
        processed_files = []
        sample_used_file_names = set()
        has_id_col = False

        for file_path in tqdm(files_to_process, desc="Processing data", unit="file"):
            sampled_file_path = self.sample_folder / file_path.relative_to(self.data_folder)
            if sampled_file_path.exists():
                continue

            if file_path.suffix.lower() not in self.included_extensions:
                continue

            if skip_subfolder_data and file_path.parent != self.data_folder:
                continue  # bypass files in subfolders

            sampled_file_path.parent.mkdir(parents=True, exist_ok=True)

            # Load the original data
            if sample_json:
                if file_path.suffix.lower() == ".json":
                    data = json.load(file_path.open())
                    data_sampled = self.data_reducer.reduce(data)
                    sample_used_file_names = [file_path.parent / i for i in self.data_reducer.sampled_files]
                    print("sample_used_file_names", len(sample_used_file_names))
            else:
                df = self.data_handler.load(file_path)
                if df is None:
                    continue

                # Create a sampled subset
                df_sampled = self.data_reducer.reduce(df)
                processed_files.append(file_path)
                # Dump the sampled data
                try:
                    self.data_handler.dump(df_sampled, sampled_file_path)
                    # Extract possible file references from the sampled data
                    if "submission" in file_path.stem:
                        continue  # Skip submission files
                    for col in df_sampled.columns:
                        if "id" in col:
                            has_id_col = True
                            sample_used_file_names.extend([df_sampled[col].astype(str).unique()])
                            continue
                    for col in df_sampled.columns:
                        sample_used_file_names.extend([df_sampled[col].astype(str).unique()])
                except Exception as e:
                    print(f"Error processing {file_path}: {e}")
                    continue

        # Process non-data files
        subfolder_dict = {}
        global_groups = defaultdict(list)
        for file_path in files_to_process:
            if file_path in processed_files:
                continue  # Already handled above
            rel_dir = file_path.relative_to(self.data_folder).parts[0]
            subfolder_dict.setdefault(rel_dir, []).append(file_path)
            global_groups[file_path.stem].append(Path(file_path))

        # For each subfolder, decide which files to copy
        selected_groups = []
        extra_tag = [".txt", ".json"]
        for rel_dir, file_list in tqdm(subfolder_dict.items(), desc="Processing files", unit="file"):
            used_files = []
            not_used_files = []
            extra_files = []

            # Check if each file is in the "used" list
            for fp in file_list:
                if (
                    str(fp.name) in sample_used_file_names
                    or str(fp.stem) in sample_used_file_names
                    or fp in sample_used_file_paths
                ):
                    used_files.append(fp)
                else:
                    for tag in extra_tag:
                        if file_types_count.get(tag, 1000) < 100 and fp.suffix.lower() == tag:
                            extra_files.append(fp)
                    not_used_files.append(fp)

            # Directly copy used files
            for uf in used_files:
                copy_file(uf, self.sample_folder, self.data_folder)

            # If no files are used, randomly sample files to keep the folder from being empty
            if len(used_files) == 0:
                if len(file_list) <= self.data_reducer.min_num:
                    num_to_keep = len(file_list)
                else:
                    num_to_keep = max(int(len(file_list) * self.data_reducer.min_frac), self.data_reducer.min_num)

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
                    sampled_file_path = self.sample_folder / nf.relative_to(self.data_folder)
                    if sampled_file_path.exists():
                        continue
                    sampled_file_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(nf, sampled_file_path)

            # Copy extra files
            print(f"Copying {len(extra_files)} extra files")
            for uf in extra_files:
                copy_file(uf, self.sample_folder, self.data_folder)

        final_files_count = sum(1 for _ in self.sample_folder.rglob("*") if _.is_file())
        print(
            f"[INFO] After sampling, the sample folder `{self.sample_folder}` contains {final_files_count} files in total."
        )


class FolderSampler(DataSampler):
    """
    Sample data from a large number of folders.
    """

    def sample(self) -> None:
        sample_used_file_names = []
        current_level = [d for d in self.data_folder.iterdir() if d.is_dir()]
        last_count = 0
        subdirs = []
        sample_dirs = []
        sample_files = []
        extra_files = [d for d in self.data_folder.iterdir() if d.is_file()]
        level = 1
        while current_level:
            subdirs = [d for current_dir in current_level for d in current_dir.iterdir() if d.is_dir()]
            subdirs_names = [d.name for d in subdirs]
            extra_files.extend([d for current_dir in current_level for d in current_dir.iterdir() if d.is_file()])
            if not subdirs:
                print("current_level", len(current_level))
                subfiles = [d for current_dir in current_level for d in current_dir.iterdir() if d.is_file()]
                sample_files = self.data_reducer.reduce(subfiles)
                extra_files = list(set(extra_files) - set(subfiles))
                print(f"sample {len(sample_files)} files from {len(subfiles)}")
                break

            print(
                f"subdirs count: {len(set(subdirs_names))}, last_count: {last_count}, subdirs[0]: {subdirs[0]}, sample_used_file_names count: {len(set(sample_used_file_names))}"
            )
            if sample_used_file_names and set(sample_used_file_names).issubset(set(subdirs_names)):
                sample_dirs = [d for d in subdirs if d.name in sample_used_file_names]
                print(f"sample {len(sample_dirs)} folders from {len(subdirs)}")
                break

            if len(subdirs_names) > 100 or (last_count and 1 < len(sample_dirs) < last_count):
                sample_dirs = self.data_reducer.reduce(subdirs)
                print(f"sample {len(sample_dirs)} folders from {len(subdirs)}")
                break
            last_count = len(set(subdirs_names))
            current_level = subdirs
            level += 1

        print(
            f"[INFO] After sampling, the sample folder `{self.sample_folder}` contains extra_files {len(extra_files)} folders in total."
        )
        for i in sample_dirs:
            copy_folder(i, self.sample_folder, self.data_folder)
        for i in sample_files:
            copy_file(i, self.sample_folder, self.data_folder)
        for i in set(extra_files):
            copy_file(i, self.sample_folder, self.data_folder)


class SingleFilePerFolderSampler(DataSampler):
    """
    For each leaf (final) subfolder under data_folder, keep exactly one file (randomly chosen).
    Files in non-leaf folders are copied unchanged.
    """

    def sample(self) -> None:
        data_folder = Path(self.data_folder)
        sample_folder = Path(self.sample_folder)

        # Find all leaf directories (no subdirectories)
        leaf_dirs = [Path(root) for root, dirs, _ in os.walk(data_folder) if not dirs]
        print(f"Found {len(leaf_dirs)} leaf directories")

        # Sample one file per leaf directory
        for leaf in tqdm(leaf_dirs, desc="Processing files", unit="file"):
            files = [f for f in leaf.iterdir() if f.is_file()]
            if not files:
                continue
            chosen = self.data_reducer.reduce(files)
            for f in chosen:
                copy_file(f, sample_folder, data_folder)

        # Copy all files in non-leaf directories
        # i.e. any file whose parent is not a leaf dir
        # Copy all files in non-leaf directories
        for root, _, files in os.walk(data_folder):
            current_dir = Path(root)
            if current_dir in leaf_dirs:
                continue
            for fname in files:
                file_path = current_dir / fname
                copy_file(file_path, sample_folder, data_folder)

        total = sum(1 for _ in sample_folder.rglob("*") if _.is_file())
        print(f"[INFO] SingleFilePerFolderSampler: copied {total} files to {sample_folder}")


def copy_file(src_fp, target_folder, data_folder):
    """
    Construct the target file path based on the file's relative location from data_folder,
    then copy the file if it doesn't already exist.
    """
    target_fp = target_folder / src_fp.relative_to(data_folder)
    if not target_fp.exists():
        target_fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy(src_fp, target_fp)


def copy_folder(src_fp, target_folder, data_folder):
    """
    Copy a folder recursively.
    """
    target_fp = target_folder / src_fp.relative_to(data_folder)
    if not target_fp.exists():
        target_fp.parent.mkdir(parents=True, exist_ok=True)
        shutil.copytree(src_fp, target_fp)


def count_files_in_folder(files_to_process):
    """
    Count the number of each file type in a folder, including files in subfolders.
    """
    total_files_count = len(files_to_process)
    print(f"[INFO] Original dataset folder has {total_files_count} files in total (including subfolders).")
    file_types_count = Counter(file.suffix.lower() for file in files_to_process)
    print("File type counts:")
    for file_type, count in file_types_count.items():
        print(f"{file_type}: {count}")
    return file_types_count


def map_competition(competition: str) -> tuple[DataReducer, DataSampler]:
    cls_map = {
        "google-research-identify-contrails-reduce-global-warming": (FolderReducer, FolderSampler),
        "smartphone-decimeter-2022": (FolderReducer, FolderSampler),
        "herbarium-2020-fgvc7": (SingleFileReducer, SingleFilePerFolderSampler),
        "herbarium-2021-fgvc8": (SingleFileReducer, SingleFilePerFolderSampler),
        "herbarium-2022-fgvc9": (SingleFileReducer, SingleFilePerFolderSampler),
        "vesuvius-challenge-ink-detection": (FileReducer, FolderSampler),
        "3d-object-detection-for-autonomous-vehicles": (FileKeepReducer, FolderSampler),
    }
    return cls_map.get(competition, (UniqueIDDataReducer, DefaultSampler))


def create_debug_data(
    competition: str,
    dataset_path: str | Path,
    min_frac=0.01,
    min_num=5,
    sample_path=None,
):
    """
    Reads the original data file, creates a reduced sample,
    and renames/moves files for easier debugging.
    Automatically detects file type (csv, pkl, parquet, hdf, etc.).
    """
    if sample_path is None:
        sample_path = Path(dataset_path) / "sample"

    # Prepare data handler and reducer
    reduce_method, sample_method = map_competition(competition)
    data_reducer = reduce_method(min_frac=min_frac, min_num=min_num)
    sampler = sample_method(Path(dataset_path) / competition, Path(sample_path) / competition, data_reducer)
    print(f"processing {competition}, sample_method: {sample_method}, reduce_method: {reduce_method}")
    sampler.sample()
