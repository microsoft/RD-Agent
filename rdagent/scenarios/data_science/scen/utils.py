import os

import pandas as pd
from PIL import Image, TiffTags

from rdagent.log import rdagent_logger as logger

""" data folder description version 1 """


def read_csv_head(file_path, indent=0, lines=5, max_col_width=100):
    """
    Reads the first few rows of a CSV file and formats them with indentation and optional truncation.

    Parameters:
        file_path (str): Path to the CSV file.
        indent (int): Number of spaces to prepend to each line for indentation.
        lines (int): Number of rows to read from the CSV file.
        max_col_width (int): Maximum width of each column's content.

    Returns:
        str: A formatted string of the first few rows of the CSV file.
    """
    try:
        # Read the CSV file with specified rows
        df = pd.read_csv(file_path, nrows=lines)

        if df.empty:
            return " " * indent + "(No data in the file)"

        # Truncate column contents to a maximum width
        truncated_df = df.copy()
        for col in truncated_df.columns:
            truncated_df[col] = (
                truncated_df[col]
                .astype(str)
                .apply(lambda x: (x[:max_col_width] + "...") if len(x) > max_col_width else x)
            )

        # Convert DataFrame to a string representation
        df_string_lines = truncated_df.to_string(index=False).split("\n")

        # Add indentation to each line
        indented_lines = [" " * indent + line for line in df_string_lines]

        return "\n".join(indented_lines)
    except FileNotFoundError:
        return f"Error: File not found at path '{file_path}'."
    except pd.errors.EmptyDataError:
        return f"Error: The file at '{file_path}' is empty."
    except Exception as e:
        return f"Error reading CSV: {e}"


def get_dir_snapshot(folder_path):
    """
    [note]
        - Returns a set of file extensions within the subfolder (excluding subfolder names)
        - Compares only the types of files contained, not specific file names or quantities
    """
    exts = set()
    try:
        with os.scandir(folder_path) as it:
            for entry in it:
                if entry.is_file():
                    file_ext = os.path.splitext(entry.name)[1]
                    exts.add(file_ext)
    except Exception as e:
        logger.error(f"Error scanning directory: {e}")

    return frozenset(exts)


def describe_data_folder(folder_path, indent=0, max_files=2, partial_expand_subfolders=2, is_top_level=True):
    """
    folder_path              : Current directory path
    indent                   : Current indentation
    max_files                : Maximum number of files of the same type to display
    partial_expand_subfolders: When all subfolders have the same internal file types, only expand this many subfolders, the rest are omitted
    is_top_level             : Indicates if the current folder is the top-level folder
    """
    result = []
    files_count = {}
    files_details = {}

    for root, dirs, files in os.walk(folder_path):
        dirs.sort()
        files.sort()
        if not dirs:
            for file in files:
                file_path = os.path.join(root, file)
                file_type = os.path.splitext(file)[1][1:]
                file_size = os.path.getsize(file_path)

                if file_type not in files_count:
                    files_count[file_type] = 0
                    files_details[file_type] = []
                files_count[file_type] += 1

                # At top level, collect all CSV and Markdown files without restrictions
                # In deeper levels, follow the max_files restriction
                if is_top_level and file_type in ["csv", "md"]:
                    files_details[file_type].append((file, file_size, file_path))
                elif len(files_details[file_type]) < max_files:
                    files_details[file_type].append((file, file_size, file_path))
            break

        # Collect "type snapshots" of subfolders
        snapshots = []
        for d in dirs:
            subfolder_path = os.path.join(root, d)
            snapshot = get_dir_snapshot(subfolder_path)
            snapshots.append(snapshot)

        # Determine if all subfolders have the same file type distribution
        first_snapshot = snapshots[0]
        all_same_structure = all(s == first_snapshot for s in snapshots)

        if all_same_structure:
            for i, d in enumerate(dirs):
                if i < partial_expand_subfolders:
                    result.append(" " * indent + f"- Folder: {d}")
                    subfolder_path = os.path.join(root, d)
                    result.append(
                        describe_data_folder(
                            folder_path=subfolder_path,
                            indent=indent + 2,
                            max_files=max_files,
                            partial_expand_subfolders=partial_expand_subfolders,
                            is_top_level=False,
                        )
                    )
                else:
                    remaining = len(dirs) - i
                    result.append(" " * indent + f"... ({remaining} more subfolders)")
                    break
        else:
            for d in dirs:
                result.append(" " * indent + f"- Folder: {d}")
                subfolder_path = os.path.join(root, d)
                result.append(
                    describe_data_folder(
                        folder_path=subfolder_path,
                        indent=indent + 2,
                        max_files=max_files,
                        partial_expand_subfolders=partial_expand_subfolders,
                        is_top_level=False,
                    )
                )

        for file in files:
            file_path = os.path.join(root, file)
            file_type = os.path.splitext(file)[1][1:]
            file_size = os.path.getsize(file_path)

            if file_type not in files_count:
                files_count[file_type] = 0
                files_details[file_type] = []
            files_count[file_type] += 1

            # At top level, collect all CSV and Markdown files without restrictions
            # In deeper levels, follow the max_files restriction
            if is_top_level and file_type in ["csv", "md"]:
                files_details[file_type].append((file, file_size, file_path))
            elif not is_top_level and len(files_details[file_type]) <= max_files:
                files_details[file_type].append((file, file_size, file_path))

        break

    # Print the folder and its contents
    for file_type, count in files_count.items():
        if count > max_files and file_type not in ["csv", "md", "txt"]:
            result.append(" " * indent + f"{count} {file_type}s:")
            for file, size, path in files_details[file_type]:
                result.append(" " * (indent + 2) + f"- {file} ({size} bytes)")
            result.append(" " * (indent + 2) + "... (file limit reached)")
        else:
            for file, size, path in files_details[file_type]:
                if file_type == "csv":
                    df = pd.read_csv(path)
                    result.append(
                        " " * indent + f"- {file} ({size} bytes, with {df.shape[0]} rows and {df.shape[1]} columns)"
                    )
                    result.append(" " * (indent + 2) + f"- Head of {file}:")
                    csv_head = read_csv_head(path, indent + 4)
                    result.append(csv_head)
                    continue
                result.append(" " * indent + f"- {file} ({size} bytes)")
                if file_type == "md":
                    result.append(" " * (indent + 2) + f"- Content of {file}:")
                    if file == "description.md":
                        result.append(" " * (indent + 4) + f"Please refer to the background of the scenario context.")
                        continue
                    with open(path, "r", encoding="utf-8") as f:
                        result.append(" " * (indent + 4) + f.read())
                if file_type == "tif":
                    result.append(" " * (indent + 2) + f"- Metadata of {file}:")
                    with Image.open(path) as img:
                        for tag, value in img.tag_v2.items():
                            tag_name = TiffTags.TAGS_V2.get(tag, f"Unknown Tag {tag}")
                            result.append(" " * (indent + 4) + f"{tag_name}: {value}")
                if file_type in ["json", "txt"]:
                    result.append(" " * (indent + 2) + f"- Content of {file}:")
                    with open(path, "r", encoding="utf-8") as f:
                        for i, line in enumerate(f):
                            if i < 2:
                                result.append(
                                    " " * (indent + 4) + line.strip()[:100] + ("..." if len(line.strip()) > 100 else "")
                                )
                            else:
                                break

    return "\n".join(result) + "\n"


""" data folder description version 2 """
import json
from pathlib import Path

import humanize
import pandas as pd
from genson import SchemaBuilder
from pandas.api.types import is_numeric_dtype

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files
# system-generated directories/files to filter out
system_names = {"__MACOSX", ".DS_Store", "Thumbs.db"}


def get_file_len_size(f: Path) -> tuple[int, str]:
    """
    Calculate the size of a file (#lines for plaintext files, otherwise #bytes)
    Also returns a human-readable string representation of the size.
    """
    if f.suffix in plaintext_files:
        num_lines = sum(1 for _ in open(f))
        return num_lines, f"{num_lines} lines"
    else:
        s = f.stat().st_size
        return s, humanize.naturalsize(s)


def file_tree(path: Path, depth=0) -> str:
    """Generate a tree structure of files in a directory"""
    result = []

    files = [p for p in Path(path).iterdir() if not p.is_dir() and p.name not in system_names]

    max_n = 4 if len(files) > 30 else 8
    for p in sorted(files)[:max_n]:
        result.append(f"{' '*depth*4}{p.name} ({get_file_len_size(p)[1]})")
    if len(files) > max_n:
        result.append(f"{' '*depth*4}... and {len(files)-max_n} other files")

    dirs = [
        p
        for p in Path(path).iterdir()
        if (p.is_dir() or (p.is_symlink() and p.resolve().is_dir())) and p.name not in system_names
    ]

    # Calculate base_path (the top-level resolved absolute directory)
    base_path = Path(path).resolve()
    # Find the top-level base_path when in recursion (depth>0)
    if depth > 0:
        # The top-level base_path is the ancestor at depth==0
        ancestor = Path(path)
        for _ in range(depth):
            ancestor = ancestor.parent
        base_path = ancestor.resolve()

    for p in sorted(dirs):
        if p.is_symlink():
            target = p.resolve()
            if str(target).startswith(str(base_path)):
                # avoid recursing into symlinks pointing inside base path
                result.append(
                    f"{' ' * depth * 4}{p.name}@ -> {os.path.relpath(target, base_path)} (symlinked dir, not expanded)"
                )
                continue
        result.append(f"{' ' * depth * 4}{p.name}/")
        result.append(file_tree(p, depth + 1))

    return "\n".join(result)


def _walk(path: Path):
    """Recursively walk a directory (analogous to os.walk but for pathlib.Path)"""
    for p in sorted(Path(path).iterdir()):
        # Filter out system-generated directories/files
        if p.name in system_names:
            continue

        if p.is_dir():
            # If this is a symlinked dir to a parent/ancestor, do not expand it
            if p.is_symlink():
                target = p.resolve()
                cur_path = p.parent.resolve()
                if target == cur_path or str(cur_path).startswith(str(target)):
                    yield p
                    continue
            yield from _walk(p)
        else:
            yield p


def preview_df(df: pd.DataFrame, file_name: str, simple=True, show_nan_columns=False) -> str:
    """Generate a textual preview of a dataframe

    Args:
        df (pd.DataFrame): the dataframe to preview
        file_name (str): the file name to use in the preview
        simple (bool, optional): whether to use a simplified version of the preview. Defaults to True.

    Returns:
        str: the textual preview
    """
    out = []

    out.append(f"-> {file_name} has {df.shape[0]} rows and {df.shape[1]} columns.")

    if simple:
        cols = df.columns.tolist()
        sel_cols = min(len(cols), 100)
        cols_str = ", ".join(cols[:sel_cols])
        res = f"The columns are: {cols_str}"
        if len(cols) > sel_cols:
            res += f"... and {len(cols)-sel_cols} more columns"
        out.append(res)
    else:
        out.append("Here is some information about the columns:")
        for col in sorted(df.columns):
            dtype = df[col].dtype
            name = f"{col} ({dtype})"

            nan_count = df[col].isnull().sum()

            if dtype == "bool":
                v = df[col][df[col].notnull()].mean()
                out.append(f"{name} is {v*100:.2f}% True, {100-v*100:.2f}% False")
            elif df[col].nunique() < 10:
                out.append(f"{name} has {df[col].nunique()} unique values: {df[col].unique().tolist()}")
            elif is_numeric_dtype(df[col]):
                out.append(f"{name} has range: {df[col].min():.2f} - {df[col].max():.2f}, {nan_count} nan values")
            elif dtype == "object":
                out.append(
                    f"{name} has {df[col].nunique()} unique values. Some example values: {df[col].value_counts().head(4).index.tolist()}"
                )
    if show_nan_columns:
        nan_cols = [col for col in df.columns.tolist() if df[col].isnull().any()]
        if nan_cols:
            out.append(f"Columns containing NaN values: {', '.join(nan_cols)}")

    return "\n".join(out)


def preview_csv(p: Path, file_name: str, simple=True, show_nan_columns=False) -> str:
    """Generate a textual preview of a csv file"""
    df = pd.read_csv(p)
    return preview_df(df, file_name, simple=simple, show_nan_columns=show_nan_columns)


def preview_parquet(p: Path, file_name: str, simple=True, show_nan_columns=False) -> str:
    """Generate a textual preview of a parquet file"""
    df = pd.read_parquet(p)
    return preview_df(df, file_name, simple=simple, show_nan_columns=show_nan_columns)


def preview_json(p: Path, file_name: str):
    """Generate a textual preview of a json file using a generated json schema"""
    builder = SchemaBuilder()
    with open(p) as f:
        first_line = f.readline().strip()

        try:
            first_object = json.loads(first_line)

            if not isinstance(first_object, dict):
                raise json.JSONDecodeError("The first line isn't JSON", first_line, 0)

            # if the the next line exists and is not empty, then it is a JSONL file
            second_line = f.readline().strip()
            if second_line:
                f.seek(0)  # so reset and read line by line
                for line in f:
                    builder.add_object(json.loads(line.strip()))
            # if it is empty, then it's a single JSON object file
            else:
                builder.add_object(first_object)

        except json.JSONDecodeError:
            # if first line isn't JSON, then it's prettified and we can read whole file
            f.seek(0)
            builder.add_object(json.load(f))

    return f"-> {file_name} has auto-generated json schema:\n" + builder.to_json(indent=2)


def describe_data_folder_v2(
    base_path, include_file_details=True, simple=False, show_nan_columns=False, max_length: int = 6_000
):
    """
    Generate a textual preview of a directory, including an overview of the directory
    structure and previews of individual files
    """
    tree = f"```\n{file_tree(base_path)}```"
    out = [tree]

    if include_file_details:
        for fn in _walk(base_path):
            file_name = str(fn.relative_to(base_path))

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple, show_nan_columns=show_nan_columns))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix == ".parquet":
                out.append(preview_parquet(fn, file_name, simple=simple, show_nan_columns=show_nan_columns))
            elif fn.suffix in plaintext_files:
                if get_file_len_size(fn)[0] < 30:
                    with open(fn) as f:
                        content = f.read()
                        if fn.suffix in code_files:
                            content = f"```\n{content}\n```"
                        out.append(f"-> {file_name} has content:\n\n{content}")
            if len("\n\n".join(out)) > max_length:
                break

    result = "\n\n".join(out)

    # if the result is very long we generate a simpler version
    if len(result) > max_length and not simple:
        return describe_data_folder_v2(
            base_path,
            include_file_details=include_file_details,
            simple=True,
            show_nan_columns=show_nan_columns,
            max_length=max_length,
        )
    # if still too long, we truncate
    if len(result) > max_length and simple:
        return result[:max_length] + "\n... (truncated)"

    return result
