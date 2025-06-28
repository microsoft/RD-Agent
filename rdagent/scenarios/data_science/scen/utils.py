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
import reprlib
from pathlib import Path

import humanize
import pandas as pd
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

    out.append(f"### {file_name}: ")
    out.append(f"#### 1.DataFrame preview:")
    out.append(f"It has {df.shape[0]} rows and {df.shape[1]} columns.")

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

    # Add: Display DataFrame directly
    if len(df) > 0:
        out.append("#### 2.DataFrame preview:(only show the first 5 rows and 15 columns)")
        # Show first 5 rows, max 15 columns to avoid overly wide output
        df_preview = df.head(5)
        if df.shape[1] > 15:
            df_preview = df_preview.iloc[:, :15]
            out.append(str(df_preview))
            out.append(f"... (showing first 15 of {df.shape[1]} columns)")
        else:
            out.append(str(df_preview))

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
    """Generate a textual preview of a json file using reprlib for compact object display"""
    result = []
    result.append(f"### {file_name}:")

    try:
        # First check if this is a JSONL format
        is_jsonl = False

        with open(p, "r", encoding="utf-8") as f:
            first_line = f.readline().strip()
            second_line = f.readline().strip()

            # Correct JSONL detection: both lines must be independent complete JSON objects
            if first_line and second_line:
                try:
                    # Try to parse the first two lines, both should be complete JSON objects
                    json.loads(first_line)  # First line is complete JSON
                    json.loads(second_line)  # Second line is also complete JSON
                    is_jsonl = True
                except json.JSONDecodeError:
                    # If any line fails to parse, it's not JSONL
                    is_jsonl = False

        if is_jsonl:
            # JSONL format: one JSON object per line
            result.append("#### 1.Format: JSONL (JSON Lines)")
            result.append("#### 2.Content preview (first few objects):")

            # Configure reprlib for JSONL
            jsonl_repr = reprlib.Repr()
            jsonl_repr.maxother = 300

            with open(p, "r", encoding="utf-8") as f:
                for i, line in enumerate(f):
                    if i >= 3:  # Only show first 3 objects
                        result.append("... (showing first 3 JSONL objects)")
                        break
                    if line.strip():
                        try:
                            obj = json.loads(line.strip())
                            result.append(f"Object {i+1}: {jsonl_repr.repr(obj)}")
                        except:
                            result.append(f"Object {i+1}: Invalid JSON")
        else:
            # Single JSON file
            with open(p, "r", encoding="utf-8") as f:
                data = json.load(f)

            result.append("#### 1.Format: Single JSON object")
            result.append("#### 2.Structure overview:")

            # Basic information
            if isinstance(data, dict):
                result.append(f"Type: Object with {len(data)} keys: {list(data.keys())}")
                for key, value in data.items():
                    if isinstance(value, list):
                        result.append(f"  - {key}: array[{len(value)}]")
                    elif isinstance(value, dict):
                        result.append(f"  - {key}: object{{{len(value)} keys}}")
                    else:
                        result.append(f"  - {key}: {type(value).__name__}")
            elif isinstance(data, list):
                result.append(f"Type: Array with {len(data)} items")
                if len(data) > 0:
                    sample_item = data[0]
                    if isinstance(sample_item, dict):
                        result.append(f"Sample item keys: {list(sample_item.keys())}")

            result.append("#### 3.Content preview (reprlib):")

            # Use reprlib to display content
            compact_repr = reprlib.Repr()
            compact_repr.maxother = 300

            result.append(compact_repr.repr(data))

    except Exception as e:
        result.append(f"Error processing JSON: {str(e)[:100]}")

    return "\n".join(result)


def select_files_for_preview(base_path: Path, max_files_per_group: int = 1, threshold: int = 10) -> list:
    """
    Intelligently select a representative subset of files for detailed preview.
    If a directory has more than `threshold` files of the same type, only `max_files_per_group` are selected.
    """
    from collections import defaultdict

    # Group files by (parent_directory, file_extension)
    files_by_group = defaultdict(list)
    for p in _walk(base_path):
        if p.is_file():
            files_by_group[(p.parent, p.suffix)].append(p)

    selected_files = []

    # Always include a root README.md if it exists
    root_readme = base_path / "README.md"
    if root_readme.exists():
        selected_files.append(root_readme)

    for group, files in files_by_group.items():
        # Sort files to be deterministic (e.g., by name)
        files.sort()

        if root_readme in files:
            try:
                files.remove(root_readme)
            except ValueError:
                pass  # was not in list

        if len(files) > threshold:
            # If many files, select a few representatives
            selected_files.extend(files[:max_files_per_group])
        else:
            # If few files, select all of them
            selected_files.extend(files)

    # Remove duplicates and maintain order
    return list(dict.fromkeys(selected_files))


def describe_data_folder_v2(
    base_path, include_file_details=True, simple=False, show_nan_columns=False, max_length: int = 10000
):
    """
    Generate a textual preview of a directory, including an overview of the directory
    structure and previews of individual files.
    """

    tree = f"## File tree:\n```\n{file_tree_v2(base_path,max_lines=200)}```"
    out = [tree]

    if include_file_details:
        out.append("\n## File details:")

        # Intelligently select a subset of files to preview
        files_to_preview = select_files_for_preview(Path(base_path))
        out.append(f" (Showing details for {len(files_to_preview)} representative files out of many)")

        for fn in files_to_preview:
            try:
                file_name = str(fn.relative_to(Path(base_path)))
            except ValueError:
                file_name = str(fn)

            if fn.suffix == ".csv":
                out.append(preview_csv(fn, file_name, simple=simple, show_nan_columns=show_nan_columns))
            elif fn.suffix == ".json":
                out.append(preview_json(fn, file_name))
            elif fn.suffix == ".parquet":
                out.append(preview_parquet(fn, file_name, simple=simple, show_nan_columns=show_nan_columns))
            elif fn.suffix in plaintext_files:
                try:
                    if get_file_len_size(fn)[0] < 30:
                        with open(fn) as f:
                            content = f.read()
                            if fn.suffix in code_files:
                                content = f"```\n{content}\n```"
                            out.append(f"-> {file_name} has content:\n\n{content}")
                except Exception:
                    pass  # Ignore read errors for small files
            if len("\n\n".join(out)) > max_length:
                out.append("\n... (File details truncated due to max_length)")
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


def file_tree_v2(
    path: Path, max_lines: int = 200, priority_files: set = {".csv", ".json", ".parquet", ".md", ".txt"}
) -> str:
    """
    Smart file tree generator v2 - focused on solving truncation issues

    Args:
        path: Target directory path
        max_lines: Maximum output lines (to prevent overly long output)
        priority_files: File extensions to prioritize for display

    Display rules:
        - First level: Show all items regardless of count
        - Other levels ≤8 items: Show all
        - Other levels >8 items: Show first 2 + remaining count
    """
    path = Path(path)
    lines = []
    line_count = 0

    def add_line(text: str) -> bool:
        """Add a line, return whether we can continue adding more"""
        nonlocal line_count
        if line_count >= max_lines:
            return False
        lines.append(text)
        line_count += 1
        return True

    # Root directory
    if not add_line(f"{path.name}/"):
        return "\n".join(lines)

    # Process recursively
    _process_directory_v2(path, 0, "", priority_files, add_line)

    # Add truncation notice if needed
    if line_count >= max_lines:
        lines.append("... (display limited, please increase max_lines parameter)")

    return "\n".join(lines)


def _process_directory_v2(path: Path, depth: int, prefix: str, priority_files: set, add_line_func) -> bool:
    """Process a single directory"""

    try:
        # Get directory contents, reuse existing system_names filtering
        items = [p for p in path.iterdir() if not p.name.startswith(".") and p.name not in system_names]
        dirs = sorted([p for p in items if p.is_dir()])
        files = sorted([p for p in items if p.is_file()])

        # Categorize files, reuse existing classification logic
        priority_files_list, other_files = _categorize_files_v2(files, priority_files)

        # Handle subdirectories - show all at first level, limited at other levels
        if depth == 0 or len(dirs) <= 8:
            # First level or ≤8 items: show all
            for d in dirs:
                if not add_line_func(f"{prefix}├── {d.name}/"):
                    return False

                # Process subdirectory recursively
                child_prefix = prefix + "│   "
                if not _process_directory_v2(d, depth + 1, child_prefix, priority_files, add_line_func):
                    return False
        else:
            # Not first level and >8 items: show first 2
            show_count = 2
            for d in dirs[:show_count]:
                if not add_line_func(f"{prefix}├── {d.name}/"):
                    return False

                # Process subdirectory recursively
                child_prefix = prefix + "│   "
                if not _process_directory_v2(d, depth + 1, child_prefix, priority_files, add_line_func):
                    return False

            # Show remaining directory count
            remaining = len(dirs) - show_count
            if not add_line_func(f"{prefix}├── ... (+{remaining} more directories)"):
                return False

        # Merge all files for unified processing
        all_files = priority_files_list + other_files

        # Handle files - show all at first level, limited at other levels
        if depth == 0 or len(all_files) <= 8:
            # First level or ≤8 items: show all
            for f in all_files:
                if not add_line_func(f"{prefix}├── {f.name} ({_get_size_str_v2(f)})"):
                    return False
        else:
            # Not first level and >8 items: show first 2
            show_count = 2
            for f in all_files[:show_count]:
                if not add_line_func(f"{prefix}├── {f.name} ({_get_size_str_v2(f)})"):
                    return False

            # Show remaining file count
            remaining = len(all_files) - show_count
            if not add_line_func(f"{prefix}├── ... (+{remaining} more files)"):
                return False

    except PermissionError:
        return add_line_func(f"{prefix}❌ Permission denied")
    except Exception as e:
        return add_line_func(f"{prefix}❌ Error: {str(e)[:30]}")

    return True


def _categorize_files_v2(files: list, priority_files: set) -> tuple:
    """Categorize files into priority and other groups"""
    priority = []
    other = []

    for f in files:
        if f.suffix.lower() in priority_files:
            priority.append(f)
        else:
            other.append(f)

    # Sort priority files by size (larger files first)
    priority.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)

    return priority, other


def _get_size_str_v2(file_path: Path) -> str:
    """Get file size string, reuse existing humanize logic"""
    try:
        size = file_path.stat().st_size
        return humanize.naturalsize(size)
    except:
        return "? B"
