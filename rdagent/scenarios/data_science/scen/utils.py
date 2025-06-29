"""
An example of the generated data folder description:

## File tree:
```
./
├── images/
│   ├── Test_0.jpg (182.7 kB)
│   ├── Test_1.jpg (362.4 kB)
│   ├── ... (+1819 more files)
├── train.csv (30.1 kB)
├── description.md (5.3 kB)
├── sample_submission.csv (5.2 kB)
├── test.csv (1.5 kB)```


## File details:

 (Showing details for representative files out of many)

### sample_submission.csv:
#### 1.DataFrame preview:
It has 183 rows and 5 columns.
Here is some information about the columns:
healthy (float64) has 1 unique values: [0.25]
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_1', 'Test_2', 'Test_3']
multiple_diseases (float64) has 1 unique values: [0.25]
rust (float64) has 1 unique values: [0.25]
scab (float64) has 1 unique values: [0.25]
#### 2.DataFrame preview:(only show the first 5 rows and 15 columns)
  image_id  healthy  multiple_diseases  rust  scab
0   Test_0     0.25               0.25  0.25  0.25
1   Test_1     0.25               0.25  0.25  0.25
2   Test_2     0.25               0.25  0.25  0.25
3   Test_3     0.25               0.25  0.25  0.25
4   Test_4     0.25               0.25  0.25  0.25

### test.csv:
#### 1.DataFrame preview:
It has 183 rows and 1 columns.
Here is some information about the columns:
image_id (object) has 183 unique values. Some example values: ['Test_0', 'Test_1', 'Test_2', 'Test_3']
#### 2.DataFrame preview:(only show the first 5 rows and 15 columns)
  image_id
0   Test_0
1   Test_1
2   Test_2
3   Test_3
4   Test_4

### train.csv:
#### 1.DataFrame preview:
It has 1638 rows and 5 columns.
Here is some information about the columns:
healthy (int64) has 2 unique values: [0, 1]
image_id (object) has 1638 unique values. Some example values: ['Train_1637', 'Train_0', 'Train_1', 'Train_2']
multiple_diseases (int64) has 2 unique values: [0, 1]
rust (int64) has 2 unique values: [1, 0]
scab (int64) has 2 unique values: [0, 1]
#### 2.DataFrame preview:(only show the first 5 rows and 15 columns)
  image_id  healthy  multiple_diseases  rust  scab
0  Train_0        0                  0     1     0
1  Train_1        1                  0     0     0
2  Train_2        0                  0     1     0
3  Train_3        1                  0     0     0
4  Train_4        0                  0     1     0

"""

import json
import os
import reprlib
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Optional, Set, Tuple, Union

import humanize
import pandas as pd
from pandas.api.types import is_numeric_dtype

from rdagent.log import rdagent_logger as logger

# these files are treated as code (e.g. markdown wrapped)
code_files = {".py", ".sh", ".yaml", ".yml", ".md", ".html", ".xml", ".log", ".rst"}
# we treat these files as text (rather than binary) files
plaintext_files = {".txt", ".csv", ".json", ".tsv"} | code_files
# system-generated directories/files to filter out
system_names = {"__MACOSX", ".DS_Store", "Thumbs.db"}


class FileTreeGenerationError(Exception):
    """File tree generation related errors"""

    pass


class MaxLinesExceededError(FileTreeGenerationError):
    """Raised when max lines limit is exceeded"""

    pass


class DirectoryPermissionError(FileTreeGenerationError):
    """Raised when directory access is denied"""

    pass


def get_file_len_size(f: Path) -> Tuple[int, str]:
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


def preview_df(df: pd.DataFrame, file_name: str, simple=True, show_nan_columns=False) -> str:
    """Generate a textual preview of a dataframe"""
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


class FileTreeGenerator:
    """
    Smart file tree generator with symlink handling and intelligent truncation.
    """

    def __init__(self, max_lines: int = 200, priority_files: Set[str] = None, hide_base_name: bool = True):
        """
        Initialize the file tree generator.

        Args:
            max_lines: Maximum output lines to prevent overly long output
            priority_files: File extensions to prioritize for display
        """
        self.max_lines = max_lines
        self.priority_files = priority_files or {".csv", ".json", ".parquet", ".md", ".txt"}
        self.lines = []
        self.line_count = 0
        self.hide_base_name = hide_base_name

    def generate_tree(self, path: Union[str, Path]) -> str:
        """
        Generate a tree structure of files in a directory.

        Args:
            path: Target directory path

        Returns:
            str: Tree structure representation

        Raises:
            FileTreeGenerationError: If tree generation fails
        """
        try:
            path = Path(path)
            base_path = path.resolve()
            self.lines = []
            self.line_count = 0
            self._add_line(f"{'.' if self.hide_base_name else path.name}/")
            self._process_directory(path, 0, "", base_path)
        except MaxLinesExceededError:
            pass  # Expected when hitting line limit
        except Exception as e:
            raise FileTreeGenerationError(f"Failed to generate tree for {path}: {str(e)}") from e

        # CORNER CASE HANDLING: Always check if we hit the limit and add truncation notice if needed
        #
        # WHY THIS IS NECESSARY:
        # The code uses a "mixed exception handling strategy":
        # - Sub-methods (_process_subdirectories, _process_files, _process_single_directory)
        #   catch MaxLinesExceededError and handle it silently (don't re-raise)
        # - This means some MaxLinesExceededError exceptions never propagate to generate_tree
        #
        # CORNER CASE SCENARIO:
        # 1. _add_line() is called and line_count reaches max_lines
        # 2. _add_line() throws MaxLinesExceededError
        # 3. A sub-method catches the exception but doesn't re-raise it (silent handling)
        # 4. The exception never reaches generate_tree's except block above
        # 5. OLD VERSION: No truncation notice is added → User doesn't know content was truncated
        # 6. NEW VERSION: This check below ensures truncation notice is always added
        #
        # DEMONSTRATION EXAMPLE (max_lines=5, processing 6 files):
        #
        # 🔴 OLD VERSION RESULT:
        # project/
        # ├── file1.csv
        # ├── file2.csv
        # ├── file3.csv
        # ├── file4.csv
        # 🔍 Truncation notice? NO → User doesn't know content was truncated!
        #
        # 🔵 NEW VERSION RESULT:
        # project/
        # ├── file1.csv
        # ├── file2.csv
        # ├── file3.csv
        # ├── file4.csv
        # ... (display limited)
        # 🔍 Truncation notice? YES → User knows content was truncated!
        #
        # The key difference:
        # - OLD: Relies on exception propagation (fails when sub-methods handle silently)
        # - NEW: Active check ensures truncation notice is always present
        if self.line_count >= self.max_lines and (
            not self.lines or not self.lines[-1].startswith("... (display limited")
        ):
            self.lines.append("... (display limited, please increase max_lines parameter)")

        return "\n".join(self.lines)

    def _add_line(self, text: str) -> None:
        """
        Add a line to the output.

        Args:
            text: Line text to add

        Raises:
            MaxLinesExceededError: If max lines limit is exceeded
        """
        if self.line_count >= self.max_lines:
            raise MaxLinesExceededError(f"Exceeded maximum lines limit of {self.max_lines}")
        self.lines.append(text)
        self.line_count += 1

    def _process_directory(self, path: Path, depth: int, prefix: str, base_path: Path) -> None:
        """
        Process a single directory.

        Args:
            path: Directory path to process
            depth: Current depth in the tree
            prefix: Prefix for tree formatting
            base_path: Base path for symlink detection

        Raises:
            DirectoryPermissionError: If directory access is denied
            FileTreeGenerationError: If processing fails
            MaxLinesExceededError: Propagated when line limit is reached
        """
        try:
            # Get directory contents, filter out system files
            items = [p for p in path.iterdir() if not p.name.startswith(".") and p.name not in system_names]
            dirs = sorted([p for p in items if p.is_dir()])
            files = sorted([p for p in items if p.is_file()])

            # Categorize files
            priority_files_list, other_files = self._categorize_files(files)

            # Process subdirectories
            self._process_subdirectories(dirs, depth, prefix, base_path)

            # Process files
            self._process_files(priority_files_list + other_files, depth, prefix)

        except MaxLinesExceededError:
            # Propagate this up so generate_tree can handle it
            raise
        except PermissionError as e:
            raise DirectoryPermissionError(f"Permission denied accessing {path}") from e
        except OSError as e:
            if e.errno == 13:  # Permission denied
                raise DirectoryPermissionError(f"Permission denied accessing {path}") from e
            else:
                raise FileTreeGenerationError(f"Error processing directory {path}: {str(e)}") from e

    def _process_subdirectories(self, dirs: List[Path], depth: int, prefix: str, base_path: Path) -> None:
        """Process subdirectories with proper truncation logic."""
        try:
            if depth == 0 or len(dirs) <= 8:
                # First level or ≤8 items: show all
                for d in dirs:
                    self._process_single_directory(d, depth, prefix, base_path)
            else:
                # Not first level and >8 items: show first 2
                show_count = 2
                for d in dirs[:show_count]:
                    self._process_single_directory(d, depth, prefix, base_path)

                # Show remaining directory count
                remaining = len(dirs) - show_count
                self._add_line(f"{prefix}├── ... (+{remaining} more directories)")
        except MaxLinesExceededError:
            # If we hit the line limit, just stop processing
            pass

    def _process_single_directory(self, d: Path, depth: int, prefix: str, base_path: Path) -> None:
        """Process a single directory entry."""
        try:
            # Handle symlinks
            if d.is_symlink():
                target = d.resolve()
                if str(target).startswith(str(base_path)):
                    # avoid recursing into symlinks pointing inside base path
                    self._add_line(
                        f"{prefix}├── {d.name}@ -> {os.path.relpath(target, base_path)} (symlinked dir, not expanded)"
                    )
                    return

            self._add_line(f"{prefix}├── {d.name}/")

            # Process subdirectory recursively
            child_prefix = prefix + "│   "
            self._process_directory(d, depth + 1, child_prefix, base_path)
        except MaxLinesExceededError:
            # If we hit the line limit, just stop processing this directory
            pass

    def _process_files(self, all_files: List[Path], depth: int, prefix: str) -> None:
        """Process files with proper truncation logic."""
        try:
            if depth == 0 or len(all_files) <= 8:
                # First level or ≤8 items: show all
                for f in all_files:
                    self._add_line(f"{prefix}├── {f.name} ({self._get_size_str(f)})")
            else:
                # Not first level and >8 items: show first 2
                show_count = 2
                for f in all_files[:show_count]:
                    self._add_line(f"{prefix}├── {f.name} ({self._get_size_str(f)})")

                # Show remaining file count
                remaining = len(all_files) - show_count
                self._add_line(f"{prefix}├── ... (+{remaining} more files)")
        except MaxLinesExceededError:
            # If we hit the line limit, just stop processing files
            pass

    def _categorize_files(self, files: List[Path]) -> Tuple[List[Path], List[Path]]:
        """Categorize files into priority and other groups."""
        priority = []
        other = []

        for f in files:
            if f.suffix.lower() in self.priority_files:
                priority.append(f)
            else:
                other.append(f)

        # Sort priority files by size (larger files first)
        priority.sort(key=lambda x: x.stat().st_size if x.exists() else 0, reverse=True)

        return priority, other

    def _get_size_str(self, file_path: Path) -> str:
        """Get file size string."""
        try:
            size = file_path.stat().st_size
            return humanize.naturalsize(size)
        except (OSError, FileNotFoundError):
            return "? B"


class DataFolderDescriptor:
    """
    Generate detailed descriptions of data folders including file previews.
    """

    def __init__(self, tree_generator: FileTreeGenerator = None):
        """
        Initialize the data folder descriptor.

        Args:
            tree_generator: Optional FileTreeGenerator instance
        """
        self.tree_generator = tree_generator or FileTreeGenerator()

    def describe_folder(
        self,
        base_path: Union[str, Path],
        include_file_details: bool = True,
        simple: bool = False,
        show_nan_columns: bool = False,
        max_length: int = 10000,
    ) -> str:
        """
        Generate a textual preview of a directory, including an overview of the directory
        structure and previews of individual files.
        """
        base_path = Path(base_path)

        tree = f"## File tree:\n```\n{self.tree_generator.generate_tree(base_path)}```"
        out = [tree]

        if include_file_details:
            out.append("\n## File details:")

            # Intelligently select a subset of files to preview
            files_to_preview = self._select_files_for_preview(base_path)
            out.append(f" (Showing details for representative files out of many)")

            for fn in files_to_preview:
                try:
                    file_name = str(fn.relative_to(base_path))
                except ValueError:
                    file_name = str(fn)

                try:
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
                except Exception as e:
                    out.append(f"-> {file_name}: Error reading file - {str(e)[:100]}")

                if len("\n\n".join(out)) > max_length:
                    out.append("\n... (File details truncated due to max_length)")
                    break

        result = "\n\n".join(out)

        # if the result is very long we generate a simpler version
        if len(result) > max_length and not simple:
            return self.describe_folder(
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

    def _select_files_for_preview(
        self, base_path: Path, max_files_per_group: int = 1, threshold: int = 10
    ) -> List[Path]:
        """
        Intelligently select a representative subset of files for detailed preview.
        If a directory has more than `threshold` files of the same type, only `max_files_per_group` are selected.
        """
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


# Convenience functions for backward compatibility
def file_tree_v2(path: Union[str, Path], max_lines: int = 200, priority_files: Set[str] = None) -> str:
    """Generate a file tree using FileTreeGenerator."""
    generator = FileTreeGenerator(max_lines=max_lines, priority_files=priority_files)
    return generator.generate_tree(path)


def describe_data_folder_v2(
    base_path: Union[str, Path],
    include_file_details: bool = True,
    simple: bool = False,
    show_nan_columns: bool = False,
    max_length: int = 10000,
) -> str:
    """Generate a data folder description using DataFolderDescriptor."""
    descriptor = DataFolderDescriptor()
    return descriptor.describe_folder(
        base_path,
        include_file_details=include_file_details,
        simple=simple,
        show_nan_columns=show_nan_columns,
        max_length=max_length,
    )
