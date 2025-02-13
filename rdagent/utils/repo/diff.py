import difflib
import fnmatch
from pathlib import Path


def generate_diff(dir1: str, dir2: str, file_pattern: str = "*.py") -> list[str]:
    """
    Generate a diff between two directories (from dir1 to dir2) using files that match the specified file pattern.
    This function mimics the behavior of `diff -durN dir1 dir2` in Linux.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.
        file_pattern (str, optional): Glob pattern to filter files. Defaults to "*.py".

    Returns:
        list[str]: A list of diffs for files that differ between the two directories.
    """

    dir1_files = {f.relative_to(dir1) for f in Path(dir1).rglob(file_pattern) if f.is_file()}
    dir2_files = {f.relative_to(dir2) for f in Path(dir2).rglob(file_pattern) if f.is_file()}

    all_files = dir1_files.union(dir2_files)
    file_dict1 = {}
    file_dict2 = {}
    for file in all_files:
        file1 = Path(dir1) / file
        file2 = Path(dir2) / file
        if file1.exists():
            with file1.open() as f1:
                file_dict1[str(file)] = f1.read()
        else:
            file_dict1[str(file)] = ""
        if file2.exists():
            with file2.open() as f2:
                file_dict2[str(file)] = f2.read()
        else:
            file_dict2[str(file)] = ""
    return generate_diff_from_dict(file_dict1, file_dict2, file_pattern="*")


def generate_diff_from_dict(file_dict1: dict, file_dict2: dict, file_pattern: str = "*.py") -> list[str]:
    """
    Generate a diff between two dictionaries of file contents.
    The dictionaries should be of the format {file_path: file_content}.

    Returns:
        List[str]: A list of diffs for files that are different between the two dictionaries.
    """
    diff_files = []
    all_files = set(file_dict1.keys()).union(file_dict2.keys())
    for file in sorted(all_files):
        if not fnmatch.fnmatch(file, file_pattern):
            continue
        content1 = file_dict1.get(file, "")
        content2 = file_dict2.get(file, "")
        diff = list(
            difflib.unified_diff(
                content1.splitlines(keepends=True),
                content2.splitlines(keepends=True),
                fromfile=file if file in file_dict1 else file + " (empty file)",
                tofile=file if file in file_dict2 else file + " (empty file)",
            )
        )
        if diff:
            diff_files.extend(diff)
    return diff_files
