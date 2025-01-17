import difflib
from pathlib import Path
from typing import List


def generate_diff(dir1: str, dir2: str) -> List[str]:
    """
    Generate a diff between two directories(from dir1 to dir2), considering only .py files.
    It is mocking `diff -durN dir1 dir2` in linux.

    Args:
        dir1 (str): Path to the first directory.
        dir2 (str): Path to the second directory.

    Returns:
        List[str]: A list of diffs for .py files that are different between the two directories.
    """

    diff_files = []

    dir1_files = {f.relative_to(dir1) for f in Path(dir1).rglob("*.py") if f.is_file()}
    dir2_files = {f.relative_to(dir2) for f in Path(dir2).rglob("*.py") if f.is_file()}

    all_files = dir1_files.union(dir2_files)

    for file in all_files:
        file1 = Path(dir1) / file
        file2 = Path(dir2) / file

        if file1.exists() and file2.exists():
            with file1.open() as f1, file2.open() as f2:
                diff = list(
                    difflib.unified_diff(f1.readlines(), f2.readlines(), fromfile=str(file1), tofile=str(file2))
                )
                if diff:
                    diff_files.extend(diff)
        else:
            if file1.exists():
                with file1.open() as f1:
                    diff = list(
                        difflib.unified_diff(
                            f1.readlines(), [], fromfile=str(file1), tofile=str(file2) + " (empty file)"
                        )
                    )
                    diff_files.extend(diff)
            elif file2.exists():
                with file2.open() as f2:
                    diff = list(
                        difflib.unified_diff(
                            [], f2.readlines(), fromfile=str(file1) + " (empty file)", tofile=str(file2)
                        )
                    )
                    diff_files.extend(diff)

    return diff_files
