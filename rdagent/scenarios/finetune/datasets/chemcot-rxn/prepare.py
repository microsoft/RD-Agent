"""Prepare function for chemcot-rxn dataset."""

from datasets import Dataset


def prepare(ds: Dataset) -> Dataset:
    """Unify schema for ChemCoT RXN dataset.

    rcr.json uses 'cot_result' while fs_*.json uses 'struct_cot'.
    This function renames 'cot_result' to 'struct_cot' for consistency.
    """
    if "cot_result" in ds.column_names and "struct_cot" not in ds.column_names:
        ds = ds.rename_column("cot_result", "struct_cot")
    return ds
