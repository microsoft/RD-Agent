import io
import re
import shutil
import random
from pathlib import Path

import pandas as pd

# render it with jinja
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.utils.env import QTDockerEnv


def generate_data_folder_from_qlib():
    template_path = Path(__file__).parent / "factor_data_template"
    qtde = QTDockerEnv()
    qtde.prepare()

    # Run the Qlib backtest
    execute_log = qtde.run(
        local_path=str(template_path),
        entry=f"python generate.py",
    )

    assert (
        Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5"
    ).exists(), "daily_pv_all.h5 is not generated."
    assert (
        Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5"
    ).exists(), "daily_pv_debug.h5 is not generated."

    Path(FACTOR_COSTEER_SETTINGS.data_folder).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "daily_pv_all.h5",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder) / "README.md",
    )

    Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).mkdir(parents=True, exist_ok=True)
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "daily_pv_debug.h5",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "daily_pv.h5",
    )
    shutil.copy(
        Path(__file__).parent / "factor_data_template" / "README.md",
        Path(FACTOR_COSTEER_SETTINGS.data_folder_debug) / "README.md",
    )


def get_file_desc(p: Path, variable_list=[]) -> str:
    """
    Get the description of a file based on its type.

    Parameters
    ----------
    p : Path
        The path of the file.

    Returns
    -------
    str
        The description of the file.
    """
    p = Path(p)

    JJ_TPL = Environment(undefined=StrictUndefined).from_string(
        """
# {{file_name}}

## File Type
{{type_desc}}

## Content Overview
{{content}}
"""
    )

    if p.name.endswith(".h5"):
        df = pd.read_hdf(p)
        # get df.head() as string with full width
        pd.set_option("display.max_columns", None)  # or 1000
        pd.set_option("display.max_rows", None)  # or 1000
        pd.set_option("display.max_colwidth", None)  # or 199

        # Basic information
        df_info = "### Data Structure\n"
        if isinstance(df.index, pd.MultiIndex):
            df_info += f"- Index: MultiIndex with levels {df.index.names}\n"
        else:
            df_info += f"- Index: {df.index.name}\n"
        
        # Column information
        df_info += "\n### Columns\n"
        columns = df.dtypes.to_dict()
        
        # Group columns by their prefixes if they exist
        grouped_columns = {}
        for col in columns:
            if col.startswith('$'):
                prefix = col.split('_')[0] if '_' in col else col
                if prefix not in grouped_columns:
                    grouped_columns[prefix] = []
                grouped_columns[prefix].append(col)
            else:
                if 'other' not in grouped_columns:
                    grouped_columns['other'] = []
                grouped_columns['other'].append(col)

        if variable_list:
            df_info += "#### Relevant Columns:\n"
            for col in variable_list:
                if col in columns:
                    df_info += f"- {col}: {columns[col]}\n"
        else:
            df_info += "#### All Columns:\n"
            # Convert grouped_columns to list of tuples and shuffle
            grouped_items = list(grouped_columns.items())
            random.shuffle(grouped_items)
            for prefix, cols in grouped_items:
                if prefix == 'other':
                    df_info += "\n#### Other Columns:\n"
                else:
                    df_info += f"\n#### {prefix} Related Columns:\n"
                # Shuffle columns within each group
                random.shuffle(cols)
                for col in cols:
                    df_info += f"- {col}: {columns[col]}\n"

        # Sample data if available
        if "REPORT_PERIOD" in df.columns:
            one_instrument = df.index.get_level_values("instrument")[0]
            df_on_one_instrument = df.loc[pd.IndexSlice[:, one_instrument], ["REPORT_PERIOD"]]
            df_info += "\n### Sample Data\n"
            df_info += f"Showing data for instrument {one_instrument}:\n"
            df_info += str(df_on_one_instrument.head(5))

        return JJ_TPL.render(
            file_name=p.name,
            type_desc="HDF5 Data File",
            content=df_info,
        )
    elif p.name.endswith(".md"):
        with open(p) as f:
            content = f.read()
            return JJ_TPL.render(
                file_name=p.name,
                type_desc="Markdown Documentation",
                content=content,
            )
    else:
        raise NotImplementedError(
            f"file type {p.name} is not supported. Please implement its description function.",
        )


def get_data_folder_intro(fname_reg: str = ".*", flags=0, variable_mapping=None) -> str:
    """
    Directly get the info of the data folder.
    It is for preparing prompting message.

    Parameters
    ----------
    fname_reg : str
        a regular expression to filter the file name.

    flags: str
        flags for re.match

    Returns
    -------
        str
            The description of the data folder.
    """

    if (
        not Path(FACTOR_COSTEER_SETTINGS.data_folder).exists()
        or not Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).exists()
    ):
        # FIXME: (xiao) I think this is writing in a hard-coded way.
        # get data folder intro does not imply that we are generating the data folder.
        generate_data_folder_from_qlib()
    content_l = []
    for p in Path(FACTOR_COSTEER_SETTINGS.data_folder_debug).iterdir():
        if re.match(fname_reg, p.name, flags) is not None:
            if variable_mapping:
                content_l.append(get_file_desc(p, variable_mapping.get(p.stem, [])))
            else:
                content_l.append(get_file_desc(p))
    return "\n----------------- file splitter -------------\n".join(content_l)