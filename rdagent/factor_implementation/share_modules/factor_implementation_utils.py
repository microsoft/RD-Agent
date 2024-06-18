from pathlib import Path

import pandas as pd

# render it with jinja
from jinja2 import Template
from rdagent.factor_implementation.share_modules.factor_implementation_config import FACTOR_IMPLEMENT_SETTINGS
from rdagent.factor_implementation.evolving.factor import FactorImplementTask


TPL = """
{{file_name}}
```{{type_desc}}
{{content}}
````
"""
# Create a Jinja template from the string
JJ_TPL = Template(TPL)


def get_data_folder_intro():
    """Direclty get the info of the data folder.
    It is for preparing prompting message.
    """
    content_l = []
    for p in Path(FACTOR_IMPLEMENT_SETTINGS.file_based_execution_data_folder).iterdir():
        if p.name.endswith(".h5"):
            df = pd.read_hdf(p)
            # get  df.head() as string with full width
            pd.set_option("display.max_columns", None)  # or 1000
            pd.set_option("display.max_rows", None)  # or 1000
            pd.set_option("display.max_colwidth", None)  # or 199
            rendered = JJ_TPL.render(
                file_name=p.name,
                type_desc="generated by `pd.read_hdf(filename).head()`",
                content=df.head().to_string(),
            )
            content_l.append(rendered)
        elif p.name.endswith(".md"):
            with open(p) as f:
                content = f.read()
                rendered = JJ_TPL.render(
                    file_name=p.name,
                    type_desc="markdown",
                    content=content,
                )
                content_l.append(rendered)
        else:
            raise NotImplementedError(
                f"file type {p.name} is not supported. Please implement its description function.",
            )
    return "\n ----------------- file spliter -------------\n".join(content_l)
