"""
This is some common utils functions.
it is not binding to the scenarios or framework (So it is not placed in rdagent.core.utils)
"""

# TODO: merge the common utils in `rdagent.core.utils` into this folder
# TODO: split the utils in this module into different modules in the future.

import importlib
import re
import sys
from types import ModuleType
from typing import Union


def get_module_by_module_path(module_path: Union[str, ModuleType]):
    """Load module from path like a/b/c/d.py or a.b.c.d

    :param module_path:
    :return:
    :raises: ModuleNotFoundError
    """
    if module_path is None:
        raise ModuleNotFoundError("None is passed in as parameters as module_path")

    if isinstance(module_path, ModuleType):
        module = module_path
    else:
        if module_path.endswith(".py"):
            module_name = re.sub("^[^a-zA-Z_]+", "", re.sub("[^0-9a-zA-Z_]", "", module_path[:-3].replace("/", "_")))
            module_spec = importlib.util.spec_from_file_location(module_name, module_path)
            module = importlib.util.module_from_spec(module_spec)
            sys.modules[module_name] = module
            module_spec.loader.exec_module(module)
        else:
            module = importlib.import_module(module_path)
    return module


def convert2bool(value: Union[str, bool]) -> bool:
    """
    Motivation: the return value of LLM is not stable. Try to convert the value into bool
    """
    # TODO: if we have more similar functions, we can build a library to converting unstable LLM response to stable results.
    if isinstance(value, str):
        v = value.lower().strip()
        if v in ["true", "yes", "ok"]:
            return True
        if v in ["false", "no"]:
            return False
        raise ValueError(f"Can not convert {value} to bool")
    elif isinstance(value, bool):
        return value
    else:
        raise ValueError(f"Unknown value type {value} to bool")
