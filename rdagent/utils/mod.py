"""
Here are utils to import things and modules

"""
import importlib
from typing import Any


def import_class(class_path: str) -> Any:
    """
    Parameters
    ----------
    class_path : str
        class path like"scripts.factor_implementation.baselines.naive.one_shot.OneshotFactorGen"

    Returns
    -------
        class of `class_path`
    """
    # 可以参考qlib
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)
