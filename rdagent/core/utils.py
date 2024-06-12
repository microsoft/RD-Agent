from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import random
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar

import yaml
from fuzzywuzzy import fuzz


class RDAgentException(Exception): # noqa: N818
    pass


class SingletonMeta(type):
    _instance_dict: ClassVar[dict] = {}

    def __call__(cls, *args: Any, **kwargs: Any) -> Any:
        # Since it's hard to align the difference call using args and kwargs, we strictly ask to use kwargs in Singleton
        if args:
            exception_message = "Please only use kwargs in Singleton to avoid misunderstanding."
            raise RDAgentException(exception_message)
        kwargs_hash = hash(tuple(sorted(kwargs.items())))
        if kwargs_hash not in cls._instance_dict:
            cls._instance_dict[kwargs_hash] = super().__call__(**kwargs)
        return cls._instance_dict[kwargs_hash]

class SingletonBaseClass(metaclass=SingletonMeta):
    """
    Because we try to support defining Singleton with `class A(SingletonBaseClass)`
    instead of `A(metaclass=SingletonMeta)` this class becomes necessary.
    """

    # TODO: Add move this class to Qlib's general utils.


def parse_json(response: str) -> Any:
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        pass
    error_message = f"Failed to parse response: {response}, please report it or help us to fix it."
    raise ValueError(error_message)


def similarity(text1: str, text2: str) -> int:
    text1 = text1 if isinstance(text1, str) else ""
    text2 = text2 if isinstance(text2, str) else ""

    # Maybe we can use other similarity algorithm such as tfidf
    return fuzz.ratio(text1, text2)


def random_string(length: int = 10) -> str:
    letters = string.ascii_letters + string.digits
    return "".join(random.SystemRandom().choice(letters) for _ in range(length))


def remove_uncommon_keys(new_dict: dict, org_dict: dict) -> None:
    keys_to_remove = []

    for key in new_dict:
        if key not in org_dict:
            keys_to_remove.append(key)
        elif isinstance(new_dict[key], dict) and isinstance(org_dict[key], dict):
            remove_uncommon_keys(new_dict[key], org_dict[key])
        elif isinstance(new_dict[key], dict) and isinstance(org_dict[key], str):
            new_dict[key] = org_dict[key]

    for key in keys_to_remove:
        del new_dict[key]


def crawl_the_folder(folder_path: Path) -> list:
    yaml_files = []
    for root, _, files in os.walk(folder_path.as_posix()):
        for file in files:
            if file.endswith((".yaml", ".yml")):
                yaml_file_path = Path(root) / file
                yaml_files.append(str(yaml_file_path.relative_to(folder_path)))
    return sorted(yaml_files)


def compare_yaml(file1: Path | str, file2: Path | str) -> bool:
    with Path(file1).open() as stream:
        data1 = yaml.safe_load(stream)
    with Path(file2).open() as stream:
        data2 = yaml.safe_load(stream)
    return data1 == data2


def remove_keys(valid_keys: set[Any], ori_dict: dict[Any, Any]) -> dict[Any, Any]:
    for key in list(ori_dict.keys()):
        if key not in valid_keys:
            ori_dict.pop(key)
    return ori_dict


class YamlConfigCache(SingletonBaseClass):
    def __init__(self) -> None:
        super().__init__()
        self.path_to_config = {}

    def load(self, path: str) -> None:
        with Path(path).open() as stream:
            data = yaml.safe_load(stream)
            self.path_to_config[path] = data

    def __getitem__(self, path: str) -> Any:
        if path not in self.path_to_config:
            self.load(path)
        return self.path_to_config[path]


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
    module_path, class_name = class_path.rsplit(".", 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)


def multiprocessing_wrapper(func_calls: list[tuple[Callable, tuple]], n: int) -> list:
    """It will use multiprocessing to call the functions in func_calls with the given parameters.
    The results equals to `return  [f(*args) for f, args in func_calls]`
    It will not call multiprocessing if `n=1`

    Parameters
    ----------
    func_calls : List[Tuple[Callable, Tuple]]
        the list of functions and their parameters
    n : int
        the number of subprocesses

    Returns
    -------
    list

    """
    if n == 1:
        return [f(*args) for f, args in func_calls]
    with mp.Pool(processes=n) as pool:
        results = [pool.apply_async(f, args) for f, args in func_calls]
        return [result.get() for result in results]


# You can test the above function
# def f(x):
#     return x**2
#
# if __name__ == "__main__":
#     print(multiprocessing_wrapper([(f, (i,)) for i in range(10)], 4))
