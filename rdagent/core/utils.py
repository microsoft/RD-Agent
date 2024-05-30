from __future__ import annotations

import importlib
import json
import multiprocessing as mp
import os
import random
import string
from collections.abc import Callable
from pathlib import Path
from typing import Any

import yaml
from fuzzywuzzy import fuzz


class FincoException(Exception):
    pass


class SingletonMeta(type):
    _instance = None

    def __call__(cls, *args, **kwargs):
        if cls._instance is None:
            cls._instance = super(SingletonMeta, cls).__call__(*args, **kwargs)
        return cls._instance


class SingletonBaseClass(metaclass=SingletonMeta):
    """
    Because we try to support defining Singleton with `class A(SingletonBaseClass)` instead of `A(metaclass=SingletonMeta)`
    This class becomes necessary

    """

    # TODO: Add move this class to Qlib's general utils.


def parse_json(response):
    try:
        return json.loads(response)
    except json.decoder.JSONDecodeError:
        pass

    raise Exception(f"Failed to parse response: {response}, please report it or help us to fix it.")


def similarity(text1, text2):
    text1 = text1 if isinstance(text1, str) else ""
    text2 = text2 if isinstance(text2, str) else ""

    # Maybe we can use other similarity algorithm such as tfidf
    return fuzz.ratio(text1, text2)


def random_string(length=10):
    letters = string.ascii_letters + string.digits
    return "".join(random.choice(letters) for i in range(length))


def remove_uncommon_keys(new_dict, org_dict):
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


def crawl_the_folder(folder_path: Path):
    yaml_files = []
    for root, _, files in os.walk(folder_path.as_posix()):
        for file in files:
            if file.endswith(".yaml") or file.endswith(".yml"):
                yaml_file_path = Path(os.path.join(root, file)).relative_to(folder_path)
                yaml_files.append(yaml_file_path.as_posix())
    return sorted(yaml_files)


def compare_yaml(file1, file2):
    with open(file1) as stream:
        data1 = yaml.safe_load(stream)
    with open(file2) as stream:
        data2 = yaml.safe_load(stream)
    return data1 == data2


def remove_keys(valid_keys, ori_dict):
    for key in list(ori_dict.keys()):
        if key not in valid_keys:
            ori_dict.pop(key)
    return ori_dict


class YamlConfigCache(SingletonBaseClass):
    def __init__(self) -> None:
        super().__init__()
        self.path_to_config = dict()

    def load(self, path):
        with open(path) as stream:
            data = yaml.safe_load(stream)
            self.path_to_config[path] = data

    def __getitem__(self, path):
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
