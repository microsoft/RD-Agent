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


class RDAgentException(Exception):  # noqa: N818
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
