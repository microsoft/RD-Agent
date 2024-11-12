from __future__ import annotations

import functools
import importlib
import json
import multiprocessing as mp
import pickle
import random
from collections.abc import Callable
from pathlib import Path
from typing import Any, ClassVar, NoReturn, cast

from filelock import FileLock
from fuzzywuzzy import fuzz  # type: ignore[import-untyped]

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.oai.llm_conf import LLM_SETTINGS


class RDAgentException(Exception):  # noqa: N818
    pass


class SingletonBaseClass:
    """
    Because we try to support defining Singleton with `class A(SingletonBaseClass)`
    instead of `A(metaclass=SingletonMeta)` this class becomes necessary.
    """

    _instance_dict: ClassVar[dict] = {}

    def __new__(cls, *args: Any, **kwargs: Any) -> Any:
        # Since it's hard to align the difference call using args and kwargs, we strictly ask to use kwargs in Singleton
        if args:
            # TODO: this restriction can be solved.
            exception_message = "Please only use kwargs in Singleton to avoid misunderstanding."
            raise RDAgentException(exception_message)
        class_name = [(-1, f"{cls.__module__}.{cls.__name__}")]
        args_l = [(i, args[i]) for i in args]
        kwargs_l = sorted(kwargs.items())
        all_args = class_name + args_l + kwargs_l
        kwargs_hash = hash(tuple(all_args))
        if kwargs_hash not in cls._instance_dict:
            cls._instance_dict[kwargs_hash] = super().__new__(cls)  # Corrected call
        return cls._instance_dict[kwargs_hash]

    def __reduce__(self) -> NoReturn:
        """
        NOTE:
        When loading an object from a pickle, the __new__ method does not receive the `kwargs`
        it was initialized with. This makes it difficult to retrieve the correct singleton object.
        Therefore, we have made it unpickable.
        """
        msg = f"Instances of {self.__class__.__name__} cannot be pickled"
        raise pickle.PicklingError(msg)


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
    return cast(int, fuzz.ratio(text1, text2))  # mypy does not reguard it as int


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


class CacheSeedGen:
    """
    It is a global seed generator to generate a sequence of seeds.
    This will support the feature `use_auto_chat_cache_seed_gen` claim

    NOTE:
    - This seed is specifically for the cache and is different from a regular seed.
    - If the cache is removed, setting the same seed will not produce the same QA trace.
    """

    def __init__(self) -> None:
        self.set_seed(LLM_SETTINGS.init_chat_cache_seed)

    def set_seed(self, seed: int) -> None:
        random.seed(seed)

    def get_next_seed(self) -> int:
        """generate next random int"""
        return random.randint(0, 10000)  # noqa: S311


LLM_CACHE_SEED_GEN = CacheSeedGen()


def _subprocess_wrapper(f: Callable, seed: int, args: list) -> Any:
    """
    It is a function wrapper. To ensure the subprocess has a fixed start seed.
    """

    LLM_CACHE_SEED_GEN.set_seed(seed)
    return f(*args)


def multiprocessing_wrapper(func_calls: list[tuple[Callable, tuple]], n: int) -> list:
    """It will use multiprocessing to call the functions in func_calls with the given parameters.
    The results equals to `return  [f(*args) for f, args in func_calls]`
    It will not call multiprocessing if `n=1`

    NOTE:
    We coooperate with chat_cache_seed feature
    We ensure get the same seed trace even we have multiple number of seed

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
    if n == 1 or max(1, min(n, len(func_calls))) == 1:
        return [f(*args) for f, args in func_calls]

    with mp.Pool(processes=max(1, min(n, len(func_calls)))) as pool:
        results = [
            pool.apply_async(_subprocess_wrapper, args=(f, LLM_CACHE_SEED_GEN.get_next_seed(), args))
            for f, args in func_calls
        ]
        return [result.get() for result in results]


def cache_with_pickle(hash_func: Callable, post_process_func: Callable | None = None) -> Callable:
    """
    This decorator will cache the return value of the function with pickle.
    The cache key is generated by the hash_func. The hash function returns a string or None.
    If it returns None, the cache will not be used. The cache will be stored in the folder
    specified by RD_AGENT_SETTINGS.pickle_cache_folder_path_str with name hash_key.pkl.
    The post_process_func will be called with the original arguments and the cached result
    to give each caller a chance to process the cached result. The post_process_func should
    return the final result.
    """

    def cache_decorator(func: Callable) -> Callable:
        @functools.wraps(func)
        def cache_wrapper(*args: Any, **kwargs: Any) -> Any:
            if not RD_AGENT_SETTINGS.cache_with_pickle:
                return func(*args, **kwargs)

            target_folder = Path(RD_AGENT_SETTINGS.pickle_cache_folder_path_str) / f"{func.__module__}.{func.__name__}"
            target_folder.mkdir(parents=True, exist_ok=True)
            hash_key = hash_func(*args, **kwargs)

            if hash_key is None:
                return func(*args, **kwargs)

            cache_file = target_folder / f"{hash_key}.pkl"
            lock_file = target_folder / f"{hash_key}.lock"

            if cache_file.exists():
                with cache_file.open("rb") as f:
                    cached_res = pickle.load(f)
                return post_process_func(*args, cached_res=cached_res, **kwargs) if post_process_func else cached_res

            if RD_AGENT_SETTINGS.use_file_lock:
                with FileLock(lock_file):
                    result = func(*args, **kwargs)
            else:
                result = func(*args, **kwargs)

            with cache_file.open("wb") as f:
                pickle.dump(result, f)

            return result

        return cache_wrapper

    return cache_decorator
