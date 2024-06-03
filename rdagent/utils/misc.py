"""Miscellaneous utility functions
"""
from __future__ import annotations

import multiprocessing as mp
from collections.abc import Callable


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
