import time
from collections.abc import Callable
from typing import Any, TypeVar

ASpecificRet = TypeVar("ASpecificRet")


def wait_retry(
    retry_n: int = 3, sleep_time: int = 1, transform_args_fn: Callable[[tuple, dict], tuple[tuple, dict]] | None = None
) -> Callable[[Callable[..., ASpecificRet]], Callable[..., ASpecificRet]]:
    """Decorator to wait and retry the function for retry_n times.

    Example:
    >>> import time
    >>> @wait_retry(retry_n=2, sleep_time=1)
    ... def test_func():
    ...     global counter
    ...     counter += 1
    ...     if counter < 3:
    ...         raise ValueError("Counter is less than 3")
    ...     return counter
    >>> counter = 0
    >>> try:
    ...     test_func()
    ... except ValueError as e:
    ...     print(f"Caught an exception: {e}")
    Error: Counter is less than 3
    Error: Counter is less than 3
    Caught an exception: Counter is less than 3
    >>> counter
    2
    """
    assert retry_n > 0, "retry_n should be greater than 0"

    def decorator(f: Callable[..., ASpecificRet]) -> Callable[..., ASpecificRet]:
        def wrapper(*args: Any, **kwargs: Any) -> ASpecificRet:
            for i in range(retry_n + 1):
                try:
                    return f(*args, **kwargs)
                except Exception as e:
                    print(f"Error: {e}")
                    time.sleep(sleep_time)
                    if i == retry_n:
                        raise
                    # Update args and kwargs using the transform function if provided.
                    if transform_args_fn is not None:
                        args, kwargs = transform_args_fn(args, kwargs)
            else:
                # just for passing mypy CI.
                return f(*args, **kwargs)

        return wrapper

    return decorator
