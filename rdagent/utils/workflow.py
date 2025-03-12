"""
This is a class that try to store/resume/traceback the workflow session


Postscripts:
- Originally, I want to implement it in a more general way with python generator.
  However, Python generator is not picklable (dill does not support pickle as well)

"""

import datetime
import pickle
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, TypeVar, Union, cast

from tqdm.auto import tqdm

from rdagent.log import rdagent_logger as logger


class LoopMeta(type):
    @staticmethod
    def _get_steps(bases: tuple[type, ...]) -> list[str]:
        """
        Recursively get all the `steps` from the base classes and combine them into a single list.

        Args:
            bases (tuple): A tuple of base classes.

        Returns:
            List[Callable]: A list of steps combined from all base classes.
        """
        steps = []
        for base in bases:
            for step in LoopMeta._get_steps(base.__bases__) + getattr(base, "steps", []):
                if step not in steps:
                    steps.append(step)
        return steps

    def __new__(mcs, clsname: str, bases: tuple[type, ...], attrs: dict[str, Any]) -> Any:
        """
        Create a new class with combined steps from base classes and current class.

        Args:
            clsname (str): Name of the new class.
            bases (tuple): Base classes.
            attrs (dict): Attributes of the new class.

        Returns:
            LoopMeta: A new instance of LoopMeta.
        """
        steps = LoopMeta._get_steps(bases)  # all the base classes of parents
        for name, attr in attrs.items():
            if not name.startswith("_") and callable(attr):
                if name not in steps:
                    # NOTE: if we override the step in the subclass
                    # Then it is not the new step. So we skip it.
                    steps.append(name)
        attrs["steps"] = steps
        return super().__new__(mcs, clsname, bases, attrs)


@dataclass
class LoopTrace:
    start: datetime.datetime  # the start time of the trace
    end: datetime.datetime  # the end time of the trace
    step_idx: int
    # TODO: more information about the trace


class LoopBase:
    """
    Assumption:
    - The last step is responsible for recording information!!!!
    """

    steps: list[str]  # a list of steps to work on
    loop_trace: dict[int, list[LoopTrace]]

    skip_loop_error: tuple[type[BaseException], ...] = ()  # you can define a list of error that will skip current loop

    EXCEPTION_KEY = "_EXCEPTION"

    def __init__(self) -> None:
        self.loop_idx = 0  # current loop index
        self.step_idx = 0  # the index of next step to be run
        self.loop_prev_out: dict[str, Any] = {}  # the step results of current loop
        self.loop_trace = defaultdict(list[LoopTrace])  # the key is the number of loop
        self.session_folder = logger.log_trace_path / "__session__"

    def run(self, step_n: int | None = None, loop_n: int | None = None) -> None:
        """

        Parameters
        ----------
        step_n : int | None
            How many steps to run;
            `None` indicates to run forever until error or KeyboardInterrupt
        loop_n: int | None
            How many steps to run; if current loop is incomplete, it will be counted as the first loop for completion
            `None` indicates to run forever until error or KeyboardInterrupt
        """
        with tqdm(total=len(self.steps), desc="Workflow Progress", unit="step") as pbar:
            while True:
                if step_n is not None:
                    if step_n <= 0:
                        break
                    step_n -= 1
                if loop_n is not None:
                    if loop_n <= 0:
                        break

                li, si = self.loop_idx, self.step_idx
                name = self.steps[si]
                logger.info(f"Start Loop {li}, Step {si}: {name}")
                with logger.tag(f"Loop_{li}.{name}"):
                    start = datetime.datetime.now(datetime.timezone.utc)
                    func: Callable[..., Any] = cast(Callable[..., Any], getattr(self, name))
                    try:
                        self.loop_prev_out[name] = func(self.loop_prev_out)
                        # TODO: Fix the error logger.exception(f"Skip loop {li} due to {e}")
                    except Exception as e:
                        if isinstance(e, self.skip_loop_error):
                            # FIXME: This does not support previous demo (due to their last step is not for recording)
                            logger.warning(f"Skip loop {li} due to {e}")
                            # NOTE: strong assumption!  The last step is responsible for recording information
                            self.step_idx = len(self.steps) - 1  # directly jump to the last step.
                            self.loop_prev_out[self.EXCEPTION_KEY] = e
                            continue
                        else:
                            raise
                    finally:
                        # make sure failure steps are displayed correclty
                        end = datetime.datetime.now(datetime.timezone.utc)
                        self.loop_trace[li].append(LoopTrace(start, end, step_idx=si))

                        # Update tqdm progress bar directly to step_idx
                        pbar.n = si + 1
                        pbar.set_postfix(
                            loop_index=li, step_index=si + 1, step_name=name
                        )  # step_name indicate  last finished step_name

                # index increase and save session
                self.step_idx = (self.step_idx + 1) % len(self.steps)
                if self.step_idx == 0:  # reset to step 0 in next round
                    self.loop_idx += 1
                    if loop_n is not None:
                        loop_n -= 1
                    self.loop_prev_out = {}
                    pbar.reset()  # reset the progress bar for the next loop

                self.dump(self.session_folder / f"{li}" / f"{si}_{name}")  # save a snapshot after the session

    def dump(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(
        cls, path: Union[str, Path], output_path: Optional[Union[str, Path]] = None, do_truncate: bool = False
    ) -> "LoopBase":
        path = Path(path)
        with path.open("rb") as f:
            session = cast(LoopBase, pickle.load(f))

        # set session folder
        if output_path:
            output_path = Path(output_path)
            output_path.mkdir(parents=True, exist_ok=True)
            session.session_folder = output_path / "__session__"

        # set trace path
        logger.set_trace_path(session.session_folder.parent)

        # truncate future message
        if do_truncate:
            max_loop = max(session.loop_trace.keys())
            logger.storage.truncate(time=session.loop_trace[max_loop][-1].end)
        return session


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
