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
from typing import Any, Callable, TypeVar, cast

import pytz
from tqdm.auto import tqdm

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer

if RD_AGENT_SETTINGS.enable_mlflow:
    import mlflow


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
                if step not in steps and step not in ["load", "dump"]:  # incase user override the load/dump method
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
                if name not in steps and name not in ["load", "dump"]:  # incase user override the load/dump method
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
    withdraw_loop_error: tuple[
        type[BaseException], ...
    ] = ()  # you can define a list of error that will withdraw current loop

    EXCEPTION_KEY = "_EXCEPTION"

    def __init__(self) -> None:
        self.loop_idx = 0  # current loop index
        self.step_idx = 0  # the index of next step to be run
        self.loop_prev_out: dict[str, Any] = {}  # the step results of current loop
        self.loop_trace = defaultdict(list[LoopTrace])  # the key is the number of loop
        self.session_folder = Path(LOG_SETTINGS.trace_path) / "__session__"
        self.timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer

    def run(self, step_n: int | None = None, loop_n: int | None = None, all_duration: str | None = None) -> None:
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

        if all_duration is not None and not self.timer.started:
            self.timer.reset(all_duration=all_duration)

        with tqdm(total=len(self.steps), desc="Workflow Progress", unit="step") as pbar:
            while True:
                if step_n is not None:
                    if step_n <= 0:
                        break
                    step_n -= 1
                if loop_n is not None:
                    if loop_n <= 0:
                        break

                if RD_AGENT_SETTINGS.enable_mlflow:
                    mlflow.log_metric("loop_index", self.loop_idx)
                    mlflow.log_metric("step_index", self.step_idx)
                    current_local_datetime = datetime.datetime.now(pytz.timezone("Asia/Shanghai"))
                    float_like_datetime = (
                        current_local_datetime.second
                        + current_local_datetime.minute * 1e2
                        + current_local_datetime.hour * 1e4
                        + current_local_datetime.day * 1e6
                        + current_local_datetime.month * 1e8
                        + current_local_datetime.year * 1e10
                    )
                    mlflow.log_metric("current_datetime", float_like_datetime)
                    mlflow.log_metric("api_fail_count", RD_Agent_TIMER_wrapper.api_fail_count)
                    lastest_api_fail_time = RD_Agent_TIMER_wrapper.latest_api_fail_time
                    if lastest_api_fail_time is not None:
                        mlflow.log_metric(
                            "lastest_api_fail_time",
                            (
                                lastest_api_fail_time.second
                                + lastest_api_fail_time.minute * 1e2
                                + lastest_api_fail_time.hour * 1e4
                                + lastest_api_fail_time.day * 1e6
                                + lastest_api_fail_time.month * 1e8
                                + lastest_api_fail_time.year * 1e10
                            ),
                        )

                if self.timer.started:
                    if RD_AGENT_SETTINGS.enable_mlflow:
                        mlflow.log_metric("remain_time", self.timer.remain_time().seconds)  # type: ignore[union-attr]
                        mlflow.log_metric(
                            "remain_percent", self.timer.remain_time() / self.timer.all_duration * 100  # type: ignore[operator]
                        )

                    if self.timer.is_timeout():
                        logger.warning("Timeout, exiting the loop.")
                        break
                    else:
                        logger.info(f"Timer remaining time: {self.timer.remain_time()}")

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
                        elif isinstance(e, self.withdraw_loop_error):
                            logger.warning(f"Withdraw loop {li} due to {e}")
                            # Back to previous loop
                            self.withdraw_loop(li)
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

    def withdraw_loop(self, loop_idx: int) -> None:
        prev_session_dir = self.session_folder / str(loop_idx - 1)
        prev_path = min(
            (p for p in prev_session_dir.glob("*_*") if p.is_file()),
            key=lambda item: int(item.name.split("_", 1)[0]),
            default=None,
        )
        if prev_path:
            loaded = type(self).load(
                prev_path,
                checkout=True,
                replace_timer=True,
            )
            logger.info(f"Load previous session from {prev_path}")
            # Overwrite current instance state
            self.__dict__ = loaded.__dict__
        else:
            logger.error(f"No previous dump found at {prev_session_dir}, cannot withdraw loop {loop_idx}")
            raise

    def dump(self, path: str | Path) -> None:
        if RD_Agent_TIMER_wrapper.timer.started:
            RD_Agent_TIMER_wrapper.timer.update_remain_time()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    def truncate_session_folder(self, li: int, si: int) -> None:
        """
        Clear the session folder by removing all session objects after the given loop index (li) and step index (si).
        """
        # clear session folders after the li
        for sf in self.session_folder.iterdir():
            if sf.is_dir() and int(sf.name) > li:
                for file in sf.iterdir():
                    file.unlink()
                sf.rmdir()

        # clear step session objects in the li
        final_loop_session_folder = self.session_folder / str(li)
        for step_session in final_loop_session_folder.glob("*_*"):
            if step_session.is_file():
                step_id = int(step_session.name.split("_", 1)[0])
                if step_id > si:
                    step_session.unlink()

    @classmethod
    def load(
        cls,
        path: str | Path,
        checkout: bool | Path | str = False,
        replace_timer: bool = True,
    ) -> "LoopBase":
        """
        Load a session from a given path.
        Parameters
        ----------
        path : str | Path
            The path to the session file.
        checkout : bool | Path | str
            If True, the new loop will use the existing folder and clear logs for sessions after the one corresponding to the given path.
            If False, the new loop will use the existing folder but keep the logs for sessions after the one corresponding to the given path.
            If a path (or a str like Path) is provided, the new loop will be saved to that path, leaving the original path unchanged.
        replace_timer : bool
            If a session is loaded, determines whether to replace the timer with session.timer.
            Default is True, which means the session timer will be replaced with the current timer.
            If False, the session timer will not be replaced.
        Returns
        -------
        LoopBase
            An instance of LoopBase with the loaded session.
        """
        path = Path(path)
        with path.open("rb") as f:
            session = cast(LoopBase, pickle.load(f))

        # set session folder
        if checkout:
            if checkout is True:
                logger.set_storages_path(session.session_folder.parent)
                max_loop = max(session.loop_trace.keys())

                # truncate log storages after the max loop
                session.truncate_session_folder(max_loop, len(session.loop_trace[max_loop]) - 1)
                logger.truncate_storages(session.loop_trace[max_loop][-1].end)
            else:
                checkout = Path(checkout)
                checkout.mkdir(parents=True, exist_ok=True)
                session.session_folder = checkout / "__session__"
                logger.set_storages_path(checkout)

        if session.timer.started:
            if replace_timer:
                RD_Agent_TIMER_wrapper.replace_timer(session.timer)
                RD_Agent_TIMER_wrapper.timer.restart_by_remain_time()
            else:
                # Use the default timer to replace the session timer
                session.timer = RD_Agent_TIMER_wrapper.timer

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
