"""
This is a class that try to store/resume/traceback the workflow session


Postscripts:
- Originally, I want to implement it in a more general way with python generator.
  However, Python generator is not picklable (dill does not support pickle as well)

"""

import asyncio
import concurrent.futures
import datetime
import pickle
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Union, cast

from tqdm.auto import tqdm

from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.log import rdagent_logger as logger
from rdagent.log.conf import LOG_SETTINGS
from rdagent.log.timer import RD_Agent_TIMER_wrapper, RDAgentTimer
from rdagent.utils.workflow.tracking import WorkflowTracker


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

    Unsolved problem:
    - Global variable synchronization when `force_subproc` is True
        - Timer
    """

    steps: list[str]  # a list of steps to work on
    loop_trace: dict[int, list[LoopTrace]]

    skip_loop_error: tuple[type[BaseException], ...] = ()  # you can define a list of error that will skip current loop
    withdraw_loop_error: tuple[
        type[BaseException], ...
    ] = ()  # you can define a list of error that will withdraw current loop

    EXCEPTION_KEY = "_EXCEPTION"
    SENTINEL = -1

    _pbar: tqdm  # progress bar instance

    class LoopTerminationError(Exception):
        """Exception raised when loop conditions indicate the loop should terminate"""

    class LoopResumeError(Exception):
        """Exception raised when loop conditions indicate the loop should stop all coroutines and resume"""

    def __init__(self) -> None:
        # progress control
        self.loop_idx: int = 0  # current loop index / next loop index to kickoff
        self.step_idx: defaultdict[int, int] = defaultdict(int)  # dict from loop index to next step index
        self.queue: asyncio.Queue[Any] = asyncio.Queue()

        # Store step results for all loops in a nested dictionary: loop_prev_out[loop_index][step_name]
        self.loop_prev_out: dict[int, dict[str, Any]] = defaultdict(dict)
        self.loop_trace = defaultdict(list[LoopTrace])  # the key is the number of loop
        self.session_folder = Path(LOG_SETTINGS.trace_path) / "__session__"
        self.timer: RDAgentTimer = RD_Agent_TIMER_wrapper.timer
        self.tracker = WorkflowTracker(self)  # Initialize tracker with this LoopBase instance

        # progress control
        self.loop_n: Optional[int] = None  # remain loop count
        self.step_n: Optional[int] = None  # remain step count

        self.semaphores: dict[str, asyncio.Semaphore] = {}

    def get_unfinished_loop_cnt(self, next_loop: int) -> int:
        n = 0
        for li in range(next_loop):
            if self.step_idx[li] < len(self.steps):  # unfinished loop
                n += 1
        return n

    def get_semaphore(self, step_name: str) -> asyncio.Semaphore:
        if isinstance(limit := RD_AGENT_SETTINGS.step_semaphore, dict):
            limit = limit.get(step_name, 1)  # default to 1 if not specified

        # NOTE:
        # (1) we assume the record step is always the last step to modify the global environment,
        #     so we set the limit to 1 to avoid race condition
        # (2) Because we support (-1,) as local selection; So it is hard to align a) the comparision target in `feedbck`
        #     and b) parent node in `record`; So we prevent parallelism in `feedback` and `record` to avoid inconsistency
        if step_name in ("record", "feedback"):
            limit = 1

        if step_name not in self.semaphores:
            self.semaphores[step_name] = asyncio.Semaphore(limit)
        return self.semaphores[step_name]

    @property
    def pbar(self) -> tqdm:
        """Progress bar property that initializes itself if it doesn't exist."""
        if getattr(self, "_pbar", None) is None:
            self._pbar = tqdm(total=len(self.steps), desc="Workflow Progress", unit="step")
        return self._pbar

    def close_pbar(self) -> None:
        if getattr(self, "_pbar", None) is not None:
            self._pbar.close()
            del self._pbar

    def _check_exit_conditions_on_step(self) -> None:
        """Check if the loop should continue or terminate.

        Raises
        ------
        LoopTerminationException
            When conditions indicate that the loop should terminate
        """
        # Check step count limitation
        if self.step_n is not None:
            if self.step_n <= 0:
                raise self.LoopTerminationError("Step count reached")
            self.step_n -= 1

        # Check timer timeout
        if self.timer.started:
            if self.timer.is_timeout():
                logger.warning("Timeout, exiting the loop.")
                raise self.LoopTerminationError("Timer timeout")
            else:
                logger.info(f"Timer remaining time: {self.timer.remain_time()}")

    async def _run_step(self, li: int, force_subproc: bool = False) -> None:
        """Execute a single step (next unrun step) in the workflow (async version with force_subproc option).

        Parameters
        ----------
        li : int
            Loop index

        force_subproc : bool
            Whether to force the step to run in a subprocess in asyncio

        Returns
        -------
        Any
            The result of the step function
        """
        si = self.step_idx[li]
        name = self.steps[si]

        async with self.get_semaphore(name):

            logger.info(f"Start Loop {li}, Step {si}: {name}")
            self.tracker.log_workflow_state()

            with logger.tag(f"Loop_{li}.{name}"):
                start = datetime.datetime.now(datetime.timezone.utc)
                func: Callable[..., Any] = cast(Callable[..., Any], getattr(self, name))

                next_step_idx = si + 1
                step_forward = True
                try:
                    # Call function with current loop's output, await if coroutine or use ProcessPoolExecutor for sync if required
                    if force_subproc:
                        curr_loop = asyncio.get_running_loop()
                        with concurrent.futures.ProcessPoolExecutor() as pool:
                            result = await curr_loop.run_in_executor(pool, func, self.loop_prev_out[li])
                    else:
                        # auto determine whether to run async or sync
                        if asyncio.iscoroutinefunction(func):
                            result = await func(self.loop_prev_out[li])
                        else:
                            # Default: run sync function directly
                            result = func(self.loop_prev_out[li])
                    # Store result in the nested dictionary
                    self.loop_prev_out[li][name] = result

                    # Record the trace
                    end = datetime.datetime.now(datetime.timezone.utc)
                    self.loop_trace[li].append(LoopTrace(start, end, step_idx=si))
                    # Save snapshot after completing the step
                    self.dump(self.session_folder / f"{li}" / f"{si}_{name}")
                except Exception as e:
                    if isinstance(e, self.skip_loop_error):
                        logger.warning(f"Skip loop {li} due to {e}")
                        # Jump to the last step (assuming last step is for recording)
                        next_step_idx = len(self.steps) - 1
                        self.loop_prev_out[li][self.EXCEPTION_KEY] = e
                    elif isinstance(e, self.withdraw_loop_error):
                        logger.warning(f"Withdraw loop {li} due to {e}")
                        # Back to previous loop
                        self.withdraw_loop(li)
                        step_forward = False

                        msg = "We have reset the loop instance, stop all the routines and resume."
                        raise self.LoopResumeError(msg) from e
                    else:
                        raise  # re-raise unhandled exceptions
                finally:
                    if step_forward:
                        # Increment step index
                        self.step_idx[li] = next_step_idx

                        # Update progress bar
                        current_step = self.step_idx[li]
                        self.pbar.n = current_step
                        next_step = self.step_idx[li] % len(self.steps)
                        self.pbar.set_postfix(
                            loop_index=li + next_step_idx // len(self.steps),
                            step_index=next_step,
                            step_name=self.steps[next_step],
                        )
                        self._check_exit_conditions_on_step()
                    else:
                        logger.warning(f"Step forward {si} of loop {li} is skipped.")

    async def kickoff_loop(self) -> None:
        while True:
            li = self.loop_idx

            # exit on loop limitation
            if self.loop_n is not None:
                if self.loop_n <= 0:
                    for _ in range(RD_AGENT_SETTINGS.get_max_parallel()):
                        self.queue.put_nowait(self.SENTINEL)
                    break
                self.loop_n -= 1

            # NOTE:
            # Try best to kick off the first step; the first step is always the ExpGen;
            # it have the right to decide when to stop yield new Experiment
            if self.step_idx[li] == 0:
                # Assume the first step is ExpGen
                # Only kick off ExpGen when it is never kicked off before
                await self._run_step(li)
            self.queue.put_nowait(li)  # the loop `li` has been kicked off, waiting for workers to pick it up
            self.loop_idx += 1

    async def execute_loop(self) -> None:
        while True:
            # 1) get the tasks to goon loop `li`
            li = await self.queue.get()
            if li == self.SENTINEL:
                break
            # 2) run the unfinished steps
            while self.step_idx[li] < len(self.steps):
                if self.step_idx[li] == len(self.steps) - 1:
                    # NOTE: assume the last step is record, it will be fast and affect the global environment
                    # if it is the last step, run it directly ()
                    await self._run_step(li)
                else:
                    # await the step; parallel running happens here!
                    # Only trigger subprocess if we have more than one process.
                    await self._run_step(li, force_subproc=RD_AGENT_SETTINGS.is_force_subproc())

    async def run(self, step_n: int | None = None, loop_n: int | None = None, all_duration: str | None = None) -> None:
        """Run the workflow loop.

        Parameters
        ----------
        loop_n: int | None
            How many loops to run; if current loop is incomplete, it will be counted as the first loop for completion
            `None` indicates to run forever until error or KeyboardInterrupt
        all_duration : str | None
            Maximum duration to run, in format accepted by the timer
        """
        # Initialize timer if duration is provided
        if all_duration is not None and not self.timer.started:
            self.timer.reset(all_duration=all_duration)

        if step_n is not None:
            self.step_n = step_n
        if loop_n is not None:
            self.loop_n = loop_n

        # empty the queue when restarting
        while not self.queue.empty():
            self.queue.get_nowait()
        self.loop_idx = (
            0  # if we rerun the loop, we should revert the loop index to 0 to make sure every loop is correctly kicked
        )

        while True:
            try:
                # run one kickoff_loop and execute_loop
                await asyncio.gather(
                    self.kickoff_loop(), *[self.execute_loop() for _ in range(RD_AGENT_SETTINGS.get_max_parallel())]
                )
                break
            except self.LoopResumeError as e:
                logger.warning(f"Stop all the routines and resume loop: {e}")
                self.loop_idx = 0
            except self.LoopTerminationError as e:
                logger.warning(f"Reach stop criterion and stop loop: {e}")
                break
            finally:
                self.close_pbar()

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
        # if the path is a directory, load the latest session
        if path.is_dir():
            if path.name != "__session__":
                path = path / "__session__"

            if not path.exists():
                raise FileNotFoundError(f"No session file found in {path}")

            # iterate the dump steps in increasing order
            files = sorted(path.glob("*/*_*"), key=lambda f: (int(f.parent.name), int(f.name.split("_")[0])))
            path = files[-1]
            logger.info(f"Loading latest session from {path}")
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

    def __getstate__(self) -> dict[str, Any]:
        res = {}
        for k, v in self.__dict__.items():
            if k not in ["queue", "semaphores", "_pbar"]:
                res[k] = v
        return res

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__.update(state)
        self.queue = asyncio.Queue()
        self.semaphores = {}
