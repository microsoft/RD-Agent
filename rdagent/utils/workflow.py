"""
This is a class that try to store/resume/traceback the workflow session


Postscripts:
- Originally, I want to implement it in a more general way with python generator.
  However, Python generator is not picklable (dill does not support pickle as well)

"""

import datetime
import pickle
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

from tqdm.auto import tqdm

from rdagent.log import rdagent_logger as logger


class LoopMeta(type):
    @staticmethod
    def _get_steps(bases):
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

    def __new__(cls, clsname, bases, attrs):
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
            if not name.startswith("_") and isinstance(attr, Callable):
                if name not in steps:
                    # NOTE: if we override the step in the subclass
                    # Then it is not the new step. So we skip it.
                    steps.append(name)
        attrs["steps"] = steps
        return super().__new__(cls, clsname, bases, attrs)


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

    steps: list[Callable]  # a list of steps to work on
    loop_trace: dict[int, list[LoopTrace]]

    skip_loop_error: tuple[Exception] = field(
        default_factory=tuple
    )  # you can define a list of error that will skip current loop

    EXCEPTION_KEY = "_EXCEPTION"

    def __init__(self):
        self.loop_idx = 0  # current loop index
        self.step_idx = 0  # the index of next step to be run
        self.loop_prev_out = {}  # the step results of current loop
        self.loop_trace = defaultdict(list[LoopTrace])  # the key is the number of loop
        self.session_folder = logger.log_trace_path / "__session__"

    def run(self, step_n: int | None = None):
        """

        Parameters
        ----------
        step_n : int | None
            How many steps to run;
            `None` indicates to run forever until error or KeyboardInterrupt
        """
        with tqdm(total=len(self.steps), desc="Workflow Progress", unit="step") as pbar:
            while True:
                if step_n is not None:
                    if step_n <= 0:
                        break
                    step_n -= 1

                li, si = self.loop_idx, self.step_idx

                name = self.steps[si]
                with logger.tag(f"Loop_{li}.{name}"):
                    start = datetime.datetime.now(datetime.timezone.utc)
                    func = getattr(self, name)
                    try:
                        self.loop_prev_out[name] = func(self.loop_prev_out)
                        # TODO: Fix the error logger.exception(f"Skip loop {li} due to {e}")
                    except self.skip_loop_error as e:
                        # FIXME: This does not support previous demo (due to their last step is not for recording)
                        logger.warning(f"Skip loop {li} due to {e}")
                        # NOTE: strong assumption!  The last step is responsible for recording information
                        self.step_idx = len(self.steps) - 1  # directly jump to the last step.
                        self.loop_prev_out[self.EXCEPTION_KEY] = e
                        continue
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
                    self.loop_prev_out = {}
                    pbar.reset()  # reset the progress bar for the next loop

                self.dump(self.session_folder / f"{li}" / f"{si}_{name}")  # save a snapshot after the session

    def dump(self, path: str | Path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str | Path):
        path = Path(path)
        with path.open("rb") as f:
            session = pickle.load(f)
        logger.set_trace_path(session.session_folder.parent)

        max_loop = max(session.loop_trace.keys())
        logger.storage.truncate(time=session.loop_trace[max_loop][-1].end)
        return session
