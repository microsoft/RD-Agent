from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import pandas as pd

"""
This file contains the all the data class for rdagent task.
"""


class BaseTask(ABC):
    # 把name放在这里作为主键
    pass


class TaskImplementation(ABC):
    def __init__(self, target_task: BaseTask) -> None:
        self.target_task = target_task

    @abstractmethod
    def execute(self, *args: list, **kwargs: dict) -> tuple[str, pd.DataFrame]:
        error_message = "__call__ method is not implemented."
        raise NotImplementedError(error_message)


class TestCase:
    def __init__(
        self,
        target_task: BaseTask,
        ground_truth: TaskImplementation,
    ) -> None:
        self.ground_truth = ground_truth
        self.target_task = target_task


class TaskLoader:
    @abstractmethod
    def load(self) -> BaseTask | list[BaseTask]:
        error_message = "load method is not implemented."
        raise NotImplementedError(error_message)
