from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd

'''
This file contains the all the data class for rdagent task.
'''
class BaseTask(ABC):
    # 把name放在这里作为主键
    pass

class TaskImplementation(ABC):
    def __init__(self, target_task: BaseTask) -> None:
        self.target_task = target_task

    @abstractmethod
    def execute(self, *args, **kwargs) -> Tuple[str, pd.DataFrame]:
        raise NotImplementedError("__call__ method is not implemented.")

class TestCase:
    def __init__(
        self,
        target_task: BaseTask,
        ground_truth: TaskImplementation,
    ):
        self.ground_truth = ground_truth
        self.target_task = target_task

