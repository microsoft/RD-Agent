from abc import ABC, abstractmethod
from typing import Tuple
import pandas as pd
from rdagent.core.evolving_framework import EvolvableSubjects

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

class FactorImplementTask(BaseTask):
    def __init__(
        self,
        factor_name,
        factor_description,
        factor_formulation,
        factor_formulation_description: str = '',
        variables: dict = {},
        resource: str = None,
    ) -> None:
        self.factor_name = factor_name
        self.factor_description = factor_description
        self.factor_formulation = factor_formulation
        self.factor_formulation_description = factor_formulation_description
        self.variables = variables
        self.factor_resources = resource

    def get_factor_information(self):
        return f"""factor_name: {self.factor_name}
factor_description: {self.factor_description}
factor_formulation: {self.factor_formulation}
factor_formulation_description: {self.factor_formulation_description}"""

    @staticmethod
    def from_dict(dict):
        return FactorImplementTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"

class ModelTask(BaseTask):
    # TODO: What is the basic info of a Model task?
    @staticmethod
    def from_dict(dict):
        return ModelTask(**dict)

    def __repr__(self) -> str:
        return f"<{self.__class__.__name__}[{self.factor_name}]>"

class FactorEvovlingItem(EvolvableSubjects):
    """
    Intermediate item of factor implementation.
    """

    def __init__(
        self,
        target_factor_tasks: list[FactorImplementTask],
    ):
        super().__init__()
        self.target_factor_tasks = target_factor_tasks
        self.corresponding_implementations: list[TaskImplementation] = []
        self.corresponding_selection: list[list] = []
        self.evolve_trace = {}