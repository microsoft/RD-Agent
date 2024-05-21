from abc import ABC, abstractmethod
from typing import List

from factor_implementation.share_modules.factor import (
    FactorImplementation,
    FactorImplementationTask,
)


class FactorGenerator(ABC):
    """
    Because implementing factors will help each other in the process of implementation, we use the interface `List[FactorImplementationTask] -> List[FactorImplementation]` instead of single factor .
    """

    def __init__(self, target_task_l: List[FactorImplementationTask]) -> None:
        self.target_task_l = target_task_l

    @abstractmethod
    def generate(self, *args, **kwargs) -> List[FactorImplementation]:
        raise NotImplementedError("generate method is not implemented.")

    def collect_feedback(self, feedback_obj_l: List[object]):
        """
        When online evaluation.
        The preivous feedbacks will be collected to support advanced factor generator

        Parameters
        ----------
        feedback_obj_l : List[object]

        """
