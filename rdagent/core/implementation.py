from abc import ABC, abstractmethod
from typing import List

from rdagent.core.task import (
    TaskImplementation,
)

class TaskGenerator(ABC):
    @abstractmethod
    def generate(self, *args, **kwargs) -> List[TaskImplementation]:
        raise NotImplementedError("generate method is not implemented.")

    def collect_feedback(self, feedback_obj_l: List[object]):
        """
        When online evaluation.
        The preivous feedbacks will be collected to support advanced factor generator

        Parameters
        ----------
        feedback_obj_l : List[object]

        """


