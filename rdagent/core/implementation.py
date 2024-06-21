from abc import ABC, abstractmethod
from typing import List, Sequence

from rdagent.core.task import (
    BaseTask,
    TaskImplementation,
)

class TaskGenerator(ABC):
    @abstractmethod
    def generate(self, task_l: Sequence[BaseTask]) -> Sequence[TaskImplementation]:
        """
        Task Generator should take in a sequence of tasks.

        Because the schedule of different tasks is crucial for the final performance
        due to it affects the learning process.

        """
        raise NotImplementedError("generate method is not implemented.")

    def collect_feedback(self, feedback_obj_l: List[object]):
        """
        When online evaluation.
        The preivous feedbacks will be collected to support advanced factor generator

        Parameters
        ----------
        feedback_obj_l : List[object]

        """


