from abc import ABC, abstractmethod
from typing import Generic, List, Sequence, TypeVar

from rdagent.core.experiment import Experiment

ASpecificExp = TypeVar("ASpecificExp", bound=Experiment)


class TaskGenerator(ABC, Generic[ASpecificExp]):
    @abstractmethod
    def generate(self, exp: ASpecificExp) -> ASpecificExp:
        """
        Task Generator should take in an experiment.

        Because the schedule of different tasks is crucial for the final performance
        due to it affects the learning process.

        """
        raise NotImplementedError("generate method is not implemented.")

    def collect_feedback(self, feedback_obj_l: List[object]):
        """
        When online evaluation.
        The previous feedbacks will be collected to support advanced factor generator

        Parameters
        ----------
        feedback_obj_l : List[object]

        """
