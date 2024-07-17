from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from rdagent.core.experiment import ASpecificExp

if TYPE_CHECKING:
    from rdagent.core.scenario import Scenario


class TaskGenerator(ABC, Generic[ASpecificExp]):
    def __init__(self, scen: Scenario) -> None:
        self.scen: Scenario = scen

    @abstractmethod
    def generate(self, exp: ASpecificExp) -> ASpecificExp:
        """
        Task Generator should take in an experiment.

        Because the schedule of different tasks is crucial for the final performance
        due to it affects the learning process.

        """
        exc = NotImplementedError("generate method is not implemented.")
        raise exc

    def collect_feedback(self, feedback_obj_l: list[object]) -> None:
        """
        When online evaluation.
        The previous feedbacks will be collected to support advanced factor generator

        Parameters
        ----------
        feedback_obj_l : List[object]

        """
