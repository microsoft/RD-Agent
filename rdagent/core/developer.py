from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Generic

from rdagent.core.experiment import ASpecificExp

if TYPE_CHECKING:
    from rdagent.core.scenario import Scenario


class Developer(ABC, Generic[ASpecificExp]):
    def __init__(self, scen: Scenario) -> None:
        self.scen: Scenario = scen

    @abstractmethod
    def develop(self, exp: ASpecificExp) -> ASpecificExp:  # TODO: remove return value
        """
        Task Generator should take in an experiment.

        Because the schedule of different tasks is crucial for the final performance
        due to it affects the learning process.

        Current constraints:
        - The developer should **inplace** edit the exp instead of returning value;
            - because we have a lot of use cases to raise errors, but we need the intermediate results in exp.
        - So we should remove the return value in the future.

        Responsibilities:
        - Generate a new experiment after developing on it.
        - If it tries to deliver message for future development, it should set a ExperimentFeedback
        """
        error_message = "generate method is not implemented."
        raise NotImplementedError(error_message)
