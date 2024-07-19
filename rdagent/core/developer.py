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
    def develop(self, exp: ASpecificExp) -> ASpecificExp:
        """
        Task Generator should take in an experiment.

        Because the schedule of different tasks is crucial for the final performance
        due to it affects the learning process.

        """
        error_message = "generate method is not implemented."
        raise NotImplementedError(error_message)
