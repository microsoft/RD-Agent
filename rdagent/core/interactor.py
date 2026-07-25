from __future__ import annotations

from abc import ABC, abstractmethod
from trace import Trace
from typing import TYPE_CHECKING, Generic

from rdagent.core.experiment import ASpecificExp

if TYPE_CHECKING:
    from rdagent.core.scenario import Scenario


class Interactor(ABC, Generic[ASpecificExp]):
    def __init__(self, scen: Scenario) -> None:
        self.scen: Scenario = scen

    @abstractmethod
    def interact(self, exp: ASpecificExp, trace: Trace | None = None) -> ASpecificExp:
        """
        Interact with the experiment to get feedback or confirmation.

        Responsibilities:
        - Present the current state of the experiment.
        - Collect input to guide the next steps in the experiment.
        - Rewrite the experiment based on feedback.
        """
