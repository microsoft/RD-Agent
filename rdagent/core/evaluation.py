from abc import ABC, abstractmethod

from rdagent.core.experiment import Implementation, Task
from rdagent.core.scenario import Scenario


class Feedback:
    pass


class Evaluator(ABC):
    def __init__(
        self,
        scen: Scenario,
    ) -> None:
        self.scen = scen

    @abstractmethod
    def evaluate(
        self,
        target_task: Task,
        implementation: Implementation,
        gt_implementation: Implementation,
        **kwargs,
    ):
        raise NotImplementedError
