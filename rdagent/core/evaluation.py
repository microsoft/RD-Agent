import typing
from abc import ABC, abstractmethod

if typing.TYPE_CHECKING:
    from rdagent.core.experiment import Task, Workspace
    from rdagent.core.scenario import Scenario


class Feedback:
    """
    Design Principle:
        It will be more like a **dataclass**.
        The building process of feedback will should be in evaluator
    """

    def __bool__(self) -> bool:
        return True


class Evaluator(ABC):
    """
    Design Principle:

        It should cover the building process of feedback from raw information.
            Typically the building of feedback will be two phases.
            1. raw information including stdout & workspace  (feedback itself will handle this)
            2. advanced/summarized feedback information. (evaluate will handle this)
    """

    def __init__(
        self,
        scen: "Scenario",
    ) -> None:
        self.scen = scen

    @abstractmethod
    def evaluate(
        self,
        target_task: "Task",
        implementation: "Workspace",
        gt_implementation: "Workspace",
        **kwargs: object,
    ) -> Feedback:
        raise NotImplementedError
