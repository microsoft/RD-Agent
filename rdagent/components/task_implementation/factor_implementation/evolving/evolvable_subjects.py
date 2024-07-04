from rdagent.components.task_implementation.factor_implementation.factor import (
    FactorExperiment,
    FactorTask,
    FileBasedFactorImplementation,
)
from rdagent.core.evolving_framework import EvolvableSubjects
from rdagent.core.log import RDAgentLog


class FactorEvolvingItem(FactorExperiment, EvolvableSubjects):
    """
    Intermediate item of factor implementation.
    """

    def __init__(
        self,
        sub_tasks: list[FactorTask],
        sub_gt_implementations: list[FileBasedFactorImplementation] = None,
    ):
        FactorExperiment.__init__(self, sub_tasks=sub_tasks)
        self.corresponding_selection: list = None
        if sub_gt_implementations is not None and len(
            sub_gt_implementations,
        ) != len(self.sub_tasks):
            self.sub_gt_implementations = None
            RDAgentLog().warning(
                "The length of sub_gt_implementations is not equal to the length of sub_tasks, set sub_gt_implementations to None",
            )
        else:
            self.sub_gt_implementations = sub_gt_implementations
