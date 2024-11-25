from rdagent.core.evolving_framework import EvolvableSubjects
from rdagent.core.experiment import Experiment, FBWorkspace
from rdagent.core.scenario import Task
from rdagent.log import rdagent_logger as logger


class EvolvingItem(Experiment, EvolvableSubjects):
    """
    Intermediate item of factor implementation.
    """

    def __init__(
        self,
        sub_tasks: list[Task],
        sub_gt_implementations: list[FBWorkspace] = None,
    ):
        Experiment.__init__(self, sub_tasks=sub_tasks)
        self.corresponding_selection: list = None
        if sub_gt_implementations is not None and len(
            sub_gt_implementations,
        ) != len(self.sub_tasks):
            self.sub_gt_implementations = None
            logger.warning(
                "The length of sub_gt_implementations is not equal to the length of sub_tasks, set sub_gt_implementations to None",
            )
        else:
            self.sub_gt_implementations = sub_gt_implementations

    @classmethod
    def from_experiment(cls, exp: Experiment) -> Experiment:
        ei = cls(sub_tasks=exp.sub_tasks)
        ei.based_experiments = exp.based_experiments
        ei.experiment_workspace = exp.experiment_workspace
        return ei
