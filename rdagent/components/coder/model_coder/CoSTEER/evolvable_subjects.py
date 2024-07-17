from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.evolving_framework import EvolvableSubjects
from rdagent.log import rdagent_logger as logger


class ModelEvolvingItem(ModelExperiment, EvolvableSubjects):
    """
    Intermediate item of model implementation.
    """

    def __init__(
        self,
        sub_tasks: list[ModelTask],
        sub_gt_implementations: list[ModelFBWorkspace] = None,
    ):
        ModelExperiment.__init__(self, sub_tasks=sub_tasks)
        if sub_gt_implementations is not None and len(
            sub_gt_implementations,
        ) != len(self.sub_tasks):
            self.sub_gt_implementations = None
            logger.warning(
                "The length of sub_gt_implementations is not equal to the length of sub_tasks, set sub_gt_implementations to None",
            )
        else:
            self.sub_gt_implementations = sub_gt_implementations
