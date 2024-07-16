from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.evolving_framework import EvolvableSubjects
from rdagent.core.log import RDAgentLog


class ModelEvolvingItem(ModelExperiment, EvolvableSubjects):
    """
    Intermediate item of model implementation.
    """

    def __init__(
        self,
        sub_tasks: list[ModelTask],
        sub_gt_workspace_list: list[ModelFBWorkspace] = None,
    ):
        ModelExperiment.__init__(self, sub_tasks=sub_tasks)
        if sub_gt_workspace_list is not None and len(
            sub_gt_workspace_list,
        ) != len(self.sub_tasks):
            self.sub_gt_workspace_list = None
            RDAgentLog().warning(
                "The length of sub_gt_workspace_list is not equal to the length of sub_tasks, set sub_gt_workspace_list to None",
            )
        else:
            self.sub_gt_workspace_list = sub_gt_workspace_list
