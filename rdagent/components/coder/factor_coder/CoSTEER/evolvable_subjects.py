from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
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
        sub_gt_workspace_list: list[FactorFBWorkspace] = None,
    ):
        FactorExperiment.__init__(self, sub_tasks=sub_tasks)
        self.corresponding_selection: list = None
        if sub_gt_workspace_list is not None and len(
            sub_gt_workspace_list,
        ) != len(self.sub_tasks):
            self.sub_gt_workspace_list = None
            RDAgentLog().warning(
                "The length of sub_gt_workspace_list is not equal to the length of sub_tasks, set sub_gt_workspace_list to None",
            )
        else:
            self.sub_gt_workspace_list = sub_gt_workspace_list
