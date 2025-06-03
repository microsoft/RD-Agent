from __future__ import annotations

from abc import abstractmethod

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evolving_framework import EvolvingStrategy, EvoStep, QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.core.utils import multiprocessing_wrapper


class MultiProcessEvolvingStrategy(EvolvingStrategy):
    def __init__(self, scen: Scenario, settings: CoSTEERSettings):
        super().__init__(scen)
        self.settings = settings

    @abstractmethod
    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: QueriedKnowledge | None = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> dict[str, str]:  # FIXME: fix interface of previous implement
        """
        This method will input the task & current workspace,
        and output the modification to applied to the workspace.
        (i.e. replace the content <filename> with <content>)

        Parameters
        ----------
        target_task : Task

        queried_knowledge : QueriedKnowledge | None

        workspace : FBWorkspace | None

        prev_task_feedback : CoSTEERSingleFeedback | None
            task feedback for previous evolving step
            None indicate it is the first loop.

        Return
        ------
        The new files {<filename>: <content>} to update the workspace.
        """
        raise NotImplementedError

    @abstractmethod
    def assign_code_list_to_evo(self, code_list: list[dict], evo: EvolvingItem) -> None:
        """
        Assign the code list to the evolving item.

        Due to the implement_one_task take `workspace` as input and output the `modification`.
        We should apply implementation to evo

        The code list is aligned with the evolving item's sub-tasks.
        If a task is not implemented, put a None in the list.
        """
        raise NotImplementedError

    def evolve(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        evolving_trace: list[EvoStep] = [],
        **kwargs,
    ) -> EvolvingItem:
        # 1.找出需要evolve的task
        to_be_finished_task_index: list[int] = []
        for index, target_task in enumerate(evo.sub_tasks):
            target_task_desc = target_task.get_task_information()
            if target_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                # NOTE: very weird logic:
                # it depends on the knowledge to set the already finished task
                evo.sub_workspace_list[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_task_desc
                ].implementation
            elif (
                target_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                and target_task_desc not in queried_knowledge.failed_task_info_set
            ):
                to_be_finished_task_index.append(index)

        last_feedback = None
        if len(evolving_trace) > 0:
            last_feedback = evolving_trace[-1].feedback
            assert isinstance(last_feedback, CoSTEERMultiFeedback)

        result = multiprocessing_wrapper(
            [
                (
                    self.implement_one_task,
                    (
                        evo.sub_tasks[target_index],
                        queried_knowledge,
                        evo.experiment_workspace,
                        None if last_feedback is None else last_feedback[target_index],
                    ),
                )
                for target_index in to_be_finished_task_index
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )
        code_list = [None for _ in range(len(evo.sub_tasks))]
        for index, target_index in enumerate(to_be_finished_task_index):
            code_list[target_index] = result[index]

        evo = self.assign_code_list_to_evo(code_list, evo)

        return evo
