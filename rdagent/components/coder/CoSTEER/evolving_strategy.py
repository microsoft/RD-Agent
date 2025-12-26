from __future__ import annotations

from abc import abstractmethod
from typing import Callable, Generator

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
    KEY_CHANGE_SUMMARY = "__change_summary__"  # Optional key for the summary of the change of evolving subjects

    def __init__(self, scen: Scenario, settings: CoSTEERSettings, improve_mode: bool = False):
        super().__init__(scen)
        self.settings = settings
        self.improve_mode = improve_mode  # improve mode means we only implement the task which has failed before. The main diff is the first loop will not implement all tasks.

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
        - Special Keys: self.KEY_CHANGE_SUMMARY;
        """
        raise NotImplementedError

    def implement_func_list(self) -> list[Callable]:
        """
        One evolve solution will be divided into multiple implement functions.
        The functions will be called sequentially.

        `implement_one_task` is the default implementation.  Please refer to its signature for more details.
        """
        return [self.implement_one_task]

    @abstractmethod
    def assign_code_list_to_evo(self, code_list: list[dict], evo: EvolvingItem) -> None:
        """
        Assign the code list to the evolving item.

        Due to the implement_one_task take `workspace` as input and output the `modification`.
        We should apply implementation to evo

        Assumptions:
        - The modidication on evo should happen in-place!!

        The code list is aligned with the evolving item's sub-tasks.
        If a task is not implemented, put a None in the list.
        """
        raise NotImplementedError

    def assign_code_list_to_evo(self, code_list: list[dict | None], evo) -> None:
        """Assign code modifications to evolving item.

        For runner, coder already generated full training config, so typically no modifications.
        But this method is required by the abstract base class.
        """
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = evo.experiment_workspace

            # If there are any modifications (usually empty for runner)
            if code_list[index]:
                # Handle change summary if present
                if self.KEY_CHANGE_SUMMARY in code_list[index]:
                    evo.sub_workspace_list[index].change_summary = code_list[index].pop(self.KEY_CHANGE_SUMMARY)
                # Inject any modified files
                evo.sub_workspace_list[index].inject_files(**code_list[index])

        return evo


    def evolve_iter(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        evolving_trace: list[EvoStep] = [],
        **kwargs,
    ) -> Generator[EvolvingItem, EvolvingItem, None]:
        if queried_knowledge is None:
            raise ValueError(
                "MultiProcessEvolvingStrategy requires queried_knowledge for efficient implementation. Please set with_knowledge=True in CoSTEER constructor."
            )
        code_list = [None for _ in range(len(evo.sub_tasks))]

        last_feedback = None
        if len(evolving_trace) > 0:
            last_feedback = evolving_trace[-1].feedback
            assert isinstance(last_feedback, CoSTEERMultiFeedback)

        # 1.找出需要evolve的task
        to_be_finished_task_index: list[int] = []
        for index, target_task in enumerate(evo.sub_tasks):
            target_task_desc = target_task.get_task_information()
            if target_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                # NOTE: very weird logic:
                # it depends on the knowledge to set the already finished task
                code_list[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_task_desc
                ].implementation.file_dict
            else:
                # Schedule the task only if:
                # - it is not marked failed
                # - and (in improve mode) we actually have prior failure feedback to act on
                skip_for_improve_mode = self.improve_mode and (
                    last_feedback is None
                    or (isinstance(last_feedback, CoSTEERMultiFeedback) and last_feedback[index] is None)
                )
                if target_task_desc not in queried_knowledge.failed_task_info_set and not skip_for_improve_mode:
                    to_be_finished_task_index.append(index)
                if skip_for_improve_mode:
                    code_list[index] = (
                        {}
                    )  # empty implementation for skipped task, but assign_code_list_to_evo will still assign it

        for implement_func in self.implement_func_list():
            result = multiprocessing_wrapper(
                [
                    (
                        implement_func,
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
            for index, target_index in enumerate(to_be_finished_task_index):
                code_list[target_index] = result[index]

            self.assign_code_list_to_evo(code_list, evo)
            yield evo
