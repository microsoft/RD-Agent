from __future__ import annotations

from abc import abstractmethod
from pathlib import Path

from rdagent.components.coder.CoSTEER.config import CoSTEERSettings
from rdagent.components.coder.CoSTEER.evolvable_subjects import EvolvingItem
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
)
from rdagent.components.coder.CoSTEER.scheduler import random_select
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Scenario
from rdagent.core.evolving_framework import EvolvingStrategy, QueriedKnowledge
from rdagent.core.experiment import Workspace
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Task
from rdagent.core.utils import multiprocessing_wrapper

implement_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class MultiProcessEvolvingStrategy(EvolvingStrategy):
    def __init__(self, scen: Scenario, settings: CoSTEERSettings):
        super().__init__(scen)
        self.settings = settings

    @abstractmethod
    def implement_one_task(
        self,
        target_task: Task,
        queried_knowledge: QueriedKnowledge = None,
    ) -> Workspace:
        raise NotImplementedError

    def select_one_round_tasks(
        self,
        to_be_finished_task_index: list,
        evo: EvolvingItem,
        selected_num: int,
        queried_knowledge: CoSTEERQueriedKnowledge,
        scen: Scenario,
    ) -> list:
        """Since scheduler is not essential, we implement a simple random selection here."""
        return random_select(to_be_finished_task_index, evo, selected_num, queried_knowledge, scen)

    @abstractmethod
    def assign_code_list_to_evo(self, code_list: list, evo: EvolvingItem) -> None:
        """
        Assign the code list to the evolving item.

        The code list is aligned with the evolving item's sub-tasks.
        If a task is not implemented, put a None in the list.
        """
        raise NotImplementedError

    def evolve(
        self,
        *,
        evo: EvolvingItem,
        queried_knowledge: CoSTEERQueriedKnowledge | None = None,
        **kwargs,
    ) -> EvolvingItem:
        # 1.找出需要evolve的task
        to_be_finished_task_index = []
        for index, target_task in enumerate(evo.sub_tasks):
            target_task_desc = target_task.get_task_information()
            if target_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                evo.sub_workspace_list[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_task_desc
                ].implementation
            elif (
                target_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                and target_task_desc not in queried_knowledge.failed_task_info_set
            ):
                to_be_finished_task_index.append(index)

        # 2. 选择selection方法
        # if the number of factors to be implemented is larger than the limit, we need to select some of them

        if self.settings.select_threshold < len(to_be_finished_task_index):
            # Select a fixed number of factors if the total exceeds the threshold
            to_be_finished_task_index = self.select_one_round_tasks(
                to_be_finished_task_index, evo, self.settings.select_threshold, queried_knowledge, self.scen
            )

        result = multiprocessing_wrapper(
            [
                (self.implement_one_task, (evo.sub_tasks[target_index], queried_knowledge))
                for target_index in to_be_finished_task_index
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )
        code_list = [None for _ in range(len(evo.sub_tasks))]
        for index, target_index in enumerate(to_be_finished_task_index):
            code_list[target_index] = result[index]

        evo = self.assign_code_list_to_evo(code_list, evo)
        evo.corresponding_selection = to_be_finished_task_index

        return evo
