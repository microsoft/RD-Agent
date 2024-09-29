from __future__ import annotations

import json
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.factor_coder.config import FACTOR_IMPLEMENT_SETTINGS
from rdagent.components.coder.factor_coder.CoSTEER.evolvable_subjects import (
    FactorEvolvingItem,
)
from rdagent.components.coder.factor_coder.CoSTEER.scheduler import (
    LLMSelect,
    RandomSelect,
)
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evolving_framework import EvolvingStrategy, QueriedKnowledge
from rdagent.core.experiment import Workspace
from rdagent.core.prompts import Prompts
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.oai.llm_utils import APIBackend

if TYPE_CHECKING:
    from rdagent.components.coder.factor_coder.CoSTEER.knowledge_management import (
        FactorQueriedKnowledge,
        FactorQueriedKnowledgeV1,
    )

implement_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


class MultiProcessEvolvingStrategy(EvolvingStrategy):
    @abstractmethod
    def implement_one_factor(
        self,
        target_task: FactorTask,
        queried_knowledge: QueriedKnowledge = None,
    ) -> Workspace:
        raise NotImplementedError

    def evolve(
        self,
        *,
        evo: FactorEvolvingItem,
        queried_knowledge: FactorQueriedKnowledge | None = None,
        **kwargs,
    ) -> FactorEvolvingItem:
        # 1.找出需要evolve的factor
        to_be_finished_task_index = []
        for index, target_factor_task in enumerate(evo.sub_tasks):
            target_factor_task_desc = target_factor_task.get_task_information()
            if target_factor_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                evo.sub_workspace_list[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_factor_task_desc
                ].implementation
            elif (
                target_factor_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                and target_factor_task_desc not in queried_knowledge.failed_task_info_set
            ):
                to_be_finished_task_index.append(index)

        # 2. 选择selection方法
        # if the number of factors to be implemented is larger than the limit, we need to select some of them

        if FACTOR_IMPLEMENT_SETTINGS.select_threshold < len(to_be_finished_task_index):
            # Select a fixed number of factors if the total exceeds the threshold
            if FACTOR_IMPLEMENT_SETTINGS.select_method == "random":
                to_be_finished_task_index = RandomSelect(
                    to_be_finished_task_index,
                    FACTOR_IMPLEMENT_SETTINGS.select_threshold,
                )

            if FACTOR_IMPLEMENT_SETTINGS.select_method == "scheduler":
                to_be_finished_task_index = LLMSelect(
                    to_be_finished_task_index,
                    FACTOR_IMPLEMENT_SETTINGS.select_threshold,
                    evo,
                    queried_knowledge.former_traces,
                    self.scen,
                )

        result = multiprocessing_wrapper(
            [
                (self.implement_one_factor, (evo.sub_tasks[target_index], queried_knowledge))
                for target_index in to_be_finished_task_index
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        for index, target_index in enumerate(to_be_finished_task_index):
            if evo.sub_workspace_list[target_index] is None:
                evo.sub_workspace_list[target_index] = FactorFBWorkspace(target_task=evo.sub_tasks[target_index])
            evo.sub_workspace_list[target_index].inject_code(**{"factor.py": result[index]})

        evo.corresponding_selection = to_be_finished_task_index

        return evo


class FactorEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_factor(
        self,
        target_task: FactorTask,
        queried_knowledge: FactorQueriedKnowledgeV1 = None,
    ) -> str:
        factor_information_str = target_task.get_task_information()

        if queried_knowledge is not None and factor_information_str in queried_knowledge.success_task_to_knowledge_dict:
            return queried_knowledge.success_task_to_knowledge_dict[factor_information_str].implementation
        elif queried_knowledge is not None and factor_information_str in queried_knowledge.failed_task_info_set:
            return None
        else:
            queried_similar_successful_knowledge = (
                queried_knowledge.working_task_to_similar_successful_knowledge_dict[factor_information_str]
                if queried_knowledge is not None
                else []
            )
            queried_former_failed_knowledge = (
                queried_knowledge.working_task_to_former_failed_knowledge_dict[factor_information_str]
                if queried_knowledge is not None
                else []
            )

            queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

            system_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    implement_prompts["evolving_strategy_factor_implementation_v1_system"],
                )
                .render(
                    scenario=self.scen.get_scenario_all_desc(target_task),
                    queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
                )
            )
            session = APIBackend(use_chat_cache=FACTOR_IMPLEMENT_SETTINGS.coder_use_cache).build_chat_session(
                session_system_prompt=system_prompt,
            )

            queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
            for _ in range(10):  # max attempt to reduce the length of user_prompt
                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(
                        implement_prompts["evolving_strategy_factor_implementation_v1_user"],
                    )
                    .render(
                        factor_information_str=factor_information_str,
                        queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                        queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
                    )
                    .strip("\n")
                )
                if (
                    session.build_chat_completion_message_and_calculate_token(
                        user_prompt,
                    )
                    < RD_AGENT_SETTINGS.chat_token_limit
                ):
                    break
                elif len(queried_former_failed_knowledge_to_render) > 1:
                    queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
                elif len(queried_similar_successful_knowledge_to_render) > 1:
                    queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[1:]

            code = json.loads(
                session.build_chat_completion(
                    user_prompt=user_prompt,
                    json_mode=True,
                ),
            )["code"]

            return code


class FactorEvolvingStrategyWithGraph(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

    def implement_one_factor(
        self,
        target_task: FactorTask,
        queried_knowledge,
    ) -> str:
        error_summary = FACTOR_IMPLEMENT_SETTINGS.v2_error_summary
        # 1. 提取因子的背景信息
        target_factor_task_information = target_task.get_task_information()

        # 2. 检查该因子是否需要继续做（是否已经作对，是否做错太多）
        if (
            queried_knowledge is not None
            and target_factor_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_factor_task_information].implementation
        elif queried_knowledge is not None and target_factor_task_information in queried_knowledge.failed_task_info_set:
            return None
        else:
            # 3. 取出knowledge里面的经验数据（similar success、similar error、former_trace）
            queried_similar_component_knowledge = (
                queried_knowledge.component_with_success_task[target_factor_task_information]
                if queried_knowledge is not None
                else []
            )  # A list, [success task implement knowledge]

            queried_similar_error_knowledge = (
                queried_knowledge.error_with_success_task[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}

            queried_former_failed_knowledge = (
                queried_knowledge.former_traces[target_factor_task_information] if queried_knowledge is not None else []
            )

            queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

            system_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    implement_prompts["evolving_strategy_factor_implementation_v1_system"],
                )
                .render(
                    scenario=self.scen.get_scenario_all_desc(target_task),
                    queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
                )
            )

            session = APIBackend(use_chat_cache=FACTOR_IMPLEMENT_SETTINGS.coder_use_cache).build_chat_session(
                session_system_prompt=system_prompt,
            )

            queried_similar_component_knowledge_to_render = queried_similar_component_knowledge
            queried_similar_error_knowledge_to_render = queried_similar_error_knowledge
            error_summary_critics = ""
            # 动态地防止prompt超长
            for _ in range(10):  # max attempt to reduce the length of user_prompt
                # 总结error（可选）
                if (
                    error_summary
                    and len(queried_similar_error_knowledge_to_render) != 0
                    and len(queried_former_failed_knowledge_to_render) != 0
                ):
                    error_summary_system_prompt = (
                        Environment(undefined=StrictUndefined)
                        .from_string(implement_prompts["evolving_strategy_error_summary_v2_system"])
                        .render(
                            scenario=self.scen.get_scenario_all_desc(target_task),
                            factor_information_str=target_factor_task_information,
                            code_and_feedback=queried_former_failed_knowledge_to_render[
                                -1
                            ].get_implementation_and_feedback_str(),
                        )
                        .strip("\n")
                    )
                    session_summary = APIBackend(
                        use_chat_cache=FACTOR_IMPLEMENT_SETTINGS.coder_use_cache
                    ).build_chat_session(
                        session_system_prompt=error_summary_system_prompt,
                    )
                    for _ in range(10):  # max attempt to reduce the length of error_summary_user_prompt
                        error_summary_user_prompt = (
                            Environment(undefined=StrictUndefined)
                            .from_string(implement_prompts["evolving_strategy_error_summary_v2_user"])
                            .render(
                                queried_similar_component_knowledge=queried_similar_component_knowledge_to_render,
                            )
                            .strip("\n")
                        )
                        if (
                            session_summary.build_chat_completion_message_and_calculate_token(error_summary_user_prompt)
                            < RD_AGENT_SETTINGS.chat_token_limit
                        ):
                            break
                        elif len(queried_similar_error_knowledge_to_render) > 0:
                            queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
                    error_summary_critics = session_summary.build_chat_completion(
                        user_prompt=error_summary_user_prompt,
                        json_mode=False,
                    )
                # 构建user_prompt。开始写代码
                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(
                        implement_prompts["evolving_strategy_factor_implementation_v2_user"],
                    )
                    .render(
                        factor_information_str=target_factor_task_information,
                        queried_similar_component_knowledge=queried_similar_component_knowledge_to_render,
                        queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                        error_summary=error_summary,
                        error_summary_critics=error_summary_critics,
                    )
                    .strip("\n")
                )
                if (
                    session.build_chat_completion_message_and_calculate_token(
                        user_prompt,
                    )
                    < RD_AGENT_SETTINGS.chat_token_limit
                ):
                    break
                elif len(queried_former_failed_knowledge_to_render) > 1:
                    queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
                elif len(queried_similar_component_knowledge_to_render) > len(
                    queried_similar_error_knowledge_to_render,
                ):
                    queried_similar_component_knowledge_to_render = queried_similar_component_knowledge_to_render[:-1]
                elif len(queried_similar_error_knowledge_to_render) > 0:
                    queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]

            response = session.build_chat_completion(
                user_prompt=user_prompt,
                json_mode=True,
            )
            code = json.loads(response)["code"]
            return code
