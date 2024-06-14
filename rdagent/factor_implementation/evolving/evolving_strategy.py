from __future__ import annotations

import json
from abc import abstractmethod
from copy import deepcopy
from pathlib import Path
from typing import TYPE_CHECKING

from jinja2 import Template

from rdagent.core.evolving_framework import EvolvingStrategy, QueriedKnowledge
from rdagent.oai.llm_utils import APIBackend
from rdagent.factor_implementation.share_modules.factor_implementation_config import (
    FactorImplementSettings,
)

from rdagent.core.task import (
    TaskImplementation,
)
from rdagent.core.prompts import Prompts

from pathlib import Path

from rdagent.factor_implementation.evolving.scheduler import (
    RandomSelect,
    LLMSelect,
)

from rdagent.factor_implementation.share_modules.factor_implementation_utils import get_data_folder_intro
from rdagent.oai.llm_utils import APIBackend

from rdagent.core.utils import multiprocessing_wrapper

from rdagent.factor_implementation.evolving.factor import (
    FactorImplementTask,
    FactorEvovlingItem,
    FileBasedFactorImplementation,
)

if TYPE_CHECKING:
    from rdagent.factor_implementation.evolving.knowledge_management import (
        FactorImplementationQueriedKnowledge,
        FactorImplementationQueriedKnowledgeV1,
    )

implement_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


class MultiProcessEvolvingStrategy(EvolvingStrategy):
    @abstractmethod
    def implement_one_factor(
        self,
        target_task: FactorImplementTask,
        queried_knowledge: QueriedKnowledge = None,
    ) -> TaskImplementation:
        raise NotImplementedError

    def evolve(
        self,
        *,
        evo: FactorEvovlingItem,
        queried_knowledge: FactorImplementationQueriedKnowledge | None = None,
        **kwargs,
    ) -> FactorEvovlingItem:
        self.num_loop += 1
        new_evo = deepcopy(evo)

        # 1.找出需要evolve的factor
        to_be_finished_task_index = []
        for index, target_factor_task in enumerate(new_evo.target_factor_tasks):
            target_factor_task_desc = target_factor_task.get_factor_information()
            if target_factor_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                new_evo.corresponding_implementations[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_factor_task_desc
                ].implementation
            elif (
                target_factor_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                and target_factor_task_desc not in queried_knowledge.failed_task_info_set
            ):
                to_be_finished_task_index.append(index)

        # 2. 选择selection方法
        # if the number of factors to be implemented is larger than the limit, we need to select some of them
        if FactorImplementSettings().select_ratio < 1:
            # if the number of loops is equal to the select_loop, we need to select some of them
            implementation_factors_per_round = int(
                FactorImplementSettings().select_ratio * len(to_be_finished_task_index)
            )
            if FactorImplementSettings().select_method == "random":
                to_be_finished_task_index = RandomSelect(
                    to_be_finished_task_index,
                    implementation_factors_per_round,
                )

            if FactorImplementSettings().select_method == "scheduler":
                to_be_finished_task_index = LLMSelect(
                    to_be_finished_task_index,
                    implementation_factors_per_round,
                    new_evo,
                    queried_knowledge.former_traces,
                )

        result = multiprocessing_wrapper(
            [
                (self.implement_one_factor, (new_evo.target_factor_tasks[target_index], queried_knowledge))
                for target_index in to_be_finished_task_index
            ],
            n=FactorImplementSettings().evo_multi_proc_n,
        )

        for index, target_index in enumerate(to_be_finished_task_index):
            new_evo.corresponding_implementations[target_index] = result[index]
            if result[index].target_task.factor_name in new_evo.evolve_trace:
                new_evo.evolve_trace[result[index].target_task.factor_name].append(result[index])
            else:
                new_evo.evolve_trace[result[index].target_task.factor_name] = [result[index]]

        new_evo.corresponding_selection.append(to_be_finished_task_index)

        return new_evo


class FactorEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_factor(
        self,
        target_task: FactorImplementTask,
        queried_knowledge: FactorImplementationQueriedKnowledgeV1 = None,
    ) -> TaskImplementation:
        factor_information_str = target_task.get_factor_information()

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

            system_prompt = Template(
                implement_prompts["evolving_strategy_factor_implementation_v1_system"],
            ).render(
                data_info=get_data_folder_intro(),
                queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            )
            session = APIBackend(use_chat_cache=False).build_chat_session(
                session_system_prompt=system_prompt,
            )

            queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
            while True:
                user_prompt = (
                    Template(
                        implement_prompts["evolving_strategy_factor_implementation_v1_user"],
                    )
                    .render(
                        factor_information_str=factor_information_str,
                        queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                    )
                    .strip("\n")
                )
                if (
                    session.build_chat_completion_message_and_calculate_token(
                        user_prompt,
                    )
                    < FactorImplementSettings().chat_token_limit
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
            # ast.parse(code)
            factor_implementation = FileBasedFactorImplementation(
                target_task,
                code,
            )

            return factor_implementation


class FactorEvolvingStrategyWithGraph(MultiProcessEvolvingStrategy):
    def __init__(self) -> None:
        self.num_loop = 0
        self.haveSelected = False

    def implement_one_factor(
        self,
        target_task: FactorImplementTask,
        queried_knowledge,
    ) -> TaskImplementation:
        error_summary = FactorImplementSettings().v2_error_summary
        # 1. 提取因子的背景信息
        target_factor_task_information = target_task.get_factor_information()

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

            system_prompt = Template(
                implement_prompts["evolving_strategy_factor_implementation_v1_system"],
            ).render(
                data_info=get_data_folder_intro(),
                queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            )

            session = APIBackend(use_chat_cache=False).build_chat_session(
                session_system_prompt=system_prompt,
            )

            queried_similar_component_knowledge_to_render = queried_similar_component_knowledge
            queried_similar_error_knowledge_to_render = queried_similar_error_knowledge
            error_summary_critics = ""
            # 动态地防止prompt超长
            while True:
                # 总结error（可选）
                if (
                    error_summary
                    and len(queried_similar_error_knowledge_to_render) != 0
                    and len(queried_former_failed_knowledge_to_render) != 0
                ):

                    error_summary_system_prompt = (
                        Template(implement_prompts["evolving_strategy_error_summary_v2_system"])
                        .render(
                            factor_information_str=target_factor_task_information,
                            code_and_feedback=queried_former_failed_knowledge_to_render[
                                -1
                            ].get_implementation_and_feedback_str(),
                        )
                        .strip("\n")
                    )
                    session_summary = APIBackend(use_chat_cache=False).build_chat_session(
                        session_system_prompt=error_summary_system_prompt,
                    )
                    while True:
                        error_summary_user_prompt = (
                            Template(implement_prompts["evolving_strategy_error_summary_v2_user"])
                            .render(
                                queried_similar_component_knowledge=queried_similar_component_knowledge_to_render,
                            )
                            .strip("\n")
                        )
                        if (
                            session_summary.build_chat_completion_message_and_calculate_token(error_summary_user_prompt)
                            < FactorImplementSettings().chat_token_limit
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
                    Template(
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
                    < FactorImplementSettings().chat_token_limit
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
            factor_implementation = FileBasedFactorImplementation(target_task, code)
            return factor_implementation
