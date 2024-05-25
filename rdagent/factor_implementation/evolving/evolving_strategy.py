from __future__ import annotations

import json
import random
from abc import abstractmethod
from copy import deepcopy
from typing import TYPE_CHECKING

from jinja2 import Template

from core.evolving_framework import EvolvingStrategy, QueriedKnowledge
from core.utils import multiprocessing_wrapper
from factor_implementation.share_modules.conf import FactorImplementSettings
from factor_implementation.share_modules.factor import (
    FactorImplementation,
    FactorImplementationTask,
    FileBasedFactorImplementation,
)
from factor_implementation.share_modules.prompt import (
    FactorImplementationPrompts,
)
from factor_implementation.share_modules.utils import get_data_folder_intro
from oai.llm_utils import APIBackend

if TYPE_CHECKING:
    from factor_implementation.evolving.evolvable_subjects import (
        FactorImplementationList,
    )
    from factor_implementation.evolving.knowledge_management import (
        FactorImplementationQueriedKnowledge,
        FactorImplementationQueriedKnowledgeV1,
    )


class MultiProcessEvolvingStrategy(EvolvingStrategy):
    @abstractmethod
    def implement_one_factor(
        self,
        target_task: FactorImplementationTask,
        queried_knowledge: QueriedKnowledge = None,
    ) -> FactorImplementation:
        raise NotImplementedError

    def evolve(
        self,
        *,
        evo: FactorImplementationList,
        queried_knowledge: FactorImplementationQueriedKnowledge | None = None,
        **kwargs,
    ) -> FactorImplementationList:
        new_evo = deepcopy(evo)
        new_evo.corresponding_implementations = [None for _ in new_evo.target_factor_tasks]

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
        if FactorImplementSettings().implementation_factors_per_round < len(to_be_finished_task_index):
            to_be_finished_task_index = random.sample(
                to_be_finished_task_index,
                FactorImplementSettings().implementation_factors_per_round,
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

        # for target_index in to_be_finished_task_index:
        #     new_evo.corresponding_implementations[target_index] = self.implement_one_factor(
        #         new_evo.target_factor_tasks[target_index], queried_knowledge
        #     )

        return new_evo


class FactorEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_factor(
        self,
        target_task: FactorImplementationTask,
        queried_knowledge: FactorImplementationQueriedKnowledgeV1 = None,
    ) -> FactorImplementation:
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
                FactorImplementationPrompts()["evolving_strategy_factor_implementation_v1_system"],
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
                        FactorImplementationPrompts()["evolving_strategy_factor_implementation_v1_user"],
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
            # print(
            #     f"length of queried_similar_successful_knowledge_to_render: {len(queried_similar_successful_knowledge_to_render)}, length of queried_former_failed_knowledge_to_render: {len(queried_former_failed_knowledge_to_render)}"
            # )

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
    def implement_one_factor(
        self,
        target_task: FactorImplementationTask,
        queried_knowledge,
    ) -> FactorImplementation:
        error_summary = FactorImplementSettings().v2_error_summary
        target_factor_task_information = target_task.get_factor_information()

        if (
            queried_knowledge is not None
            and target_factor_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_factor_task_information].implementation
        elif queried_knowledge is not None and target_factor_task_information in queried_knowledge.failed_task_info_set:
            return None
        else:
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
                FactorImplementationPrompts()["evolving_strategy_factor_implementation_v1_system"],
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
            while True:
                if (
                    error_summary
                    and len(queried_similar_error_knowledge_to_render) != 0
                    and len(queried_former_failed_knowledge_to_render) != 0
                ):
                    error_summary_system_prompt = (
                        Template(FactorImplementationPrompts()["evolving_strategy_error_summary_v2_system"])
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
                            Template(FactorImplementationPrompts()["evolving_strategy_error_summary_v2_user"])
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

                user_prompt = (
                    Template(
                        FactorImplementationPrompts()["evolving_strategy_factor_implementation_v2_user"],
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

            # print(
            #     len(queried_similar_component_knowledge_to_render),
            #     len(queried_similar_error_knowledge_to_render),
            #     len(queried_former_failed_knowledge_to_render),
            # )

            response = session.build_chat_completion(
                user_prompt=user_prompt,
                json_mode=True,
            )
            code = json.loads(response)["code"]
            factor_implementation = FileBasedFactorImplementation(target_task, code)
            return factor_implementation
