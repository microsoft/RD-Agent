from __future__ import annotations

import json
import re
from typing import Dict

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.factor_coder.config import FACTOR_COSTEER_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorFBWorkspace, FactorTask
from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class FactorMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.num_loop = 0
        self.haveSelected = False

    def error_summary(
        self,
        target_task: FactorTask,
        queried_former_failed_knowledge_to_render: list,
        queried_similar_error_knowledge_to_render: list,
    ) -> str:
        error_summary_system_prompt = T(".prompts:evolving_strategy_error_summary_v2_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task),
            factor_information_str=target_task.get_task_information(),
            code_and_feedback=queried_former_failed_knowledge_to_render[-1].get_implementation_and_feedback_str(),
        )
        for _ in range(10):  # max attempt to reduce the length of error_summary_user_prompt
            error_summary_user_prompt = T(".prompts:evolving_strategy_error_summary_v2_user").r(
                queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt
                )
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        error_summary_critics = APIBackend(
            use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
        ).build_messages_and_create_chat_completion(
            user_prompt=error_summary_user_prompt, system_prompt=error_summary_system_prompt, json_mode=False
        )
        return error_summary_critics

    def implement_one_task(
        self,
        target_task: FactorTask,
        queried_knowledge: CoSTEERQueriedKnowledge,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> str:
        target_factor_task_information = target_task.get_task_information()

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_factor_task_information]
            if queried_knowledge is not None
            else []
        )  # A list, [success task implement knowledge]

        if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2):
            queried_similar_error_knowledge = (
                queried_knowledge.task_to_similar_error_successful_knowledge[target_factor_task_information]
                if queried_knowledge is not None
                else {}
            )  # A dict, {{error_type:[[error_imp_knowledge, success_imp_knowledge],...]},...}
        else:
            queried_similar_error_knowledge = {}

        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[target_factor_task_information][0]
            if queried_knowledge is not None
            else []
        )

        queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

        latest_attempt_to_latest_successful_execution = queried_knowledge.task_to_former_failed_traces[
            target_factor_task_information
        ][1]
        system_prompt = T(".prompts:evolving_strategy_factor_implementation_v1_system").r(
            scenario=self.scen.get_scenario_all_desc(target_task, filtered_tag="feature"),
            queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
        )
        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        queried_similar_error_knowledge_to_render = queried_similar_error_knowledge
        # 动态地防止prompt超长
        for _ in range(10):  # max attempt to reduce the length of user_prompt
            # 总结error（可选）
            if (
                isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
                and FACTOR_COSTEER_SETTINGS.v2_error_summary
                and len(queried_similar_error_knowledge_to_render) != 0
                and len(queried_former_failed_knowledge_to_render) != 0
            ):
                error_summary_critics = self.error_summary(
                    target_task,
                    queried_former_failed_knowledge_to_render,
                    queried_similar_error_knowledge_to_render,
                )
            else:
                error_summary_critics = None
            # 构建user_prompt。开始写代码
            user_prompt = T(".prompts:evolving_strategy_factor_implementation_v2_user").r(
                factor_information_str=target_factor_task_information,
                queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                queried_similar_error_knowledge=queried_similar_error_knowledge_to_render,
                error_summary_critics=error_summary_critics,
                latest_attempt_to_latest_successful_execution=latest_attempt_to_latest_successful_execution,
            )
            if (
                APIBackend().build_messages_and_calculate_token(user_prompt=user_prompt, system_prompt=system_prompt)
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_successful_knowledge_to_render) > len(
                queried_similar_error_knowledge_to_render,
            ):
                queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[:-1]
            elif len(queried_similar_error_knowledge_to_render) > 0:
                queried_similar_error_knowledge_to_render = queried_similar_error_knowledge_to_render[:-1]
        for _ in range(10):
            try:
                response = APIBackend(
                    use_chat_cache=FACTOR_COSTEER_SETTINGS.coder_use_cache
                ).build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                    json_target_type=Dict[str, str],
                )

                try:
                    code = json.loads(response)["code"]
                except json.decoder.JSONDecodeError:
                    # extract python code block
                    match = re.search(r"```python(.*?)```", response, re.DOTALL)
                    if match:
                        code = match.group(1).strip()
                    else:
                        raise  # continue to retry

                return code

            except (json.decoder.JSONDecodeError, KeyError):
                pass
        else:
            return ""  # return empty code if failed to get code after 10 attempts

    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = FactorFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_files(**{"factor.py": code_list[index]})
        return evo
