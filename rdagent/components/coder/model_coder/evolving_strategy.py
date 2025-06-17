import json
from typing import Dict

from rdagent.components.coder.CoSTEER.config import CoSTEER_SETTINGS
from rdagent.components.coder.CoSTEER.evaluators import CoSTEERSingleFeedback
from rdagent.components.coder.CoSTEER.evolving_strategy import (
    MultiProcessEvolvingStrategy,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledge,
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.experiment import FBWorkspace
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class ModelMultiProcessEvolvingStrategy(MultiProcessEvolvingStrategy):
    def implement_one_task(
        self,
        target_task: ModelTask,
        queried_knowledge: CoSTEERQueriedKnowledge = None,
        workspace: FBWorkspace | None = None,
        prev_task_feedback: CoSTEERSingleFeedback | None = None,
    ) -> str:
        model_information_str = target_task.get_task_information()

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[model_information_str]
            if queried_knowledge is not None
            else []
        )
        queried_former_failed_knowledge = (
            queried_knowledge.task_to_former_failed_traces[model_information_str]
            if queried_knowledge is not None
            else []
        )

        queried_former_failed_knowledge_to_render = (
            queried_former_failed_knowledge[0]
            if isinstance(queried_knowledge, CoSTEERQueriedKnowledgeV2)
            else queried_former_failed_knowledge
        )
        system_prompt = T(".prompts:evolving_strategy_model_coder.system").r(
            scenario=self.scen.get_scenario_all_desc(filtered_tag="model"),
            queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            current_code=workspace.file_dict.get("model.py"),
        )

        queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
        for _ in range(10):  # max attempt to reduce the length of user_prompt
            user_prompt = T(".prompts:evolving_strategy_model_coder.user").r(
                model_information_str=model_information_str,
                queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                < LLM_SETTINGS.chat_token_limit
            ):
                break
            elif len(queried_former_failed_knowledge_to_render) > 1:
                queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
            elif len(queried_similar_successful_knowledge_to_render) > 1:
                queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[1:]

        code = json.loads(
            APIBackend(use_chat_cache=CoSTEER_SETTINGS.coder_use_cache).build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_target_type=Dict[str, str],
            ),
        )["code"]
        return code

    def assign_code_list_to_evo(self, code_list, evo):
        for index in range(len(evo.sub_tasks)):
            if code_list[index] is None:
                continue
            if evo.sub_workspace_list[index] is None:
                evo.sub_workspace_list[index] = ModelFBWorkspace(target_task=evo.sub_tasks[index])
            evo.sub_workspace_list[index].inject_files(**{"model.py": code_list[index]})
        return evo
