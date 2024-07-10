import json
from copy import deepcopy
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.conf import MODEL_IMPL_SETTINGS
from rdagent.components.coder.model_coder.CoSTEER.evolvable_subjects import (
    ModelEvolvingItem,
)
from rdagent.components.coder.model_coder.CoSTEER.knowledge_management import (
    ModelQueriedKnowledge,
)
from rdagent.components.coder.model_coder.model import ModelImplementation, ModelTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evolving_framework import EvolvingStrategy
from rdagent.core.prompts import Prompts
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.oai.llm_utils import APIBackend

coder_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


class ModelCoderEvolvingStrategy(EvolvingStrategy):
    def implement_one_model(
        self,
        target_task: ModelTask,
        queried_knowledge: ModelQueriedKnowledge = None,
    ) -> ModelImplementation:
        model_information_str = target_task.get_information()

        if queried_knowledge is not None and model_information_str in queried_knowledge.success_task_to_knowledge_dict:
            return queried_knowledge.success_task_to_knowledge_dict[model_information_str].implementation
        elif queried_knowledge is not None and model_information_str in queried_knowledge.failed_task_info_set:
            return None
        else:
            queried_similar_successful_knowledge = (
                queried_knowledge.working_task_to_similar_successful_knowledge_dict[model_information_str]
                if queried_knowledge is not None
                else []
            )
            queried_former_failed_knowledge = (
                queried_knowledge.working_task_to_former_failed_knowledge_dict[model_information_str]
                if queried_knowledge is not None
                else []
            )

            queried_former_failed_knowledge_to_render = queried_former_failed_knowledge

            system_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    coder_prompts["evolving_strategy_model_coder"]["system"],
                )
                .render(
                    scenario=self.scen.get_scenario_all_desc(),
                    queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
                )
            )

            queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge
            for _ in range(10):  # max attempt to reduce the length of user_prompt
                user_prompt = (
                    Environment(undefined=StrictUndefined)
                    .from_string(
                        coder_prompts["evolving_strategy_model_coder"]["user"],
                    )
                    .render(
                        model_information_str=model_information_str,
                        queried_similar_successful_knowledge=queried_similar_successful_knowledge_to_render,
                        queried_former_failed_knowledge=queried_former_failed_knowledge_to_render,
                    )
                    .strip("\n")
                )
                if (
                    APIBackend().build_messages_and_calculate_token(
                        user_prompt=user_prompt,
                        system_prompt=system_prompt,
                    )
                    < RD_AGENT_SETTINGS.chat_token_limit
                ):
                    break
                elif len(queried_former_failed_knowledge_to_render) > 1:
                    queried_former_failed_knowledge_to_render = queried_former_failed_knowledge_to_render[1:]
                elif len(queried_similar_successful_knowledge_to_render) > 1:
                    queried_similar_successful_knowledge_to_render = queried_similar_successful_knowledge_to_render[1:]

            code = json.loads(
                APIBackend(use_chat_cache=True).build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                ),
            )["code"]
            # ast.parse(code)
            model_implementation = ModelImplementation(
                target_task,
            )
            model_implementation.prepare()
            model_implementation.inject_code(**{"model.py": code})

            return model_implementation

    def evolve(
        self,
        *,
        evo: ModelEvolvingItem,
        queried_knowledge: ModelQueriedKnowledge | None = None,
        **kwargs,
    ) -> ModelEvolvingItem:
        new_evo = deepcopy(evo)

        # 1.找出需要evolve的model
        to_be_finished_task_index = []
        for index, target_model_task in enumerate(new_evo.sub_tasks):
            target_model_task_desc = target_model_task.get_information()
            if target_model_task_desc in queried_knowledge.success_task_to_knowledge_dict:
                new_evo.sub_implementations[index] = queried_knowledge.success_task_to_knowledge_dict[
                    target_model_task_desc
                ].implementation
            elif (
                target_model_task_desc not in queried_knowledge.success_task_to_knowledge_dict
                and target_model_task_desc not in queried_knowledge.failed_task_info_set
            ):
                to_be_finished_task_index.append(index)

        result = multiprocessing_wrapper(
            [
                (self.implement_one_model, (new_evo.sub_tasks[target_index], queried_knowledge))
                for target_index in to_be_finished_task_index
            ],
            n=MODEL_IMPL_SETTINGS.evo_multi_proc_n,
        )

        for index, target_index in enumerate(to_be_finished_task_index):
            new_evo.sub_implementations[target_index] = result[index]

        # for target_index in to_be_finished_task_index:
        #     new_evo.sub_implementations[target_index] = self.implement_one_model(
        #         new_evo.sub_tasks[target_index], queried_knowledge
        #     )

        new_evo.corresponding_selection = to_be_finished_task_index

        return new_evo
