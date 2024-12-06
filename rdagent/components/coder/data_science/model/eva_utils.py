import json
from pathlib import Path
from typing import Tuple

import numpy as np
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.data_science.model.exp import ModelFBWorkspace, ModelTask
from rdagent.core.evaluation import Evaluator
from rdagent.core.experiment import Task, Workspace
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend

evaluate_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

class ModelCodeEvaluator(Evaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        model_execution_feedback: str = "",
    ):
        assert isinstance(target_task, ModelTask)
        assert isinstance(implementation, ModelFBWorkspace)

        model_task_information = target_task.get_task_information()
        code = implementation.code

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(evaluate_prompts["evaluator_code_feedback"]["system"])
            .render(
                scenario=(
                    # self.scen.get_scenario_all_desc(target_task, filtered_tag=target_task.model_type)
                    # if self.scen is not None
                    # else "No scenario description."
                    "No scenario description."
                )
            )
        )

        execution_feedback_to_render = model_execution_feedback
        for _ in range(10):  # 10 times to split the content is enough
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    evaluate_prompts["evaluator_code_feedback"]["user"],
                )
                .render(
                    model_information=model_task_information,
                    code=code,
                    model_execution_feedback=execution_feedback_to_render,
                )
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > LLM_SETTINGS.chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        critic_response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=system_prompt,
            json_mode=False,
        )

        return critic_response, None


class ModelFinalEvaluator(Evaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        model_execution_feedback: str,
        model_shape_feedback: str,
        model_code_feedback: str,
    ):
        assert isinstance(target_task, ModelTask)
        assert isinstance(implementation, ModelFBWorkspace)


        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(evaluate_prompts["evaluator_final_feedback"]["system"])
            .render(
                scenario=(
                    # self.scen.get_scenario_all_desc(target_task, filtered_tag=target_task.model_type)
                    # if self.scen is not None
                    # else "No scenario description."
                    "No scenario description."
                )
            )
        )

        execution_feedback_to_render = model_execution_feedback

        for _ in range(10):  # 10 times to split the content is enough
            user_prompt = (
                Environment(undefined=StrictUndefined)
                .from_string(
                    evaluate_prompts["evaluator_final_feedback"]["user"],
                )
                .render(
                    model_information=target_task.get_task_information(),
                    model_execution_feedback=execution_feedback_to_render,
                    model_shape_feedback=model_shape_feedback,
                    model_code_feedback=model_code_feedback,
                )
            )
            if (
                APIBackend().build_messages_and_calculate_token(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                )
                > LLM_SETTINGS.chat_token_limit
            ):
                execution_feedback_to_render = execution_feedback_to_render[len(execution_feedback_to_render) // 2 :]
            else:
                break

        final_evaluation_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
            ),
        )
        if isinstance(final_evaluation_dict["final_decision"], str) and final_evaluation_dict[
            "final_decision"
        ].lower() in ("true", "false"):
            final_evaluation_dict["final_decision"] = bool(final_evaluation_dict["final_decision"])
        return (
            final_evaluation_dict["final_feedback"],
            final_evaluation_dict["final_decision"],
        )
