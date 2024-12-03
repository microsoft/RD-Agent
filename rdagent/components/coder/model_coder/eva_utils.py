import json
from pathlib import Path
from typing import Tuple

import numpy as np
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.evaluation import Evaluator
from rdagent.core.experiment import Task, Workspace
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend

evaluate_prompts = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


def shape_evaluator(prediction: np.ndarray, target_shape: Tuple = None) -> Tuple[str, bool]:
    if target_shape is None or prediction is None:
        return (
            "No output generated from the model. No shape evaluation conducted.",
            False,
        )
    pre_shape = prediction.shape

    if pre_shape == target_shape:
        return "The shape of the output is correct.", True
    else:
        return (
            f"The shape of the output is incorrect. Expected {target_shape}, but got {pre_shape}.",
            False,
        )


def value_evaluator(
    prediction: np.ndarray,
    target: np.ndarray,
) -> Tuple[np.ndarray, bool]:
    if prediction is None:
        return "No output generated from the model. Skip value evaluation", False
    elif target is None:
        return (
            "No ground truth output provided. Value evaluation not impractical",
            False,
        )
    else:
        # Calculate the mean absolute difference
        diff = np.mean(np.abs(target - prediction))
        return (
            f"The value of the output is correct. The mean absolute difference is {diff}.",
            diff < 0.1,
        )


class ModelCodeEvaluator(Evaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        model_execution_feedback: str = "",
        model_value_feedback: str = "",
    ):
        assert isinstance(target_task, ModelTask)
        assert isinstance(implementation, ModelFBWorkspace)
        if gt_implementation is not None:
            assert isinstance(gt_implementation, ModelFBWorkspace)

        model_task_information = target_task.get_task_information()
        code = implementation.code

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(evaluate_prompts["evaluator_code_feedback"]["system"])
            .render(
                scenario=(
                    self.scen.get_scenario_all_desc(target_task, filtered_tag=target_task.model_type)
                    if self.scen is not None
                    else "No scenario description."
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
                    model_value_feedback=model_value_feedback,
                    gt_code=gt_implementation.code if gt_implementation else None,
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
        gt_implementation: Workspace,
        model_execution_feedback: str,
        model_shape_feedback: str,
        model_value_feedback: str,
        model_code_feedback: str,
    ):
        assert isinstance(target_task, ModelTask)
        assert isinstance(implementation, ModelFBWorkspace)
        if gt_implementation is not None:
            assert isinstance(gt_implementation, ModelFBWorkspace)

        system_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(evaluate_prompts["evaluator_final_feedback"]["system"])
            .render(
                scenario=(
                    self.scen.get_scenario_all_desc(target_task, filtered_tag=target_task.model_type)
                    if self.scen is not None
                    else "No scenario description."
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
                    model_value_feedback=model_value_feedback,
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
