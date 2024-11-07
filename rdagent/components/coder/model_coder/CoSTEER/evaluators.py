import json
import random
from pathlib import Path
from typing import List, Tuple

import numpy as np
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.conf import MODEL_IMPL_SETTINGS
from rdagent.components.coder.model_coder.CoSTEER.evolvable_subjects import (
    ModelEvolvingItem,
)
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.evaluation import Evaluator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Task, Workspace
from rdagent.core.prompts import Prompts
from rdagent.core.utils import multiprocessing_wrapper
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_conf import LLM_SETTINGS
from rdagent.oai.llm_utils import APIBackend

evaluate_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


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


class ModelCoderFeedback:
    """This feedback includes all the content to the model coder"""

    def __init__(
        self,
        execution_feedback: str,
        shape_feedback: str,
        value_feedback: str,
        code_feedback: str,
        final_feedback: str,
        final_decision: bool,
    ):
        self.execution_feedback: str = execution_feedback
        self.shape_feedback: str = shape_feedback
        self.value_feedback: str = value_feedback
        self.code_feedback: str = code_feedback
        self.final_feedback: str = final_feedback
        self.final_decision: str = final_decision

    def __str__(self) -> str:
        return f"""------------------Model Execution Feedback------------------
{self.execution_feedback}
------------------Model Shape Feedback------------------
{self.shape_feedback}
------------------Model Value Feedback------------------
{self.value_feedback}
------------------Model Code Feedback------------------
{self.code_feedback}
------------------Model Final Feedback------------------
{self.final_feedback}
------------------Model Final Decision------------------
This implementation is {'SUCCESS' if self.final_decision else 'FAIL'}.
"""


class ModelCoderEvaluator(Evaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> ModelCoderFeedback:
        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return ModelCoderFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                shape_feedback="This task has failed too many times, skip implementation.",
                value_feedback="This task has failed too many times, skip implementation.",
                code_feedback="This task has failed too many times, skip implementation.",
                final_feedback="This task has failed too many times, skip implementation.",
                final_decision=False,
            )
        assert isinstance(target_task, ModelTask)

        # NOTE: Use fixed input to test the model to avoid randomness
        batch_size = 8
        num_features = 30
        num_timesteps = 40
        input_value = 0.4
        param_init_value = 0.6

        assert isinstance(implementation, ModelFBWorkspace)
        model_execution_feedback, gen_np_array = implementation.execute(
            batch_size=batch_size,
            num_features=num_features,
            num_timesteps=num_timesteps,
            input_value=input_value,
            param_init_value=param_init_value,
        )
        if gt_implementation is not None:
            assert isinstance(gt_implementation, ModelFBWorkspace)
            _, gt_np_array = gt_implementation.execute(
                batch_size=batch_size,
                num_features=num_features,
                num_timesteps=num_timesteps,
                input_value=input_value,
                param_init_value=param_init_value,
            )
        else:
            gt_np_array = None

        shape_feedback, shape_decision = shape_evaluator(gen_np_array, (batch_size, 1))
        value_feedback, value_decision = value_evaluator(gen_np_array, gt_np_array)
        code_feedback, _ = ModelCodeEvaluator(scen=self.scen).evaluate(
            target_task=target_task,
            implementation=implementation,
            gt_implementation=gt_implementation,
            model_execution_feedback=model_execution_feedback,
            model_value_feedback="\n".join([shape_feedback, value_feedback]),
        )
        final_feedback, final_decision = ModelFinalEvaluator(scen=self.scen).evaluate(
            target_task=target_task,
            implementation=implementation,
            gt_implementation=gt_implementation,
            model_execution_feedback=model_execution_feedback,
            model_value_feedback=value_feedback,
            model_code_feedback=code_feedback,
        )

        return ModelCoderFeedback(
            execution_feedback=model_execution_feedback,
            shape_feedback=shape_feedback,
            value_feedback=value_feedback,
            code_feedback=code_feedback,
            final_feedback=final_feedback,
            final_decision=final_decision,
        )


class ModelCoderMultiEvaluator(Evaluator):
    def evaluate(
        self,
        evo: ModelEvolvingItem,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> List[ModelCoderFeedback]:
        multi_implementation_feedback = multiprocessing_wrapper(
            [
                (
                    ModelCoderEvaluator(scen=self.scen).evaluate,
                    (
                        evo.sub_tasks[index],
                        evo.sub_workspace_list[index],
                        evo.sub_gt_implementations[index] if evo.sub_gt_implementations is not None else None,
                        queried_knowledge,
                    ),
                )
                for index in range(len(evo.sub_tasks))
            ],
            n=RD_AGENT_SETTINGS.multi_proc_n,
        )

        final_decision = [
            None if single_feedback is None else single_feedback.final_decision
            for single_feedback in multi_implementation_feedback
        ]
        logger.info(f"Final decisions: {final_decision} True count: {final_decision.count(True)}")

        return multi_implementation_feedback
