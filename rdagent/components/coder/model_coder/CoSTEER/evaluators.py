from typing import List

import numpy as np
import torch

from rdagent.components.coder.model_coder.conf import MODEL_IMPL_SETTINGS
from rdagent.components.coder.model_coder.CoSTEER.evolvable_subjects import (
    ModelEvolvingItem,
)
from rdagent.components.coder.model_coder.model import ModelImplementation, ModelTask
from rdagent.core.evaluation import Evaluator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Implementation, Task
from rdagent.core.log import RDAgentLog
from rdagent.core.utils import multiprocessing_wrapper


def shape_evaluator(target, prediction):
    if target is None or prediction is None:
        return None, 0
    tar_shape = target.shape
    pre_shape = prediction.shape

    diff = []
    for i in range(max(len(tar_shape), len(pre_shape))):
        dim_tar = tar_shape[i] if i < len(tar_shape) else 0
        dim_pre = pre_shape[i] if i < len(pre_shape) else 0
        diff.append(abs(dim_tar - dim_pre))

    metric = 1 / (np.exp(np.mean(diff)) + 1)
    return diff, metric


def reshape_tensor(original_tensor, target_shape):
    new_tensor = torch.zeros(target_shape)
    for i, dim in enumerate(original_tensor.shape):
        new_tensor = new_tensor.narrow(i, 0, dim).copy_(original_tensor)

    return new_tensor


def value_evaluator(target, prediction):
    if target is None or prediction is None:
        return None, 0
    tar_shape = target.shape
    pre_shape = prediction.shape

    # Determine the shape of the padded tensors
    dims = [
        max(s1, s2)
        for s1, s2 in zip(
            tar_shape + (1,) * (len(pre_shape) - len(tar_shape)),
            pre_shape + (1,) * (len(tar_shape) - len(pre_shape)),
        )
    ]
    # Reshape both tensors to the determined shape
    target = target.reshape(*tar_shape, *(1,) * (max(len(tar_shape), len(pre_shape)) - len(tar_shape)))
    prediction = prediction.reshape(*pre_shape, *(1,) * (max(len(tar_shape), len(pre_shape)) - len(pre_shape)))
    target_padded = reshape_tensor(target, dims)
    prediction_padded = reshape_tensor(prediction, dims)

    # Calculate the mean absolute difference
    diff = torch.abs(target_padded - prediction_padded)
    metric = 1 / (1 + np.exp(torch.mean(diff).item()))
    return diff, metric


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
        implementation: Implementation,
        gt_implementation: Implementation,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> ModelCoderFeedback:
        target_task_information = target_task.get_information()
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

        assert isinstance(implementation, ModelImplementation)
        execution_feedback, gen_tensor = implementation.execute()
        if gt_implementation is not None:
            assert isinstance(gt_implementation, ModelImplementation)
            _, gt_tensor = gt_implementation.execute()
        else:
            gt_tensor = None

        shape_feedback = shape_evaluator(gt_tensor, gen_tensor)
        value_feedback = value_evaluator(gt_tensor, gen_tensor)

        model_coder_feedback = ModelCoderFeedback(
            execution_feedback=execution_feedback,
            shape_feedback=shape_feedback,
            value_feedback=value_feedback,
            code_feedback=None,
            final_feedback=None,
            final_decision=True,
        )


class ModelCoderMultiEvaluator(Evaluator):
    def evaluate(
        self,
        evo: ModelEvolvingItem,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> List[ModelCoderFeedback]:
        multi_implementation_feedback = []

        calls = []
        for index in range(len(evo.sub_tasks)):
            corresponding_implementation = evo.sub_implementations[index]
            corresponding_gt_implementation = (
                evo.sub_gt_implementations[index] if evo.sub_gt_implementations is not None else None
            )
            calls.append(
                (
                    ModelCoderEvaluator(scen=self.scen).evaluate,
                    (
                        evo.sub_tasks[index],
                        corresponding_implementation,
                        corresponding_gt_implementation,
                        queried_knowledge,
                    ),
                ),
            )
        multi_implementation_feedback = multiprocessing_wrapper(calls, n=MODEL_IMPL_SETTINGS.evo_multi_proc_n)

        final_decision = [
            None if single_feedback is None else single_feedback.final_decision
            for single_feedback in multi_implementation_feedback
        ]
        RDAgentLog().info(f"Final decisions: {final_decision} True count: {final_decision.count(True)}")

        return multi_implementation_feedback


if __name__ == "__main__":
    tar = torch.rand(4, 5, 5)
    pre = torch.rand(4, 1)
    print(shape_evaluator(tar, pre))
    print(value_evaluator(tar, pre)[1])
