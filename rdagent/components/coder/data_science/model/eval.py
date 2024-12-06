"""
Beyond previous tests
- 
"""

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.model.eva_utils import (
    ModelCodeEvaluator,
    ModelFinalEvaluator,
)
from rdagent.components.coder.model_coder.eva_utils import shape_evaluator
from rdagent.components.coder.data_science.model.exp import ModelFBWorkspace, ModelTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Task, Workspace

ModelSingleFeedback = CoSTEERSingleFeedback
ModelMultiFeedback = CoSTEERMultiFeedback


# Below are unit tests for testing the specification of the implemented model ------------------
#
class ModelGeneralCaseSpecEvaluator(CoSTEEREvaluator):
    """
    Motivation case:
    - Simplest case, we already split the data into train_data, valid_data, and test_data. We require the model to learn (optionally validate on valid data), and infer on test data.

    Test workflow:
    - Build train, valid, and test data to run it, and test the output (e.g., shape, etc.)
    """

    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> ModelSingleFeedback:
        # target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return ModelSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                shape_feedback="This task has failed too many times, skip implementation.",
                value_feedback="This task has failed too many times, skip implementation.",
                code_feedback="This task has failed too many times, skip implementation.",
                final_feedback="This task has failed too many times, skip implementation.",
                final_decision=False,
            )
        # assert isinstance(target_task, ModelTask)
        
        batch_size = 8
        assert isinstance(implementation, ModelFBWorkspace)
        model_execution_feedback, pred_list= implementation.execute(
            batch_size=batch_size,
        )
        val_pred_array, test_pred_array = pred_list

        # TODO: auto specify shape from spec.md using GPT
        
        num_classes = self.scen.model_output_channel if hasattr(self.scen, "model_output_channel") else 1
        shape_feedback = ""  
        expected_val_shape = (batch_size, num_classes)  
        expected_test_shape = (batch_size, num_classes)  
        val_shape_feedback, val_shape_decision = shape_evaluator(
            val_pred_array,
            expected_val_shape,
        )
        shape_feedback += f"Validation Output: {val_shape_feedback}\n"
        test_shape_feedback, test_shape_decision = shape_evaluator(
            test_pred_array,
            expected_test_shape,
        )
        shape_feedback += f"Test Output: {test_shape_feedback}\n"
        value_feedback = "The value feedback is ignored, and the value decision is automatically set as true." 
        code_feedback, _ = ModelCodeEvaluator(scen=self.scen).evaluate(
            target_task=target_task,
            implementation=implementation,
            model_execution_feedback=model_execution_feedback,
        )
        final_feedback, final_decision = ModelFinalEvaluator(scen=self.scen).evaluate(
            target_task=target_task,
            implementation=implementation,
            model_execution_feedback=model_execution_feedback,
            model_shape_feedback=shape_feedback,
            model_code_feedback=code_feedback,
        )

        return ModelSingleFeedback(
            execution_feedback=model_execution_feedback,
            shape_feedback=shape_feedback,
            value_feedback=value_feedback,
            code_feedback=code_feedback,
            final_feedback=final_feedback,
            final_decision=final_decision,
            value_generated_flag=(val_pred_array is not None and test_pred_array is not None),
            final_decision_based_on_gt=False,
        )


class XXX2SpecEval:
    """
    Based on XXX1SpecEval, but considering the following case:

    Motivation case:
    - Sometimes we don't need validation (e.g., simple models not prone to overfitting, or data is too scarce to split).

    Test workflow:
    - Build train and test data to run it, and test the output (e.g., shape, etc.)
    - valid_data == None
    """


class XXX3SpecEval:
    """
    Motivation case:
    - We need to tune hyperparameters.

    Test workflow:
    - Input:
        - Build train and valid data
        - test == None
        - Hyperparameters are not blank
    - Output:
        - The early stop hyperparameters must be returned
    """


class XXX4SpecEval:
    """
    Motivation case:
    - After obtaining good hyperparameters, we retrain the model.

    Test workflow:
    - Test1: Since we have already tested it in XXX2SpecEval, we'll focus on another aspect.
        - Input:
            - Build train and test data
            - valid == None
            - Previous good hyperparameters (a parameter representing early stop)
    - Test2: Ensure the hyperparameters are 1) being used, and 2) the model remains stable.
        - Different hyperparameters will yield different results
        - Same hyperparameters will yield the same results
    """
