from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.model_coder.eva_utils import (
    ModelCodeEvaluator,
    ModelFinalEvaluator,
    shape_evaluator,
    value_evaluator,
)
from rdagent.components.coder.model_coder.model import ModelFBWorkspace, ModelTask
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import Task, Workspace

ModelSingleFeedback = CoSTEERSingleFeedback
ModelMultiFeedback = CoSTEERMultiFeedback


class ModelCoSTEEREvaluator(CoSTEEREvaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: Workspace,
        gt_implementation: Workspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> ModelSingleFeedback:
        target_task_information = target_task.get_task_information()
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

        shape_feedback, shape_decision = shape_evaluator(
            gen_np_array,
            (batch_size, self.scen.model_output_channel if hasattr(self.scen, "model_output_channel") else 1),
        )
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
            model_shape_feedback=shape_feedback,
            model_value_feedback=value_feedback,
            model_code_feedback=code_feedback,
        )

        return ModelSingleFeedback(
            execution_feedback=model_execution_feedback,
            shape_feedback=shape_feedback,
            value_feedback=value_feedback,
            code_feedback=code_feedback,
            final_feedback=final_feedback,
            final_decision=final_decision,
            value_generated_flag=(gen_np_array is not None),
            final_decision_based_on_gt=(gt_implementation is not None),
        )
