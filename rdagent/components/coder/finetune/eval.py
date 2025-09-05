"""
LLM Fine-tuning Evaluation Components

Provides unified evaluation functionality for LLM fine-tuning tasks
with comprehensive configuration validation.
"""

from pathlib import Path

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import get_ft_env
from rdagent.components.coder.finetune.unified_validator import create_unified_validator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent


class LLMFinetuneEvaluator(CoSTEEREvaluator):
    """Evaluator for LLM fine-tuning implementations with unified validation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.config_validator = create_unified_validator()

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        """Evaluate LLM fine-tuning implementation"""

        task_info = target_task.get_task_information()

        # Check task history
        if queried_knowledge is not None:
            if task_info in queried_knowledge.success_task_to_knowledge_dict:
                return queried_knowledge.success_task_to_knowledge_dict[task_info].feedback
            elif task_info in queried_knowledge.failed_task_info_set:
                return CoSTEERSingleFeedback(
                    execution="Task failed too many times, skipping.",
                    return_checking="Task failed too many times, skipping.",
                    code="Task failed too many times, skipping.",
                    final_decision=False,
                )

        env = get_ft_env(
            running_timeout_period=self.scen.real_debug_timeout() if hasattr(self.scen, "real_debug_timeout") else 3600,
        )

        config_yaml = implementation.file_dict.get("train.yaml", "")
        if not config_yaml:
            return CoSTEERSingleFeedback(
                execution="No train.yaml found",
                return_checking="Configuration file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        # Run comprehensive validation with micro-batch test
        validation_result = self.config_validator.validate_config_comprehensive(
            config_yaml=config_yaml, enable_micro_batch_test=True, workspace=implementation, env=env
        )

        validation_report = self.config_validator.generate_validation_report(validation_result)
        logger.info(f"Validation report:\n{validation_report}")

        # Update config if needed
        if validation_result.success and validation_result.filtered_config != config_yaml:
            implementation.inject_files(**{"train.yaml": validation_result.filtered_config})

        # Generate LLM feedback
        system_prompt = T(".prompts:finetune_eval.system").r(
            task_desc=task_info,
            code=validation_result.filtered_config,
        )

        user_prompt = T(".prompts:finetune_eval.user").r(stdout=validation_report)

        try:
            feedback = build_cls_from_json_with_retry(
                CoSTEERSingleFeedback,
                system_prompt=system_prompt,
                user_prompt=user_prompt,
                init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
            )
            feedback.final_decision = feedback.final_decision and validation_result.success
            return feedback
        except Exception as e:
            logger.error(f"Evaluation feedback generation failed: {e}")
            return CoSTEERSingleFeedback(
                execution=f"Evaluation failed: {str(e)}",
                return_checking="Could not evaluate due to evaluation failure.",
                code="Could not evaluate code quality due to evaluation failure.",
                final_decision=False,
            )
