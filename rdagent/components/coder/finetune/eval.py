"""
LLM Fine-tuning Evaluation Components

Provides simplified evaluation: parameter filtering + micro-batch testing.
No redundant LLM feedback generation - test results speak for themselves.
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

DIRNAME = Path(__file__).absolute().resolve().parent


class FTCoderEvaluator(CoSTEEREvaluator):
    """Evaluator for LLM fine-tuning implementations with simplified validation"""

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
        """Evaluate LLM fine-tuning implementation with two-step validation"""

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

        # Two-step validation: parameter filtering + micro-batch test
        validation_result = self.config_validator.validate_and_test(
            config_yaml=config_yaml, workspace=implementation, env=env
        )

        validation_report = self.config_validator.generate_validation_report(validation_result)
        logger.info(f"Validation report:\n{validation_report}")

        # Update config with filtered version
        if validation_result.filtered_config != config_yaml:
            implementation.inject_files(**{"train.yaml": validation_result.filtered_config})

        # Return feedback directly from test results - no LLM interpretation needed
        return CoSTEERSingleFeedback(
            execution=validation_result.execution_output[:1000] if validation_result.execution_output else "No output",
            return_checking=f"Validation {'passed' if validation_result.success else 'failed'} in {validation_result.execution_time:.2f}s",
            code=(
                "; ".join(validation_result.errors)
                if validation_result.errors
                else "Configuration validated successfully"
            ),
            final_decision=validation_result.success,
        )
