"""
LLM Fine-tuning Evaluation Components

Provides simplified evaluation: parameter filtering + micro-batch testing.
No redundant LLM feedback generation - test results speak for themselves.
"""

from pathlib import Path
from typing import Optional

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.finetune.conf import FT_YAML_FILE_NAME, get_ft_env
from rdagent.components.coder.finetune.unified_validator import LLMConfigValidator
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent


class FTDataEvaluator(CoSTEEREvaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: Optional[QueriedKnowledge] = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:
        # TODO: we just have a dummy evaluator for now
        return CoSTEERSingleFeedback(execution="", return_checking="", code="", final_decision=True)
        # return CoSTEERSingleFeedback(execution="data failed", return_checking="data failed", code="data failed", final_decision=False)


class FTCoderEvaluator(CoSTEEREvaluator):
    """Evaluator for LLM fine-tuning implementations with simplified validation"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

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
        config_yaml = implementation.file_dict.get(FT_YAML_FILE_NAME, "")
        if not config_yaml:
            return CoSTEERSingleFeedback(
                execution=f"No {FT_YAML_FILE_NAME} found",
                return_checking="Configuration file missing",
                code="No valid configuration file",
                final_decision=False,
            )

        # Two-step validation: parameter filtering + micro-batch test
        validation_result = LLMConfigValidator().validate_and_test(
            config_yaml=config_yaml, workspace=implementation, env=env
        )

        # Update config with filtered version
        if validation_result.filtered_config != config_yaml:
            implementation.inject_files(**{FT_YAML_FILE_NAME: validation_result.filtered_config})

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_task.get_task_information()]
            if queried_knowledge is not None
            else []
        )

        system_prompt = T(".prompts:finetune_eval.system").r(
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
        )
        user_prompt = T(".prompts:finetune_eval.user").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            stdout=validation_result.execution_output or "No output",
            code_yaml=implementation.file_dict[FT_YAML_FILE_NAME],
            workspace_files="\n".join(
                [
                    f"- {file.name} ({file.stat().st_size} bytes)"
                    for file in implementation.workspace_path.rglob("*")
                    if file.is_file() and "checkpoint" not in file.absolute().as_posix()
                ]
            ),
        )
        return build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=CoSTEERSingleFeedback.val_and_update_init_dict,
        )
