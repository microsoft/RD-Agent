"""
LLM Fine-tuning Experiment Feedback Generation

Provides feedback analysis for LLM fine-tuning experiments, including
model performance evaluation, training metrics analysis, and improvement suggestions.
"""

import json
from typing import Dict

from rdagent.app.finetune.llm.conf import FT_RD_SETTING
from rdagent.core.proposal import (
    Experiment2Feedback,
    ExperimentFeedback,
    HypothesisFeedback,
)
from rdagent.core.scenario import Scenario
from rdagent.log.utils import dict_get_with_warning
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.finetune.experiment.experiment import FTExperiment
from rdagent.scenarios.finetune.proposal.proposal import FTHypothesis
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T


class FTExperiment2Feedback(Experiment2Feedback):
    """Generate feedback for LLM fine-tuning experiments"""

    def __init__(self, scen: Scenario, version: str = "exp_feedback") -> None:
        super().__init__(scen)
        self.version = version

    def generate_feedback(self, exp: FTExperiment, trace=None, error_info: str | None = None) -> ExperimentFeedback:
        """
        Generate comprehensive feedback for LLM fine-tuning experiment.

        Args:
            exp: The experiment to analyze
            trace: Experiment trace (optional)
            error_info: If provided, indicates experiment failed and contains error details

        Note: If error_info is None, it means training succeeded and we evaluate quality/effectiveness.
              If error_info is provided, we analyze the failure cause.
        """
        # Get task information
        task_desc = exp.sub_tasks[0].get_task_information()

        if error_info is not None:
            # Error case: use error analysis prompt
            version = "exp_feedback_error"

            # Try to get FTRunnerEvaluator's analysis result from workspace
            # This contains structured feedback (execution, return_checking, code) instead of raw error string
            runner_feedback = None
            if exp.sub_workspace_list:
                for ws in exp.sub_workspace_list:
                    if ws and hasattr(ws, "feedback") and ws.feedback:
                        runner_feedback = ws.feedback
                        break

            if runner_feedback:
                # Use FTRunnerEvaluator's structured analysis result
                error_info = f"""## Execution Analysis
{runner_feedback.execution}

## Return Checking
{runner_feedback.return_checking}

## Code Analysis
{runner_feedback.code}"""

            system_prompt = T(f".prompts:{version}.system").r(
                scenario=self.scen.get_scenario_all_desc(),
            )
            # Get workspace files safely
            workspace_files = {}
            if hasattr(exp, "experiment_workspace") and exp.experiment_workspace is not None:
                workspace_files = exp.experiment_workspace.file_dict
            user_prompt = T(f".prompts:{version}.user").r(
                hypothesis=exp.hypothesis,
                task_desc=task_desc,
                workspace_files=workspace_files,
                error_info=error_info,
            )
        else:
            # Success case: use normal feedback prompt
            version = self.version
            # Process experiment result - handle both new and legacy formats
            exp_result = exp.experiment_workspace.running_info.result
            if isinstance(exp_result, dict) and "benchmark" in exp_result:
                # New format: contains benchmark and training_metrics
                benchmark = exp_result.get("benchmark", {})
                raw_metrics = exp_result.get("training_metrics", {})
                # Convert loss_history to summary to save tokens
                loss_history = raw_metrics.get("loss_history", [])
                if loss_history:
                    losses = [e["loss"] for e in loss_history]
                    min_idx = losses.index(min(losses))
                    training_metrics = {
                        "logged_entries": len(loss_history),
                        "final_step": loss_history[-1].get("step"),
                        "initial_loss": round(loss_history[0]["loss"], 4),
                        "final_loss": round(loss_history[-1]["loss"], 4),
                        "min_loss": round(min(losses), 4),
                        "min_loss_at_entry": min_idx + 1,
                        "loss_trend": "rising_late" if losses[-1] > min(losses) * 1.1 else "stable",
                    }
                else:
                    training_metrics = raw_metrics
            else:
                # Legacy format: exp_result is directly the benchmark result (list of dicts)
                benchmark = {"accuracy_summary": exp_result, "error_samples": []}
                training_metrics = {}

            system_prompt = T(f".prompts:{version}.system").r(
                scenario=self.scen.get_scenario_all_desc(),
            )
            user_prompt = T(f".prompts:{version}.user").r(
                hypothesis=exp.hypothesis,
                task_desc=task_desc,
                workspace_files=exp.experiment_workspace.file_dict,
                execution_time=exp.experiment_workspace.running_info.running_time,
                benchmark=benchmark,
                training_metrics=training_metrics,
            )

        resp_dict = json.loads(
            APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt,
                system_prompt=system_prompt,
                json_mode=True,
                json_target_type=Dict[str, str | bool | int],
            )
        )

        # Extract feedback components
        error_type = resp_dict.get("Error Type") if error_info is not None else None
        hypothesis_feedback = HypothesisFeedback(
            code_change_summary=dict_get_with_warning(resp_dict, "Code Summary", "No code summary provided"),
            reason=dict_get_with_warning(resp_dict, "Reason", "No reasoning provided"),
            decision=convert2bool(dict_get_with_warning(resp_dict, "Decision", "no")),
            acceptable=error_info is None,  # Only acceptable if no error
            observations=error_type,  # Store error type for history display
        )

        return hypothesis_feedback
