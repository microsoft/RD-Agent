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

    def generate_feedback(self, exp: FTExperiment, trace=None) -> ExperimentFeedback:
        """
        Generate comprehensive feedback for LLM fine-tuning experiment.

        Note: If this method is called, it means training has already succeeded
        (runner.develop() returned without exception). We only evaluate the quality/effectiveness.
        """

        # Get experiment hypothesis
        hypothesis = exp.hypothesis
        if not isinstance(hypothesis, FTHypothesis):
            # Simple fallback feedback for non-LLM hypotheses
            return ExperimentFeedback(
                reason="Non-LLM hypothesis detected. Basic feedback provided.",
                decision=False,
            )

        # Get task information
        task = exp.sub_tasks[0] if exp.sub_tasks else None
        task_desc = task.get_task_information() if task else "No task information available"

        # Get workspace files and execution results
        workspace_files = list(exp.experiment_workspace.file_dict.keys()) if exp.experiment_workspace else []
        execution_analysis = self._analyze_execution_results(exp)

        # Generate LLM-based feedback using prompts.yaml templates
        system_prompt = T(f".prompts:{self.version}.system").r(
            scenario=(
                self.scen.get_scenario_all_desc() if hasattr(self.scen, "get_scenario_all_desc") else str(self.scen)
            ),
            task_desc=task_desc,
        )
        user_prompt = T(f".prompts:{self.version}.user").r(
            hypothesis=hypothesis,
            workspace_files=workspace_files,
            execution_analysis=execution_analysis,
            base_model=hypothesis.base_model,
            finetune_method=hypothesis.finetune_method,
            dataset=getattr(hypothesis, "dataset", FT_RD_SETTING.dataset),
            task_desc=task_desc,
        )

        try:
            resp_dict = json.loads(
                APIBackend().build_messages_and_create_chat_completion(
                    user_prompt=user_prompt,
                    system_prompt=system_prompt,
                    json_mode=True,
                    json_target_type=Dict[str, str | bool | int],
                )
            )

            # Extract feedback components
            hypothesis_feedback = HypothesisFeedback(
                observations=dict_get_with_warning(resp_dict, "Observations", "No observations provided"),
                hypothesis_evaluation=dict_get_with_warning(
                    resp_dict, "Hypothesis Evaluation", "No evaluation provided"
                ),
                new_hypothesis=dict_get_with_warning(resp_dict, "New Hypothesis", "No new hypothesis provided"),
                reason=dict_get_with_warning(resp_dict, "Reasoning", "No reasoning provided"),
                code_change_summary=dict_get_with_warning(resp_dict, "Code Summary", "No code summary provided"),
                decision=convert2bool(dict_get_with_warning(resp_dict, "Accept Experiment", "no")),
                acceptable=convert2bool(dict_get_with_warning(resp_dict, "Overall Acceptable", "no")),
            )

            return hypothesis_feedback

        except Exception as e:
            # Fallback feedback in case of LLM failure
            return ExperimentFeedback(
                reason=f"Failed to generate LLM feedback: {str(e)}. Using fallback evaluation.",
                decision=execution_analysis.get("success", False),
            )

    def _analyze_execution_results(self, exp: FTExperiment) -> Dict:
        """
        Extract execution information from runner's results for LLM-based feedback.

        Note: This method only reads files from workspace, it does NOT run any evaluation.
        All training and evaluation should be completed by runner before reaching this point.
        """
        analysis = {
            "success": True,  # Training succeeded (runner returned without exception)
            "execution_time": 0,
            "error_analysis": "",
            "model_outputs": [],
        }

        workspace = exp.experiment_workspace
        if not workspace:
            analysis["error_analysis"] = "No workspace available"
            analysis["success"] = False
            return analysis

        # Extract execution time
        if hasattr(workspace, "running_info") and workspace.running_info:
            analysis["execution_time"] = getattr(workspace.running_info, "running_time", 0)

        # Extract file information from workspace
        # Runner has already completed training and evaluation
        try:
            task = exp.sub_tasks[0] if exp.sub_tasks else None
            if task:
                # Read output files generated by runner
                workspace_path = workspace.workspace_path
                model_files = []
                for pattern in ["*.safetensors", "*.bin", "adapter_*", "training_*.json"]:
                    model_files.extend(workspace_path.glob(pattern))

                if model_files:
                    analysis["model_outputs"] = [f.name for f in model_files]
                    analysis["error_analysis"] = (
                        f"Training completed successfully. Generated {len(model_files)} output files."
                    )
                else:
                    analysis["error_analysis"] = "Training completed but no model output files found."

        except Exception as e:
            from rdagent.log import rdagent_logger as logger

            logger.error(f"Failed to analyze execution results: {str(e)}")
            analysis["error_analysis"] = f"Failed to extract execution details: {str(e)}"

        return analysis
