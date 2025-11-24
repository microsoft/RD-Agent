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
        # Get task information
        task_desc = exp.sub_tasks[0].get_task_information()

        # Generate LLM-based feedback using prompts.yaml templates
        system_prompt = T(f".prompts:{self.version}.system").r(
            scenario=self.scen.get_scenario_all_desc(),
        )
        user_prompt = T(f".prompts:{self.version}.user").r(
            hypothesis=exp.hypothesis,
            task_desc=task_desc,
            workspace_files=exp.experiment_workspace.file_dict,
            execution_time=exp.experiment_workspace.running_info.running_time,
            exp_result=exp.experiment_workspace.running_info.result,
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
        hypothesis_feedback = HypothesisFeedback(
            code_change_summary=dict_get_with_warning(resp_dict, "Code Summary", "No code summary provided"),
            reason=dict_get_with_warning(resp_dict, "Reason", "No reasoning provided"),
            decision=convert2bool(dict_get_with_warning(resp_dict, "Decision", "no")),
            acceptable=True,  # Always True for FT experiments if training succeeded
        )

        return hypothesis_feedback
