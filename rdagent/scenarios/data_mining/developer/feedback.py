# TODO:
# Implement to feedback.

import json
from pathlib import Path
from typing import Dict

from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Experiment2Feedback, HypothesisFeedback, Trace
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils import convert2bool
from rdagent.utils.agent.tpl import T
from rdagent.scenarios.data_mining.experiment.model_experiment import DMModelScenario

DIRNAME = Path(__file__).absolute().resolve().parent


class DMModelExperiment2Feedback(Experiment2Feedback):
    """Generated feedbacks on the hypothesis from **Executed** Implementations of different tasks & their comparisons with previous performances"""

    def generate_feedback(self, exp: Experiment, trace: Trace) -> HypothesisFeedback:
        """
        The `ti` should be executed and the results should be included, as well as the comparison between previous results (done by LLM).
        For example: `mlflow` of Qlib will be included.
        """
        hypothesis = exp.hypothesis

        logger.info("Generating feedback...")
        # Define the system prompt for hypothesis feedback
        sys_prompt = T("scenarios.qlib.prompts:factor_feedback_generation.system").r(
            scenario=self.scen.get_scenario_all_desc()
        )

        # Define the user prompt for hypothesis feedback
        context = trace.scen
        SOTA_hypothesis, SOTA_experiment = trace.get_sota_hypothesis_and_experiment()

        import pdb; pdb.set_trace()
        user_prompt = T("scenarios.data_mining.prompts:model_feedback_generation.user").r(
            sota_hypothesis=SOTA_hypothesis,
            sota_task=SOTA_experiment.sub_tasks[0].get_task_information() if SOTA_hypothesis else None,
            sota_code=SOTA_experiment.sub_workspace_list[0].file_dict.get("model.py") if SOTA_hypothesis else None,
            sota_result=SOTA_experiment.result if SOTA_hypothesis else None,
            hypothesis=hypothesis,
            exp=exp,
            exp_result=exp.result if exp.result is not None else "execution failed",
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response_hypothesis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | bool | int],
        )

        # Parse the JSON response to extract the feedback
        response_json_hypothesis = json.loads(response_hypothesis)
        return HypothesisFeedback(
            observations=response_json_hypothesis.get("Observations", "No observations provided"),
            hypothesis_evaluation=response_json_hypothesis.get("Feedback for Hypothesis", "No feedback provided"),
            new_hypothesis=response_json_hypothesis.get("New Hypothesis", "No new hypothesis provided"),
            reason=response_json_hypothesis.get("Reasoning", "No reasoning provided"),
            decision=convert2bool(response_json_hypothesis.get("Decision", "false")),
        )
