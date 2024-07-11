# TODO:
# Implement to feedback.

from pathlib import Path

from jinja2 import Environment, StrictUndefined
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import HypothesisExperiment2Feedback
from rdagent.core.proposal import Trace
from rdagent.core.experiment import Experiment
from rdagent.core.proposal import Hypothesis, HypothesisFeedback
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.env import QTDockerEnv
from rdagent.core.log import RDAgentLog
import json
import pandas as pd
import pickle

feedback_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")
DIRNAME = Path(__file__).absolute().resolve().parent
logger = RDAgentLog()

class QlibModelHypothesisExperiment2Feedback(HypothesisExperiment2Feedback): ...

class QlibFactorHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    def generateFeedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        Generate feedback for the given experiment and hypothesis.

        Args:
            exp (QlibFactorExperiment): The experiment to generate feedback for.
            hypothesis (QlibFactorHypothesis): The hypothesis to generate feedback for.
            trace (Trace): The trace of the experiment.

        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = [task.get_factor_information() for task in exp.sub_tasks]
        sota_result = exp.based_experiments[-1].result

        # Generate the system prompt
        sys_prompt = Environment(undefined=StrictUndefined).from_string(feedback_prompts["data_feedback_generation"]["system"]).render(scenario=self.scen.get_scenario_all_desc())

        # Generate the user prompt
        usr_prompt = Environment(undefined=StrictUndefined).from_string(feedback_prompts["data_feedback_generation"]["user"]).render(
            hypothesis_text=hypothesis_text,
            task_details=tasks_factors,
            current_result=current_result,
            sota_result=sota_result
        )

        # Call the APIBackend to generate the response for hypothesis feedback
        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        # Parse the JSON response to extract the feedback
        response_json = json.loads(response)
        
        # Extract fields from JSON response
        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = response_json.get("Replace Best Result", "no").lower() == "yes"
        
        # Create HypothesisFeedback object
        hypothesis_feedback = HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision
        )

        logger.info(
            "Generated Hypothesis Feedback:\n"
            f"Observations: {observations}\n"
            f"Feedback for Hypothesis: {hypothesis_evaluation}\n"
            f"New Hypothesis: {new_hypothesis}\n"
            f"Reason: {reason}\n"
            f"Replace Best Result: {'Yes' if decision else 'No'}"
        )

        return hypothesis_feedback
