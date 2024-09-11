import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.core.experiment import Experiment
from rdagent.core.prompts import Prompts
from rdagent.core.proposal import (
    Hypothesis,
    HypothesisExperiment2Feedback,
    HypothesisFeedback,
    Trace,
)
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils import convert2bool

feedback_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")
DIRNAME = Path(__file__).absolute().resolve().parent


def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Combine the dataframes on the Metric index
    combined_df = pd.DataFrame({"Current Result": current_df, "SOTA Result": sota_df})

    # Add a new column to show which result is bigger
    combined_df["Bigger Result"] = combined_df.apply(
        lambda row: "Equal"
        if row["Current Result"] == row["SOTA Result"]
        else ("Current Result" if row["Current Result"] > row["SOTA Result"] else "SOTA Result"),
        axis=1,
    )

    # Add a note about metric direction
    combined_df["Note"] = "Direction of improvement (higher/lower is better) should be judged per metric"

    return combined_df.to_string()


class KGHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
        """
        The `ti` should be executed and the results should be included, as well as the comparison between previous results (done by LLM).
        For example: `mlflow` of Qlib will be included.
        """

        """
        Generate feedback for the given experiment and hypothesis.
        Args:
            exp: The experiment to generate feedback for.
            hypothesis: The hypothesis to generate feedback for.
            trace: The trace of the experiment.
        Returns:
            Any: The feedback generated for the given experiment and hypothesis.
        """
        logger.info("Generating feedback...")
        hypothesis_text = hypothesis.hypothesis
        current_result = exp.result
        tasks_factors = []
        if exp.sub_tasks:
            tasks_factors = []
            for task in exp.sub_tasks:
                try:
                    task_info = task.get_task_information_and_implementation_result()
                    tasks_factors.append(task_info)
                except AttributeError:
                    print(f"Warning: Task {task} does not have get_task_information_and_implementation_result method")

        # Check if there are any based experiments
        if exp.based_experiments:
            sota_result = exp.based_experiments[-1].result
            # Process the results to filter important metrics
            combined_result = process_results(current_result, sota_result)
        else:
            # If there are no based experiments, we'll only use the current result
            combined_result = process_results(current_result, current_result)  # Compare with itself
            print("Warning: No previous experiments to compare against. Using current result as baseline.")

        # Generate the system prompt
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )

        # Generate the user prompt based on the action type
        if hypothesis.action == "Model Tuning":  # TODO Add other prompts here
            prompt_key = "model_feedback_generation"
        else:
            prompt_key = "factor_feedback_generation"

        # Generate the user prompt
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(feedback_prompts[prompt_key]["user"])
            .render(
                hypothesis_text=hypothesis_text,
                task_details=tasks_factors,
                combined_result=combined_result,
            )
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
        decision = convert2bool(response_json.get("Replace Best Result", "no"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )
