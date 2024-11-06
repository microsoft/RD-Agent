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

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")
DIRNAME = Path(__file__).absolute().resolve().parent


class KGHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    def process_results(self, current_result, sota_result):
        # Convert the results to dataframes
        current_df = pd.DataFrame(current_result)
        sota_df = pd.DataFrame(sota_result)

        # Combine the dataframes on the Metric index
        combined_df = pd.concat([current_df, sota_df], axis=1)
        combined_df.columns = ["current_df", "sota_df"]

        # combined_df["the largest"] = combined_df.apply(
        #     lambda row: "sota_df"
        #     if row["sota_df"] > row["current_df"]
        #     else ("Equal" if row["sota_df"] == row["current_df"] else "current_df"),
        #     axis=1,
        # )

        # Add a note about metric direction
        evaluation_direction = "higher" if self.scen.evaluation_metric_direction else "lower"
        evaluation_description = f"Direction of improvement (higher/lower is better) should be judged per metric. Here '{evaluation_direction}' is better for the metrics."
        combined_df["Note"] = evaluation_description

        return combined_df, evaluation_description

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

        evaluation_description = None
        # Check if there are any based experiments
        if exp.based_experiments:
            sota_result = exp.based_experiments[-1].result
            # Process the results to filter important metrics
            combined_result, evaluation_description = self.process_results(current_result, sota_result)
        else:
            # If there are no based experiments, we'll only use the current result
            combined_result, evaluation_description = self.process_results(
                current_result, current_result
            )  # Compare with itself
            print("Warning: No previous experiments to compare against. Using current result as baseline.")

        available_features = {
            task_info: feature_shape for task_info, feature_shape in exp.experiment_workspace.data_description
        }
        model_code = exp.experiment_workspace.model_description

        # Generate the user prompt based on the action type
        if hypothesis.action == "Model tuning":
            prompt_key = "model_tuning_feedback_generation"
        elif hypothesis.action == "Model feature selection":
            prompt_key = "feature_selection_feedback_generation"
        else:
            prompt_key = "factor_feedback_generation"

        # Generate the system prompt
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict[prompt_key]["system"])
            .render(scenario=self.scen.get_scenario_all_desc(filtered_tag="feedback"))
        )

        last_task_and_code = None
        if trace.hist:
            last_task_and_code = (
                trace.hist[-1][1].experiment_workspace.data_description
                if trace.hist[-1][0].action == "Feature engineering" or trace.hist[-1][0].action == "Feature processing"
                else trace.hist[-1][1].experiment_workspace.model_description
            )

        # Prepare render dictionary
        render_dict = {
            "last_hypothesis": trace.hist[-1][0] if trace.hist else None,
            "last_task_and_code": last_task_and_code,
            "last_result": trace.hist[-1][1].result if trace.hist else None,
            "sota_task_and_code": (
                exp.based_experiments[-1].experiment_workspace.data_description if exp.based_experiments else None
            ),
            "sota_result": exp.based_experiments[-1].result if exp.based_experiments else None,
            "hypothesis": hypothesis,
            "exp": exp,
            "model_code": model_code,  # This turn
            "available_features": available_features,  # This turn
            "combined_result": combined_result,  # This turn and sota
            "hypothesis_text": hypothesis_text,  # This turn
            "task_details": tasks_factors,  # This turn
            "evaluation_description": evaluation_description,
        }

        usr_prompt = (
            Environment(undefined=StrictUndefined).from_string(prompt_dict[prompt_key]["user"]).render(**render_dict)
        )

        response = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=usr_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        response_json = json.loads(response)

        observations = response_json.get("Observations", "No observations provided")
        hypothesis_evaluation = response_json.get("Feedback for Hypothesis", "No feedback provided")
        new_hypothesis = response_json.get("New Hypothesis", "No new hypothesis provided")
        reason = response_json.get("Reasoning", "No reasoning provided")
        decision = convert2bool(response_json.get("Replace Best Result", "no"))
        leaderboard = self.scen.leaderboard
        current_score = current_result.iloc[0]
        sorted_scores = sorted(leaderboard, reverse=True)
        import bisect

        if self.scen.evaluation_metric_direction:
            insert_position = bisect.bisect_right([-score for score in sorted_scores], -current_score)
        else:
            insert_position = bisect.bisect_left(sorted_scores, current_score, lo=0, hi=len(sorted_scores))
        percentile_ranking = (insert_position) / (len(sorted_scores)) * 100

        experiment_feedback = {
            "current_competition": self.scen.get_competition_full_desc(),
            "hypothesis_text": hypothesis_text,
            "current_result": current_result,
            "model_code": model_code,
            "available_features": available_features,
            "observations": observations,
            "hypothesis_evaluation": hypothesis_evaluation,
            "reason": reason,
            "percentile_ranking": percentile_ranking,
        }

        if self.scen.if_using_vector_rag:
            self.scen.vector_base.add_experience_to_vector_base(experiment_feedback)
            self.scen.vector_base.dump()
        elif self.scen.if_using_graph_rag:
            trace.knowledge_base.add_document(experiment_feedback, self.scen)

        if self.scen.if_action_choosing_based_on_UCB:
            self.scen.action_counts[hypothesis.action] += 1

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )
