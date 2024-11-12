import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.knowledge_management.graph import UndirectedNode
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
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import KG_SELECT_MAPPING
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
        current_result = exp.result

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

        sota_exp = exp.based_experiments[-1] if exp.based_experiments else None
        assert sota_exp is not None
        sota_features = str(exp.based_experiments[-1].experiment_workspace.data_description)
        sota_models = json.dumps(exp.based_experiments[-1].experiment_workspace.model_description, indent=2)
        sota_result = exp.based_experiments[-1].result
        sota_sub_results = exp.based_experiments[-1].sub_results

        current_hypothesis = hypothesis.hypothesis
        current_hypothesis_reason = hypothesis.reason
        current_target_action = hypothesis.action
        current_sub_exps_to_code = {}
        if hypothesis.action == "Model tuning":
            current_sub_exps_to_code[exp.sub_tasks[0].get_task_information()] = exp.sub_workspace_list[0].code
        elif hypothesis.action == "Model feature selection":
            current_sub_exps_to_code[exp.sub_tasks[0].get_task_information()] = exp.experiment_workspace.code_dict[
                KG_SELECT_MAPPING[exp.sub_tasks[0].model_type]
            ]
        else:
            current_sub_exps_to_code = {
                sub_ws.target_task.get_task_information(): sub_ws.code for sub_ws in exp.sub_workspace_list
            }
        current_sub_exps_to_code_str = json.dumps(current_sub_exps_to_code, indent=2)
        current_result = exp.result
        current_sub_results = exp.sub_results

        last_hypothesis_and_feedback = None
        if trace.hist and len(trace.hist) > 0:
            last_hypothesis_and_feedback = (trace.hist[-1][0], trace.hist[-1][2])

        # Prepare render dictionary
        render_dict = {
            "sota_features": sota_features,
            "sota_models": sota_models,
            "sota_result": sota_result,
            "sota_sub_results": sota_sub_results,
            "current_hypothesis": current_hypothesis,
            "current_hypothesis_reason": current_hypothesis_reason,
            "current_target_action": current_target_action,
            "current_sub_exps_to_code": current_sub_exps_to_code_str,
            "current_result": current_result,
            "current_sub_results": current_sub_results,
            "combined_result": combined_result,
            "evaluation_description": evaluation_description,
            "last_hypothesis_and_feedback": last_hypothesis_and_feedback,
        }

        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["kg_feedback_generation_user"])
            .render(**render_dict)
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
            "hypothesis_text": current_hypothesis,
            "tasks_factors": current_sub_exps_to_code,
            "current_result": current_result,
        }

        if self.scen.if_using_vector_rag:
            raise NotImplementedError("Vector RAG is not implemented yet since there are plenty bugs!")
            self.scen.vector_base.add_experience_to_vector_base(experiment_feedback)
            self.scen.vector_base.dump()
        elif self.scen.if_using_graph_rag:
            competition_node = UndirectedNode(content=self.scen.get_competition_full_desc(), label="competition")
            hypothesis_node = UndirectedNode(content=hypothesis.hypothesis, label=hypothesis.action)
            exp_code_nodes = []
            for exp, code in current_sub_exps_to_code.items():
                exp_code_nodes.append(UndirectedNode(content=exp, label="experiments"))
                if code != "":
                    exp_code_nodes.append(UndirectedNode(content=code, label="code"))
            conclusion_node = UndirectedNode(content=response, label="conclusion")
            all_nodes = [competition_node, hypothesis_node, *exp_code_nodes, conclusion_node]
            all_nodes = trace.knowledge_base.batch_embedding(all_nodes)
            for node in all_nodes:
                if node is not competition_node:
                    trace.knowledge_base.add_node(node, competition_node)

        if self.scen.if_action_choosing_based_on_UCB:
            self.scen.action_counts[hypothesis.action] += 1

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )
