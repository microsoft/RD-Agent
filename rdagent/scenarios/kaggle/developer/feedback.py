import json
from pathlib import Path
import pickle

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
from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.factor_coder.config import FACTOR_IMPLEMENT_SETTINGS

feedback_prompts = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")
DIRNAME = Path(__file__).absolute().resolve().parent


def process_results(current_result, sota_result):
    # Convert the results to dataframes
    current_df = pd.DataFrame(current_result)
    sota_df = pd.DataFrame(sota_result)

    # Combine the dataframes on the Metric index
    combined_df = pd.concat([current_df, sota_df], axis=1)
    combined_df.columns = ["current_df", "sota_df"]

    combined_df["the largest"] = combined_df.apply(
        lambda row: "sota_df"
        if row["sota_df"] > row["current_df"]
        else ("Equal" if row["sota_df"] == row["current_df"] else "current_df"),
        axis=1,
    )

    # Add a note about metric direction
    combined_df["Note"] = "Direction of improvement (higher/lower is better) should be judged per metric"

    return combined_df


class KGHypothesisExperiment2Feedback(HypothesisExperiment2Feedback):
    def get_available_features(self, exp: Experiment):
        features = []
        
        # Get original features
        org_data_path = Path(FACTOR_IMPLEMENT_SETTINGS.data_folder) / KAGGLE_IMPLEMENT_SETTING.competition / "valid.pkl"
        with open(org_data_path, "rb") as f:
            org_data = pickle.load(f)
        
        for i in range(org_data.shape[-1]):
            features.append({"name": f"original_feature_{i}", "description": "Original feature"})
        
        # Get engineered features
        for feature_file in sorted(exp.experiment_workspace.workspace_path.glob("feature/feature*.py")):
            with open(feature_file, 'r') as f:
                content = f.read()
                # This is a simple extraction method. You might need to adjust this based on the actual structure of your feature files.
                for line in content.split('\n'):
                    if line.strip().startswith('X['):
                        feature_name = line.split('[')[1].split(']')[0].strip("'\"")
                        features.append({"name": feature_name, "description": "Engineered feature"})
        
        return features

    def get_model_code(self, exp: Experiment):
        model_type = exp.sub_tasks[0].model_type if exp.sub_tasks else None
        if model_type == "XGBoost":
            return exp.sub_workspace_list[0].code_dict.get("model_xgb.py")
        elif model_type == "RandomForest":
            return exp.sub_workspace_list[0].code_dict.get("model_rf.py")
        elif model_type == "LightGBM":
            return exp.sub_workspace_list[0].code_dict.get("model_lgb.py")
        elif model_type == "NN":
            return exp.sub_workspace_list[0].code_dict.get("model_nn.py")
        else:
            return None

    def generate_feedback(self, exp: Experiment, hypothesis: Hypothesis, trace: Trace) -> HypothesisFeedback:
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
            combined_result = process_results(current_result, sota_result)
        else:
            combined_result = process_results(current_result, current_result)
            print("Warning: No previous experiments to compare against. Using current result as baseline.")

        available_features = self.get_available_features(exp)

        # Generate the system prompt
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(feedback_prompts["factor_feedback_generation"]["system"])
            .render(scenario=self.scen.get_scenario_all_desc())
        )

        # Get the appropriate model code
        model_code = self.get_model_code(exp)

        # Generate the user prompt based on the action type
        if hypothesis.action in ["Model tuning", "Model feature selection"]:  
            prompt_key = "model_feedback_generation"
            render_dict = {
                "context": self.scen.get_scenario_all_desc(),
                "last_hypothesis": trace.hist[-1][0] if trace.hist else None,
                "last_task": trace.hist[-1][1] if trace.hist else None,
                "last_code": self.get_model_code(trace.hist[-1][1]) if trace.hist else None,
                "last_result": trace.hist[-1][1].result if trace.hist else None,
                "hypothesis": hypothesis,
                "exp": exp,
                "model_code": model_code,
                "available_features": available_features,
            }
        else:
            prompt_key = "factor_feedback_generation"
            render_dict = {
                "hypothesis_text": hypothesis_text,
                "task_details": tasks_factors,
                "combined_result": combined_result,
            }
        
        # Generate the user prompt
        usr_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(feedback_prompts[prompt_key]["user"])
            .render(**render_dict)
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
        decision = convert2bool(response_json.get("Decision", "false"))

        return HypothesisFeedback(
            observations=observations,
            hypothesis_evaluation=hypothesis_evaluation,
            new_hypothesis=new_hypothesis,
            reason=reason,
            decision=decision,
        )
