import io
import json
import pickle
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.core.experiment import Task
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import KGFactorExperiment
from rdagent.scenarios.kaggle.kaggle_crawler import (
    crawl_descriptions,
    leaderboard_scores,
)
from rdagent.scenarios.kaggle.knowledge_management.vector_base import (
    KaggleExperienceBase,
)

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")

KG_ACTION_FEATURE_PROCESSING = "Feature processing"
KG_ACTION_FEATURE_ENGINEERING = "Feature engineering"
KG_ACTION_MODEL_FEATURE_SELECTION = "Model feature selection"
KG_ACTION_MODEL_TUNING = "Model tuning"
KG_ACTION_LIST = [
    KG_ACTION_FEATURE_PROCESSING,
    KG_ACTION_FEATURE_ENGINEERING,
    KG_ACTION_MODEL_FEATURE_SELECTION,
    KG_ACTION_MODEL_TUNING,
]


class KGScenario(Scenario):
    def __init__(self, competition: str) -> None:
        super().__init__()
        self.competition = competition
        self.competition_descriptions = crawl_descriptions(competition)
        self.input_shape = None

        self.competition_type = None
        self.competition_description = None
        self.target_description = None
        self.competition_features = None
        self.submission_specifications = None
        self.model_output_channel = None
        self.evaluation_desc = None
        self.leaderboard = leaderboard_scores(competition)
        self.evaluation_metric_direction = float(self.leaderboard[0]) > float(self.leaderboard[-1])
        self.vector_base = None
        self.mini_case = KAGGLE_IMPLEMENT_SETTING.mini_case
        self._analysis_competition_description()
        self.if_action_choosing_based_on_UCB = KAGGLE_IMPLEMENT_SETTING.if_action_choosing_based_on_UCB
        self.if_using_graph_rag = KAGGLE_IMPLEMENT_SETTING.if_using_graph_rag
        self.if_using_vector_rag = KAGGLE_IMPLEMENT_SETTING.if_using_vector_rag

        if self.if_using_vector_rag and KAGGLE_IMPLEMENT_SETTING.rag_path:
            self.vector_base = KaggleExperienceBase(KAGGLE_IMPLEMENT_SETTING.rag_path)
            self.vector_base.path = Path(datetime.now(timezone.utc).strftime("%Y-%m-%d-%H-%M-%S") + "_kaggle_kb.pkl")
            self.vector_base.dump()

        self.action_counts = dict.fromkeys(KG_ACTION_LIST, 0)
        self.reward_estimates = {action: 0.0 for action in KG_ACTION_LIST}
        # self.reward_estimates["Model feature selection"] = 0.2
        # self.reward_estimates["Model tuning"] = 1.0
        self.reward_estimates["Feature processing"] = 0.2
        self.reward_estimates["Feature engineering"] = 1.0
        self.confidence_parameter = 1.0
        self.initial_performance = 0.0

    def _analysis_competition_description(self):
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["kg_description_template"]["system"])
            .render()
        )

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["kg_description_template"]["user"])
            .render(
                competition_descriptions=self.competition_descriptions,
                raw_data_information=self.source_data,
                evaluation_metric_direction=self.evaluation_metric_direction,
            )
        )

        response_analysis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        response_json_analysis = json.loads(response_analysis)
        self.competition_type = response_json_analysis.get("Competition Type", "No type provided")
        self.competition_description = response_json_analysis.get("Competition Description", "No description provided")
        self.target_description = response_json_analysis.get("Target Description", "No target provided")
        self.competition_features = response_json_analysis.get("Competition Features", "No features provided")
        self.submission_specifications = response_json_analysis.get(
            "Submission Specifications", "No submission requirements provided"
        )
        self.model_output_channel = response_json_analysis.get("Submission channel number to each sample", 1)
        self.evaluation_desc = response_json_analysis.get(
            "Evaluation Description", "No evaluation specification provided."
        )

    def get_competition_full_desc(self) -> str:
        evaluation_direction = "higher the better" if self.evaluation_metric_direction else "lower the better"
        return f"""Competition Type: {self.competition_type}
    Competition Description: {self.competition_description}
    Target Description: {self.target_description}
    Competition Features: {self.competition_features}
    Submission Specifications: {self.submission_specifications}
    Model Output Channel: {self.model_output_channel}
    Evaluation Descriptions: {self.evaluation_desc}
    Is the evaluation metric the higher the better: {evaluation_direction}
    """

    @property
    def background(self) -> str:
        background_template = prompt_dict["kg_background"]

        train_script = (
            Path(__file__).parent / f"{KAGGLE_IMPLEMENT_SETTING.competition}_template" / "train.py"
        ).read_text()

        background_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(background_template)
            .render(
                train_script=train_script,
                competition_type=self.competition_type,
                competition_description=self.competition_description,
                target_description=self.target_description,
                competition_features=self.competition_features,
                submission_specifications=self.submission_specifications,
                evaluation_desc=self.evaluation_desc,
                evaluate_bool=self.evaluation_metric_direction,
            )
        )
        return background_prompt

    @property
    def source_data(self) -> str:
        data_folder = Path(KAGGLE_IMPLEMENT_SETTING.local_data_path) / self.competition

        if not (data_folder / "X_valid.pkl").exists():
            preprocess_experiment = KGFactorExperiment([])
            (
                X_train,
                X_valid,
                y_train,
                y_valid,
                X_test,
                *others,
            ) = preprocess_experiment.experiment_workspace.generate_preprocess_data()

            data_folder.mkdir(exist_ok=True, parents=True)
            pickle.dump(X_train, open(data_folder / "X_train.pkl", "wb"))
            pickle.dump(X_valid, open(data_folder / "X_valid.pkl", "wb"))
            pickle.dump(y_train, open(data_folder / "y_train.pkl", "wb"))
            pickle.dump(y_valid, open(data_folder / "y_valid.pkl", "wb"))
            pickle.dump(X_test, open(data_folder / "X_test.pkl", "wb"))
            pickle.dump(others, open(data_folder / "others.pkl", "wb"))

        X_valid = pd.read_pickle(data_folder / "X_valid.pkl")
        # TODO: Hardcoded for now, need to be fixed
        if self.competition == "feedback-prize-english-language-learning":
            return "This is a sparse matrix of descriptive text."

        buffer = io.StringIO()
        X_valid.info(verbose=True, buf=buffer, show_counts=False)
        data_info = buffer.getvalue()
        self.input_shape = X_valid.shape
        return data_info

    def output_format(self, tag=None) -> str:
        assert tag in [None, "feature", "model"]
        feature_output_format = f"""The feature code should output following the format:
{prompt_dict['kg_feature_output_format']}"""
        model_output_format = f"""The model code should output following the format:\n""" + (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["kg_model_output_format"])
            .render(channel=self.model_output_channel)
        )
        if tag is None:
            return feature_output_format + "\n" + model_output_format
        elif tag == "feature":
            return feature_output_format
        elif tag == "model":
            return model_output_format

    def interface(self, tag=None) -> str:
        assert tag in [None, "feature", "XGBoost", "RandomForest", "LightGBM", "NN"]
        feature_interface = f"""The feature code should follow the interface:
{prompt_dict['kg_feature_interface']}"""
        if tag == "feature":
            return feature_interface

        model_interface = "The model code should follow the interface:\n" + (
            Environment(undefined=StrictUndefined).from_string(prompt_dict["kg_model_interface"]).render(tag=tag)
        )
        if tag is None:
            return feature_interface + "\n" + model_interface
        else:
            return model_interface

    def simulator(self, tag=None) -> str:
        assert tag in [None, "feature", "model"]
        kg_feature_simulator = "The feature code will be sent to the simulator:\n" + prompt_dict["kg_feature_simulator"]

        kg_model_simulator = "The model code will be sent to the simulator:\n" + (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["kg_model_simulator"])
            .render(submission_specifications=self.submission_specifications)
        )
        if tag is None:
            return kg_feature_simulator + "\n" + kg_model_simulator
        elif tag == "feature":
            return kg_feature_simulator
        elif tag == "model":
            return kg_model_simulator

    @property
    def rich_style_description(self) -> str:
        return f"""
### Kaggle Agent: Automated Feature Engineering & Model Tuning Evolution

#### [Overview](#_summary)

In this scenario, our automated system proposes hypothesis, choose action, implements code, conducts validation, and utilizes feedback in a continuous, iterative process.

#### Kaggle Competition info

Current Competition: [{self.competition}](https://www.kaggle.com/competitions/{self.competition})

#### [Automated R&D](#_rdloops)

- **[R (Research)](#_research)**
- Iteration of ideas and hypotheses.
- Continuous learning and knowledge construction.

- **[D (Development)](#_development)**
- Evolving code generation, model refinement, and features generation.
- Automated implementation and testing of models/features.

#### [Objective](#_summary)

To automatically optimize performance metrics within the validation set or Kaggle Leaderboard, ultimately discovering the most efficient features and models through autonomous research and development.
"""

    def get_scenario_all_desc(
        self, task: Task | None = None, filtered_tag: str | None = None, simple_background: bool | None = None
    ) -> str:
        def common_description() -> str:
            return f"""\n------Background of the scenario------
{self.background}

------The source dataset you can use to generate the features------
{self.source_data}

------The expected output & submission format specifications------
{self.submission_specifications}
"""

        def interface(tag: str | None) -> str:
            return f"""
------The interface you should follow to write the runnable code------
{self.interface(tag)}
"""

        def output(tag: str | None) -> str:
            return f"""
------The output of your code should be in the format------
{self.output_format(tag)}
"""

        def simulator(tag: str | None) -> str:
            return f"""
------The simulator user can use to test your solution------
{self.simulator(tag)}
"""

        if filtered_tag is None:
            return common_description() + interface(None) + output(None) + simulator(None)
        elif filtered_tag == "hypothesis_and_experiment" or filtered_tag == "feedback":
            return common_description() + simulator(None)
        elif filtered_tag == "feature":
            return common_description() + interface("feature") + output("feature") + simulator("feature")
        else:
            return common_description() + interface(filtered_tag) + output("model") + simulator("model")
