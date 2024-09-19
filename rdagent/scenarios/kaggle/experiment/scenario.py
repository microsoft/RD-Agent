import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.factor_coder.config import FACTOR_IMPLEMENT_SETTINGS
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import KGFactorExperiment
from rdagent.scenarios.kaggle.kaggle_crawler import crawl_descriptions
from rdagent.scenarios.kaggle.knowledge_management.vector_base import (
    KaggleExperienceBase,
)

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class KGScenario(Scenario):
    def __init__(self, competition: str) -> None:
        super().__init__()
        self.competition = competition
        self.competition_descriptions = crawl_descriptions(competition)
        self._source_data = self.source_data
        self._output_format = self.output_format
        self._interface = self.interface
        self._simulator = self.simulator

        self.competition_type = None
        self.competition_description = None
        self.target_description = None
        self.competition_features = None
        self._analysis_competition_description()

        self._background = self.background

        # all competitions are based on the same vector base
        self.vector_base = KaggleExperienceBase()
        if KAGGLE_IMPLEMENT_SETTING.rag_path and Path(KAGGLE_IMPLEMENT_SETTING.rag_path).exists():
            self.vector_base.load(KAGGLE_IMPLEMENT_SETTING.rag_path)

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
                raw_data_information=self._source_data,
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
        self.competition_features = self.source_data

    @property
    def background(self) -> str:
        background_template = prompt_dict["kg_background"]

        train_script = (Path(__file__).parent / "meta_tpl" / "train.py").read_text()

        background_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(background_template)
            .render(
                train_script=train_script,
                competition_type=self.competition_type,
                competition_description=self.competition_description,
                target_description=self.target_description,
                competition_features=self.competition_features,
            )
        )
        return background_prompt

    @property
    def source_data(self) -> str:
        data_folder = Path(FACTOR_IMPLEMENT_SETTINGS.data_folder) / self.competition

        if (data_folder / "valid.pkl").exists():
            X_valid = pd.read_pickle(data_folder / "valid.pkl")
            return X_valid.head()

        preprocess_experiment = KGFactorExperiment([])
        (
            X_train,
            X_valid,
            y_train,
            y_valid,
            X_test,
            passenger_ids,
        ) = preprocess_experiment.experiment_workspace.generate_preprocess_data()

        data_folder.mkdir(exist_ok=True, parents=True)
        X_valid.to_pickle(data_folder / "valid.pkl")
        return X_valid.head()

    @property
    def output_format(self) -> str:
        return prompt_dict["kg_model_output_format"]

    @property
    def interface(self) -> str:
        return f"""The feature code should follow the interface:
{prompt_dict['kg_feature_interface']}
The model code should follow the interface:
{prompt_dict['kg_model_interface']}
"""

    @property
    def simulator(self) -> str:
        return f"""The feature code should follow the simulator:
{prompt_dict['kg_feature_simulator']}
The model code should follow the simulator:
{prompt_dict['kg_model_simulator']}
"""

    @property
    def rich_style_description(self) -> str:
        return """
kaggle scen """

    def get_scenario_all_desc(self) -> str:
        return f"""Background of the scenario:
{self._background}
The source dataset you can use to generate the features:
{self._source_data}
The interface you should follow to write the runnable code:
{self._interface}
The output of your code should be in the format:
{self._output_format}
The simulator user can use to test your model:
{self._simulator}
"""
