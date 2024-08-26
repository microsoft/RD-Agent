import json
from pathlib import Path

import pandas as pd
from jinja2 import Environment, StrictUndefined

from rdagent.components.coder.model_coder.model import (
    ModelExperiment,
    ModelFBWorkspace,
    ModelTask,
)
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.workspace import KGFBWorkspace
from rdagent.scenarios.kaggle.kaggle_crawler import crawl_descriptions

prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class KGModelExperiment(ModelExperiment[ModelTask, KGFBWorkspace, ModelFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.experiment_workspace = KGFBWorkspace(template_folder_path=Path(__file__).parent / "model_template")


class KGModelScenario(Scenario):
    def __init__(self, competition: str) -> None:
        super().__init__()
        self.competition = competition
        self.competition_descriptions = crawl_descriptions(competition)
        self.competition_type = None
        self.competition_description = None
        self.target_description = None
        self.competition_features = None
        self._analysis_competition_description()

    def _analysis_competition_description(self):
        # TODO: use gpt to analyze the competition description

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

    @property
    def background(self) -> str:
        background_template = prompt_dict["kg_model_background"]

        background_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(background_template)
            .render(
                competition_type=self.competition_type,
                competition_description=self.competition_description,
                target_description=self.target_description,
                competition_features=self.competition_features,
            )
        )
        return background_prompt

    @property
    def source_data(self) -> str:
        kaggle_conf = KGDockerConf()
        data_path = Path(f"{kaggle_conf.share_data_path}/{self.competition}")

        csv_files = list(data_path.glob("*.csv"))

        if not csv_files:
            return "No CSV files found in the specified path."

        dataset = pd.concat([pd.read_csv(file) for file in csv_files], ignore_index=True)

        simple_eda = dataset.info(buf=None)  # Capture the info output
        data_shape = dataset.shape
        data_head = dataset.head()

        eda = (
            f"Basic Info about the data:\n{simple_eda}\n"
            f"Shape of the dataset: {data_shape}\n"
            f"Sample Data:\n{data_head}\n"
        )

        data_description = self.competition_descriptions.get("Data Description", "No description provided")
        eda += f"\nData Description:\n{data_description}"

        return eda

    @property
    def output_format(self) -> str:
        return prompt_dict["kg_model_output_format"]

    @property
    def interface(self) -> str:
        return prompt_dict["kg_model_interface"]

    @property
    def simulator(self) -> str:
        return prompt_dict["kg_model_simulator"]

    @property
    def rich_style_description(self) -> str:
        return """
kaggle scen """

    def get_scenario_all_desc(self) -> str:
        return f"""Background of the scenario:
{self.background}
The interface you should follow to write the runnable code:
{self.interface}
The output of your code should be in the format:
{self.output_format}
The simulator user can use to test your model:
{self.simulator}
"""
