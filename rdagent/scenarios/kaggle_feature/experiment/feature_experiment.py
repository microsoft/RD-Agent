import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined
import pandas as pd

from rdagent.components.coder.factor_coder.factor import (
    FactorExperiment,
    FactorFBWorkspace,
    FactorTask,
)
from rdagent.components.coder.feature_coder.config import FEATURE_IMPLEMENT_SETTINGS
from rdagent.core.prompts import Prompts
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle_feature.experiment.workspace import KGFFBWorkspace
from rdagent.scenarios.kaggle.kaggle_crawler import crawl_descriptions
from rdagent.utils.env import KGDockerConf


prompt_dict = Prompts(file_path=Path(__file__).parent / "prompts.yaml")


class KGFeatureExperiment(FactorExperiment[FactorTask, KGFFBWorkspace, KGFFBWorkspace]):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        print(Path(__file__).parent.parent.parent/"kaggle/experiment/meta_tpl")
        self.experiment_workspace = KGFFBWorkspace(template_folder_path=Path(__file__).parent.parent.parent/"kaggle/experiment/meta_tpl")


class KGFeatureScenario(Scenario):
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

        X_train, X_valid, y_train, y_valid, X_test, passenger_ids = self.preprocess_script()
        self.competition_features = X_train.columns.tolist()
        self.competition_features = X_train.head()

    @property
    def background(self) -> str:
        background_template = prompt_dict["kg_feature_background"]

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
    def source_data(self) -> pd.DataFrame:
        # #TODO: Implement source_data
        # kaggle_conf = KGDockerConf()
        # data_path = Path(f"{kaggle_conf.share_data_path}/{self.competition}")
        # file_path = data_path / "train.csv"
        # data = pd.read_csv(file_path)
        #TODO later we should improve this part
        data_folder = Path(FEATURE_IMPLEMENT_SETTINGS.data_folder)

        if (data_folder / "valid.csv").exists():
            X_valid = pd.read_csv(data_folder / "valid.csv")
            # X_valid = pd.read_csv(data_folder / "valid.csv")
            # X_test = pd.read_csv(data_folder / "test.csv")
            return X_valid.head()
        
        X_train, X_valid, y_train, y_valid, X_test, passenger_ids = self.preprocess_script()

        data_folder.mkdir(exist_ok=True, parents=True)
        X_train.to_csv(data_folder / "train.csv", index=False)
        X_valid.to_csv(data_folder / "valid.csv", index=False)
        X_test.to_csv(data_folder / "test.csv", index=False)
        return X_train.head()
        raise NotImplementedError("source_data is not implemented")

    @property
    def output_format(self) -> str:
        return prompt_dict["kg_feature_output_format"]

    @property
    def interface(self) -> str:
        return prompt_dict["kg_feature_interface"]

    @property
    def simulator(self) -> str:
        return prompt_dict["kg_feature_simulator"]

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
