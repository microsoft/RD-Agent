import json
import runpy
from pathlib import Path
from typing import Dict

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.debug.data import create_debug_data
from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2
from rdagent.scenarios.kaggle.kaggle_crawler import (
    crawl_descriptions,
    download_data,
    get_metric_direction,
)
from rdagent.utils.agent.tpl import T


class DataScienceScen(Scenario):
    """Data Science Scenario"""

    def __init__(self, competition: str) -> None:

        # 1) prepare data
        if not Path(f"{DS_RD_SETTING.local_data_path}/{competition}").exists():
            logger.error(f"Please prepare data for competition {competition} first.")
            raise FileNotFoundError(f"Cannot find {competition} in {DS_RD_SETTING.local_data_path}")

        local_path = DS_RD_SETTING.local_data_path
        if not DS_RD_SETTING.sample_data_by_LLM:
            self.debug_path = f"{local_path}/sample/{competition}"
            if not Path(self.debug_path).exists():
                sample_py_path = Path(local_path) / competition / "sample.py"
                if sample_py_path.exists():
                    runpy.run_path(
                        str(sample_py_path),
                        init_globals={
                            "dataset_path": str(local_path),
                            "output_path": str(self.debug_path),
                        },
                    )
                else:
                    create_debug_data(competition, dataset_path=local_path)
        else:
            self.debug_path = f"{local_path}/{competition}"

        # 2) collect information of competition.
        self.metric_name: str | None = (
            None  # It is None when initialization. After analysing, we'll assign the metric name
        )

        self.competition = competition
        self.raw_description = self._get_description()
        self.processed_data_folder_description = self._get_data_folder_description()
        self._analysis_competition_description()
        self.metric_direction: bool = (
            self._get_direction()
        )  # True indicates higher is better, False indicates lower is better

    def reanalyze_competition_description(self):
        self.raw_description = self._get_description()
        self.processed_data_folder_description = self._get_data_folder_description()
        self._analysis_competition_description()
        self.metric_direction: bool = self._get_direction()

    def _get_description(self):
        if (fp := Path(f"{DS_RD_SETTING.local_data_path}/{self.competition}/description.md")).exists():
            logger.info(f"{self.competition}/Found description.md, loading from local file.")
            return fp.read_text()
        elif (fp := Path(f"{DS_RD_SETTING.local_data_path}/{self.competition}.json")).exists():
            logger.info(f"Found {self.competition}.json, loading from local file.")
            with fp.open("r") as f:
                return json.load(f)
        else:
            logger.error(
                f"Cannot find '{self.competition}.json' in {DS_RD_SETTING.local_data_path} or 'description.md' file, please check the file."
            )

    def _get_direction(self):
        return self.metric_direction_guess if hasattr(self, "metric_direction_guess") else True

    def _analysis_competition_description(self):
        sys_prompt = T(".prompts:competition_description_template.system").r()
        user_prompt = T(".prompts:competition_description_template.user").r(
            competition_raw_description=self.raw_description,
            competition_processed_data_folder_description=self.processed_data_folder_description,
        )

        response_analysis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
            json_target_type=Dict[str, str | int | bool],
        )

        response_json_analysis = json.loads(response_analysis)
        self.task_type = response_json_analysis.get("Task Type", "No type provided")
        self.data_type = response_json_analysis.get("Data Type", "No data type provided")
        self.brief_description = response_json_analysis.get("Brief Description", "No brief description provided")
        self.dataset_description = response_json_analysis.get("Dataset Description", "No dataset description provided")
        self.submission_specifications = response_json_analysis.get(
            "Submission Specifications", "No submission requirements provided"
        )
        self.model_output_channel = response_json_analysis.get("Submission channel number to each sample", 1)
        self.metric_description = response_json_analysis.get(
            "Metric Evaluation Description", "No target description provided"
        )
        self.metric_name = response_json_analysis.get("Metric Name", "custom_metric")
        self.metric_direction_guess = response_json_analysis.get("Metric Direction", True)

    @property
    def background(self) -> str:
        background_template = T(".prompts:competition_background")
        background_prompt = background_template.r(
            task_type=self.task_type,
            data_type=self.data_type,
            brief_description=self.brief_description,
            dataset_description=self.dataset_description,
            model_output_channel=self.model_output_channel,
            metric_description=self.metric_description,
        )
        return background_prompt

    @property
    def rich_style_description(self) -> str:
        return T(".prompts:rich_style_description").r(
            name="Data Science",
            competition=self.competition,
        )

    def get_competition_full_desc(self) -> str:
        return T(".prompts:scenario_description").r(
            background=self.background,
            submission_specifications=self.submission_specifications,
            evaluation=self.metric_description,
            metric_name=self.metric_name,
            metric_direction=self.metric_direction,
            raw_description=self.raw_description,
            use_raw_description=DS_RD_SETTING.use_raw_description,
            time_limit=None,
            eda_output=None,
        )

    def get_scenario_all_desc(self, eda_output=None) -> str:
        """
        eda_output depends on dynamic .md files from current workspace, not fixed.
        """
        return T(".prompts:scenario_description").r(
            background=self.background,
            submission_specifications=self.submission_specifications,
            evaluation=self.metric_description,
            metric_name=self.metric_name,
            metric_direction=self.metric_direction,
            raw_description=self.raw_description,
            use_raw_description=DS_RD_SETTING.use_raw_description,
            time_limit=f"{DS_RD_SETTING.full_timeout / 60 / 60 : .2f} hours",
            eda_output=eda_output,
        )

    def get_runtime_environment(self) -> str:
        # TODO:  add it into base class.  Environment should(i.e. `DSDockerConf`) should be part of the scenario class.
        env = get_ds_env()
        implementation = FBWorkspace()
        fname = "runtime_info.py"
        implementation.inject_files(
            **{fname: (Path(__file__).absolute().resolve().parent / "runtime_info.py").read_text()}
        )
        stdout = implementation.execute(env=env, entry=f"python {fname}")
        return stdout

    def _get_data_folder_description(self) -> str:
        return describe_data_folder_v2(
            Path(DS_RD_SETTING.local_data_path) / self.competition, show_nan_columns=DS_RD_SETTING.show_nan_columns
        )


class KaggleScen(DataScienceScen):
    """Kaggle Scenario
    It is based on kaggle now.
        - But it is not use the same interface with previous kaggle version.
        - Ideally, we should reuse previous kaggle scenario.
          But we found that too much scenario unrelated code in kaggle scenario and hard to reuse.
          So we start from a simple one....
    """

    def __init__(self, competition: str) -> None:
        download_data(competition=competition, settings=DS_RD_SETTING, enable_create_debug_data=False)
        super().__init__(competition)

    def _get_description(self):
        return crawl_descriptions(self.competition, DS_RD_SETTING.local_data_path)

    def _get_direction(self):
        return get_metric_direction(self.competition)

    @property
    def rich_style_description(self) -> str:
        return T(".prompts:rich_style_description").r(
            name="Kaggle",
            competition=f"[{self.competition}](https://www.kaggle.com/competitions/{self.competition})",
        )


if __name__ == "__main__":
    print(describe_data_folder(Path("/data/userdata/share/mle_kaggle") / "stanford-covid-vaccine"))

    print(describe_data_folder_v2(Path("/data/userdata/share/mle_kaggle") / "stanford-covid-vaccine"))
