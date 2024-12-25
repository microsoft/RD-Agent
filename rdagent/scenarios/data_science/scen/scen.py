import json
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T


class DataScienceScen(Scenario):
    """Data Science Scenario"""

    def __init__(self, competition: str) -> None:
        self.competition = competition
        self.raw_description = self._get_description()
        self.metric_direction = self._get_direction()
        self._analysis_competition_description()

    def _get_description(self):
        if (fp := Path(f"{DS_RD_SETTING.local_data_path}/{self.competition}.json")).exists():
            logger.info(f"Found {self.competition}.json, loading from local file.")
            with fp.open("r") as f:
                return json.load(f)
        else:
            logger.error(
                f"Cannot find {self.competition}.json in {DS_RD_SETTING.local_data_path}, please check the file."
            )

    def _get_direction(self):
        return self.raw_description.get("metric_direction", "minimize")

    def _analysis_competition_description(self):
        sys_prompt = T(".prompts:competition_description_template.system").r()
        user_prompt = T(".prompts:competition_description_template.user").r(
            competition_raw_description=self.raw_description,
        )

        response_analysis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        response_json_analysis = json.loads(response_analysis)
        self.task_type = response_json_analysis.get("Task Type", "No type provided")
        self.data_type = response_json_analysis.get("Data Type", "No data type provided")
        self.brief_description = response_json_analysis.get("Brief Description", "No brief description provided")
        self.data_description = response_json_analysis.get("Data Description", "No data description provided")
        self.target_description = response_json_analysis.get("Target Description", "No target description provided")
        self.submission_specifications = response_json_analysis.get(
            "Submission Specifications", "No submission requirements provided"
        )
        self.model_output_channel = response_json_analysis.get("Submission channel number to each sample", 1)

    def get_competition_full_desc(self) -> str:
        return f"""Task Type: {self.task_type}
    Data Type: {self.data_type}
    Brief Description: {self.brief_description}
    Data Description: {self.data_description}
    Target Description: {self.target_description}
    Submission Specifications: {self.submission_specifications}
    Model Output Channel: {self.model_output_channel}
    """

    @property
    def background(self) -> str:
        background_template = T(".prompts:competition_background")
        background_prompt = background_template.r(
            task_type=self.task_type,
            data_type=self.data_type,
            brief_description=self.brief_description,
            data_description=self.data_description,
            target_description=self.target_description,
        )
        return background_prompt

    @property
    def rich_style_description(self) -> str:
        return T(".prompts:rich_style_description").r(
            name="Data Science",
            competition=self.competition,
        )

    def get_scenario_all_desc(self) -> str:
        return T(".prompts:scenario_description").r(
            background=self.background,
            submission_specifications=self.submission_specifications,
            metric_direction=self.metric_direction,
        )
