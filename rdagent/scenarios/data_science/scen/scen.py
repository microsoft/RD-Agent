import json

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.kaggle_crawler import (
    crawl_descriptions,
    leaderboard_scores,
)
from rdagent.utils.agent.tpl import T


class DataScienceScen(Scenario):
    """Data Science Scenario
    It is based on kaggle now.
        - But it is not use the same interface with previous kaggle version.
        - Ideally, we should reuse previous kaggle scenario.
          But we found that too much scenario unrelated code in kaggle scenario and hard to reuse.
          So we start from a simple one....
    """

    def __init__(self, competition: str) -> None:
        self.competition = competition
        self.competition_raw_description = crawl_descriptions(competition, DS_RD_SETTING.local_data_path)

        leaderboard = leaderboard_scores(competition)
        self.competition_metric_direction = float(leaderboard[0]) > float(leaderboard[-1])

        self._analysis_competition_description()

    def _analysis_competition_description(self):
        sys_prompt = T(".prompts:competition_description_template.system").r()
        user_prompt = T(".prompts:competition_description_template.user").r(
            competition_raw_description=self.competition_raw_description,
        )

        response_analysis = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        response_json_analysis = json.loads(response_analysis)
        self.competition_task_type = response_json_analysis.get("Competition Task Type", "No type provided")
        self.competition_data_type = response_json_analysis.get("Competition Data Type", "No data type provided")
        self.competition_brief_description = response_json_analysis.get("Competition Brief Description", "No brief description provided")
        self.competition_target_description = response_json_analysis.get("Competition Target Description", "No target description provided")
        self.submission_specifications = response_json_analysis.get(
            "Submission Specifications", "No submission requirements provided"
        )
        self.model_output_channel = response_json_analysis.get("Submission channel number to each sample", 1)

    def get_competition_full_desc(self) -> str:
        return f"""Competition Task Type: {self.competition_task_type}
    Competition Data Type: {self.competition_data_type}
    Competition Brief Description: {self.competition_brief_description}
    Competition Target Description: {self.competition_target_description}
    Submission Specifications: {self.submission_specifications}
    Model Output Channel: {self.model_output_channel}
    """

    @property
    def background(self) -> str:
        background_template = T(".prompts:competition_background")
        background_prompt = background_template.r(
            competition_task_type=self.competition_task_type,
            competition_data_type=self.competition_data_type,
            competition_brief_description=self.competition_brief_description,
            target_description=self.competition_target_description,
            submission_specifications=self.submission_specifications,
            evaluate_bool=self.competition_metric_direction,
        )
        return background_prompt

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

    def get_scenario_all_desc(self) -> str:
        return T(".prompts:scen_desc").r(scen=self)
