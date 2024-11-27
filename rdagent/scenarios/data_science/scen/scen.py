from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.experiment import Task
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.kaggle.experiment.scenario import prompt_dict as kaggle_prompt_dict
from rdagent.scenarios.kaggle.kaggle_crawler import crawl_descriptions, leaderboard_scores
from rdagent.utils.agent.tpl import T
import json


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
        self.competition_descriptions = crawl_descriptions(competition, DS_RD_SETTING.local_data_path)
        self.leaderboard = leaderboard_scores(competition)
        self.evaluation_metric_direction = float(self.leaderboard[0]) > float(self.leaderboard[-1])
        self._analysis_competition_description()

    def _analysis_competition_description(self):
        sys_prompt = T("scenarios.kaggle.experiment.prompts:kg_description_template.system").r()
        user_prompt = T("scenarios.kaggle.experiment.prompts:kg_description_template.user").r(
                competition_descriptions=self.competition_descriptions,
                raw_data_information=self.source_data,
                evaluation_metric_direction=self.evaluation_metric_direction,
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
        self.submission_specifications = response_json_analysis.get("Submission Specifications",
                                                                    "No submission requirements provided")
        self.model_output_channel = response_json_analysis.get("Submission channel number to each sample", 1)
        self.evaluation_desc = response_json_analysis.get("Evaluation Description",
                                                          "No evaluation specification provided.")

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
        background_template = T("scenarios.kaggle.experiment.prompts:kg_background")
        background_prompt = background_template.r(
            competition_type=self.competition_type,
            competition_description=self.competition_description,
            target_description=self.target_description,
            competition_features=self.competition_features,
            submission_specifications=self.submission_specifications,
            evaluation_desc=self.evaluation_desc,
            evaluate_bool=self.evaluation_metric_direction,
        )
        return background_prompt

    @property
    def source_data(self) -> str:
        # TODO: remove me if not used
        # TODO: (bowen)
        # phase1:
        # - If we have not implement load data and dump cache
        # - describe the raw data
        # phase2: (cache detected)
        # - Describe the cached data (preprocessed data).
        return "!!!!!!!!! I'm the fake source data !!!!!!!!"
        raise NotImplementedError(f"We are not sure how it is called. We place a exception here")

    def output_format(self, tag=None) -> str:
        # TODO: remove me if not used
        raise NotImplementedError(f"We are not sure how it is called. We place a exception here")

    def simulator(self, tag=None) -> str:
        # TODO: remove me if not used
        raise NotImplementedError(f"We are not sure how it is called. We place a exception here")

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

    def get_scenario_all_desc(self, task: Task | None = None, filtered_tag: str | None = None) -> str:
        # TODO: remove me if not used
        raise NotImplementedError(f"We are not sure how it is called. We place a exception here")
        # if filtered_tag is None:
        #     return common_description() + interface(None) + output(None) + simulator(None)
        # NOTE: we suggest such implementation: `return T(".prompts:scen_desc").r()`
