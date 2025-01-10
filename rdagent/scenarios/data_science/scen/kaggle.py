import json

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.kaggle.kaggle_crawler import (
    crawl_descriptions,
    leaderboard_scores,
)
from rdagent.utils.agent.tpl import T


class KaggleScen(DataScienceScen):
    """Kaggle Scenario
    It is based on kaggle now.
        - But it is not use the same interface with previous kaggle version.
        - Ideally, we should reuse previous kaggle scenario.
          But we found that too much scenario unrelated code in kaggle scenario and hard to reuse.
          So we start from a simple one....
    """

    def _get_description(self):
        return crawl_descriptions(self.competition, DS_RD_SETTING.local_data_path)

    def _get_direction(self):
        if DS_RD_SETTING.if_using_mle_data:
            return super()._get_direction()
        leaderboard = leaderboard_scores(self.competition)
        return "maximize" if float(leaderboard[0]) > float(leaderboard[-1]) else "minimize"

    @property
    def rich_style_description(self) -> str:
        return T(".prompts:rich_style_description").r(
            name="Kaggle",
            competition=f"[{self.competition}](https://www.kaggle.com/competitions/{self.competition})",
        )
