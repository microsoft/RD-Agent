from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2
from rdagent.utils.agent.tpl import T


class LLMFinetuneScen(DataScienceScen):
    """LLMFinetuneScen Scenario"""

    def __init__(self, competition: str) -> None:
        self._download_data(competition=competition)
        super().__init__(competition)
        self._analysis_competition_description()

    def _get_data_folder_description(self) -> str:
        folder_desc = describe_data_folder_v2(
            Path(DS_RD_SETTING.local_data_path) / self.competition, show_nan_columns=DS_RD_SETTING.show_nan_columns
        )
        return folder_desc

    def _download_data(self, competition: str):
        """
        Download dateset from Hugging Face Hub

        Parameters
        ----------
        - competition (str): Dateset ID, like "shibing624/alpaca-zh".
        """
        save_path = f"{DS_RD_SETTING.local_data_path}/{competition}"
        if Path(save_path).exists():
            logger.info(f"{save_path} already exists.")
        else:
            logger.info(f"Downloading {competition} to {save_path}")
            try:
                from huggingface_hub import snapshot_download

                snapshot_download(
                    repo_id=competition,
                    repo_type="dataset",
                    local_dir=save_path,
                    local_dir_use_symlinks=False,
                )
            except ImportError:
                raise ImportError(
                    "Please install huggingface_hub first. "
                    'You can install it with `pip install -U "huggingface_hub[cli]"`.'
                )
            except Exception as e:
                logger.error(f"Error when downloading {competition}: {e}")
                raise e

    def _get_description(self):
        if (fp := Path(f"{DS_RD_SETTING.local_data_path}/{self.competition}/README.md")).exists():
            logger.info(f"{self.competition}/Found README.md, loading from local file.")
            return fp.read_text()

    def _get_direction(self):
        return True

    @property
    def rich_style_description(self) -> str:
        raise NotImplementedError

    @property
    def background(self) -> str:
        background_template = T(".prompts:competition_background")
        background_prompt = background_template.r(
            raw_description=self.raw_description,
        )
        return background_prompt

    def get_competition_full_desc(self) -> str:
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
        )

    def get_scenario_all_desc(self, eda_output=None) -> str:
        """
        eda_output depends on dynamic .md files from current workspace, not fixed.
        """
        return T(".prompts:scenario_description").r(
            raw_description=self.raw_description,
        )
