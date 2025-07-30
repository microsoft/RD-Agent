from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.scenario import Scenario
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.scen import DataScienceScen
from rdagent.scenarios.data_science.scen.utils import describe_data_folder_v2
from rdagent.utils.agent.tpl import T


class DSFinetuneScen(DataScienceScen):
    """DSFinetuneScen Scenario"""

    def _get_data_folder_description(self) -> str:
        folder_desc = describe_data_folder_v2(
            Path(DS_RD_SETTING.local_data_path) / self.competition,
            show_nan_columns=DS_RD_SETTING.show_nan_columns,
            max_length=20000,  # more context for model script
        )
        return folder_desc
