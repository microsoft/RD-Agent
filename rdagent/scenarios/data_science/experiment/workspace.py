from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import DockerEnv, DSDockerConf


class DSFBWorkspace(FBWorkspace):
    
    # TODO: use the cache_with_pickle decorator.
    def execute(self, run_env: dict = {}, *args, **kwargs) -> pd.DataFrame:
        """
        Executes the experiment(a competition) within the specified workspace.

        Args:
            run_env (dict): The runtime environment variables.

        Returns:
            pd.DataFrame: Scores of each Model and ensemble Model.
            Example:
            | Model                 | <Metric like ACC/AUROC/MAE...> |
            |-----------------------|--------------------------------|
            | model1                | 0.9                            |
            | model2                | 0.8                            |
            | <ensemble model name> | 0.95                           |
        """
        logger.info(f"Running the experiment in {self.workspace_path}")

        de = DockerEnv(DSDockerConf())
        de.prepare()

        running_extra_volume = {}
        if DS_RD_SETTING.competition:
            running_extra_volume = {
                DS_RD_SETTING.local_data_path + "/" + DS_RD_SETTING.competition: "/kaggle/input"
            }
        else:
            running_extra_volume = {}

        execute_log = de.run(
            local_path=str(self.workspace_path),
            env=run_env,
            running_extra_volume=running_extra_volume,
        )

        csv_path = self.workspace_path / "scores.csv"

        if not csv_path.exists():
            logger.error(f"File {csv_path} does not exist.")
            return None
        return pd.read_csv(csv_path, index_col=0)
