import subprocess
import zipfile
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.artifact_transport import ARTIFACT_DUMP_CODE
from rdagent.utils.env import KGDockerEnv

KG_FEATURE_PREPROCESS_SCRIPT = ARTIFACT_DUMP_CODE + """

from fea_share_preprocess import preprocess_script

X_train, X_valid, y_train, y_valid, X_test, *others = preprocess_script()

dump_safe_artifacts([X_train, X_valid, y_train, y_valid, X_test, others])
"""


class KGFBWorkspace(FBWorkspace):
    def __init__(self, template_folder_path: Path, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.inject_code_from_folder(template_folder_path)
        self.data_description: List[Tuple[str, int]] = []

    @property
    def model_description(self) -> dict[str, str]:
        model_description = {}
        for k, v in self.file_dict.items():
            if k.startswith("model/"):
                model_description[k] = v
        return model_description

    def generate_preprocess_data(
        self,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.DataFrame, Any]:
        kgde = KGDockerEnv(KAGGLE_IMPLEMENT_SETTING.competition)
        kgde.prepare()

        execute_log, results = kgde.dump_python_code_run_and_get_results(
            code=KG_FEATURE_PREPROCESS_SCRIPT,
            local_path=str(self.workspace_path),
            dump_file_names=[
                "rdagent_artifacts/manifest.json",
            ],
            running_extra_volume=(
                {KAGGLE_IMPLEMENT_SETTING.local_data_path + "/" + KAGGLE_IMPLEMENT_SETTING.competition: "/kaggle/input"}
                if KAGGLE_IMPLEMENT_SETTING.competition
                else None
            ),
        )
        if len(results) == 0:
            logger.error("Feature preprocess failed.")
            raise Exception("Feature preprocess failed.")
        else:
            X_train, X_valid, y_train, y_valid, X_test, others = results
            return X_train, X_valid, y_train, y_valid, X_test, *others

    def execute(self, run_env: dict = {}, *args, **kwargs) -> str:
        logger.info(f"Running the experiment in {self.workspace_path}")

        kgde = KGDockerEnv(KAGGLE_IMPLEMENT_SETTING.competition)
        kgde.prepare()

        running_extra_volume = {}
        if KAGGLE_IMPLEMENT_SETTING.competition:
            running_extra_volume = {
                KAGGLE_IMPLEMENT_SETTING.local_data_path + "/" + KAGGLE_IMPLEMENT_SETTING.competition: "/kaggle/input"
            }
        else:
            running_extra_volume = {}

        execute_log = kgde.check_output(
            local_path=str(self.workspace_path),
            env=run_env,
            running_extra_volume=running_extra_volume,
        )

        csv_path = self.workspace_path / "submission_score.csv"

        if not csv_path.exists():
            logger.error(f"File {csv_path} does not exist.")
            return None
        return pd.read_csv(csv_path, index_col=0).iloc[:, 0]
