import subprocess
import zipfile
from pathlib import Path
from typing import Any, List, Tuple

import pandas as pd

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.core.experiment import FBWorkspace
from rdagent.log import rdagent_logger as logger
from rdagent.utils.env import KGDockerEnv

KG_FEATURE_PREPROCESS_SCRIPT = """import pickle

from fea_share_preprocess import preprocess_script

X_train, X_valid, y_train, y_valid, X_test, *others = preprocess_script()

pickle.dump(X_train, open("X_train.pkl", "wb"))
pickle.dump(X_valid, open("X_valid.pkl", "wb"))
pickle.dump(y_train, open("y_train.pkl", "wb"))
pickle.dump(y_valid, open("y_valid.pkl", "wb"))
pickle.dump(X_test, open("X_test.pkl", "wb"))
pickle.dump(others, open("others.pkl", "wb"))
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
                "X_train.pkl",
                "X_valid.pkl",
                "y_train.pkl",
                "y_valid.pkl",
                "X_test.pkl",
                "others.pkl",
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
