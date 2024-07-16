import shutil
import uuid
from pathlib import Path

import pandas as pd

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelFBWorkspace
from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.log import RDAgentLog
from rdagent.utils.env import QTDockerEnv


class QlibModelRunner(CachedRunner[ModelFBWorkspace]):
    """
    Docker run
    Everything in a folder
    - config.yaml
    - Pytorch `model.py`
    - results in `mlflow`

    https://github.com/microsoft/qlib/blob/main/qlib/contrib/model/pytorch_nn.py
    - pt_model_uri:  hard-code `model.py:Net` in the config
    - let LLM modify model.py
    """

    def develop(self, exp: ModelExperiment) -> ModelExperiment:

        if RUNNER_SETTINGS.runner_cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp
        TEMPLATE_PATH = Path(__file__).parent / "model_template"  # Can be updated

        # To set the experiment level workspace and prepare the workspaces use the first task as the target task
        exp.experiment_workspace = ModelFBWorkspace(target_task=exp.sub_tasks[0])
        exp.experiment_workspace.prepare()

        # to copy_template_to_workspace
        for file_path in TEMPLATE_PATH.iterdir():
            shutil.copyfile(file_path, exp.experiment_workspace.workspace_folder_path / file_path.name)

        # to replace & inject code
        exp.experiment_workspace.inject_code(**{"model.py": exp.sub_implementations[0].code_dict["model.py"]})

        env_to_use = {}

        if exp.sub_tasks[0].model_type == "TimeSeries":
            env_to_use = {"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20}
        elif exp.sub_tasks[0].model_type == "Tabular":
            env_to_use = {"dataset_cls": "DatasetH"}
        # to execute
        qtde = QTDockerEnv()

        # Preparing the Docker environment
        qtde.prepare()

        # Run the Docker container with the specified entry

        execute_log = qtde.run(
            local_path=exp.experiment_workspace.workspace_folder_path,
            entry="qrun conf.yaml",
            env={"PYTHONPATH": "./", **env_to_use},
        )

        # Run the experiment analysis code
        execute_log = qtde.run(
            local_path=exp.experiment_workspace.workspace_folder_path, entry="python read_exp_res.py"
        )

        csv_path = exp.experiment_workspace.workspace_folder_path / "qlib_res.csv"

        if not csv_path.exists():
            RDAgentLog().error(f"File {csv_path} does not exist.")
            return None

        result = pd.read_csv(csv_path, index_col=0).iloc[:, 0]

        exp.result = result
        if RUNNER_SETTINGS.runner_cache_result:
            self.dump_cache_result(exp, result)

        return exp
