import shutil
import uuid
from pathlib import Path

import pandas as pd

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelFBWorkspace
from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import ModelEmptyException
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.utils.env import QTDockerEnv


class QlibModelRunner(CachedRunner[QlibModelExperiment]):
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

    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        if RUNNER_SETTINGS.cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp

        if exp.sub_workspace_list[0].code_dict.get("model.py") is None:
            raise ModelEmptyException("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_code(**{"model.py": exp.sub_workspace_list[0].code_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        if exp.sub_tasks[0].model_type == "TimeSeries":
            env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})
        elif exp.sub_tasks[0].model_type == "Tabular":
            env_to_use.update({"dataset_cls": "DatasetH"})

        result = exp.experiment_workspace.execute(qlib_config_name="conf.yaml", run_env=env_to_use)

        exp.result = result
        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp
