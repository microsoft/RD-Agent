import shutil
import uuid
from pathlib import Path

import pandas as pd

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelFBWorkspace
from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import ModelEmptyError
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.kaggle.experiment.model_experiment import KGModelExperiment
from rdagent.utils.env import KGDockerEnv


class KGModelRunner(CachedRunner[KGModelExperiment]):
    def develop(self, exp: KGModelExperiment) -> KGModelExperiment:
        if RUNNER_SETTINGS.cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp

        if exp.sub_workspace_list[0].code_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_code(**{"model.py": exp.sub_workspace_list[0].code_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        exp.result = result
        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp
