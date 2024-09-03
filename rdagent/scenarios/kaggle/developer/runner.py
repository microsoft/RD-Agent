import shutil
import uuid
from pathlib import Path

import pandas as pd

from rdagent.components.coder.model_coder.model import ModelExperiment, ModelFBWorkspace
from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.developer import Developer
from rdagent.core.exception import ModelEmptyError
from rdagent.core.experiment import ASpecificExp
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.kaggle.experiment.model_experiment import (
    KGFactorExperiment,
    KGModelExperiment,
)
from rdagent.utils.env import KGDockerEnv

META_TPL_DIR = Path(__file__).parent.parent / "experiment" / "meta_tpl"


class KGCachedRunner(CachedRunner[ASpecificExp]):
    def build_from_SOTA(self, exp: ASpecificExp) -> None:
        if len(exp.based_experiments) > 0:
            exp.experiment_workspace.inject_code(**exp.based_experiments[-1].experiment_workspace.code_dict)
            exp.experiment_workspace.data_description = exp.based_experiments[-1].experiment_workspace.data_description
            exp.experiment_workspace.model_description = exp.based_experiments[
                -1
            ].experiment_workspace.model_description

    def get_cache_key(self, exp: ASpecificExp) -> str:
        codes = []
        for f in sorted((exp.experiment_workspace.workspace_path / "feature").glob("*.py"), key=lambda x: x.name):
            codes.append(f.read_text())
        for f in sorted((exp.experiment_workspace.workspace_path / "model").glob("*.py"), key=lambda x: x.name):
            codes.append(f.read_text())
        codes = "\n".join(codes)
        return md5_hash(codes)


class KGModelRunner(KGCachedRunner[KGModelExperiment]):
    def develop(self, exp: KGModelExperiment) -> KGModelExperiment:
        self.build_from_SOTA(exp)
        if exp.sub_workspace_list[0].model_type == "XGBoost":
            exp.experiment_workspace.inject_code(**{"model_xgb.py": exp.sub_workspace_list[0].code_dict["model.py"]})
        elif exp.sub_workspace_list[0].model_type == "RandomForest":
            exp.experiment_workspace.inject_code(**{"model_rf.py": exp.sub_workspace_list[0].code_dict["model.py"]})
        elif exp.sub_workspace_list[0].model_type == "LightGBM":
            exp.experiment_workspace.inject_code(**{"model_lgb.py": exp.sub_workspace_list[0].code_dict["model.py"]})
        elif exp.sub_workspace_list[0].model_type == "NN":
            exp.experiment_workspace.inject_code(**{"model_nn.py": exp.sub_workspace_list[0].code_dict["model.py"]})
        if RUNNER_SETTINGS.cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp

        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        exp.result = result
        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp


class KGFactorRunner(KGCachedRunner[KGFactorExperiment]):
    def develop(self, exp: KGFactorExperiment) -> KGFactorExperiment:
        self.build_from_SOTA(exp)
        current_feature_file_count = len(list(exp.experiment_workspace.workspace_path.glob("feature/feature*.py")))
        for sub_ws in exp.sub_workspace_list:
            if sub_ws.code_dict == {}:
                continue
            target_feature_file_name = f"feature/feature_{current_feature_file_count:05d}.py"
            exp.experiment_workspace.inject_code(**{target_feature_file_name: sub_ws.code_dict["factor.py"]})
            feature_shape = sub_ws.execute()[1].shape[-1]
            exp.experiment_workspace.data_description.append((sub_ws.target_task.get_task_information(), feature_shape))
            current_feature_file_count += 1

        if RUNNER_SETTINGS.cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp

        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        exp.result = result
        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp
