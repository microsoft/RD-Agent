import pickle
import shutil
from pathlib import Path

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.factor_coder.config import FACTOR_IMPLEMENT_SETTINGS
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.exception import FactorEmptyError, ModelEmptyError
from rdagent.core.experiment import ASpecificExp
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import (
    KGFactorExperiment,
    KGModelExperiment,
)


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
        if exp.sub_workspace_list[0].target_task.model_type == "XGBoost":
            if exp.sub_workspace_list[0].code_dict == {}:
                raise ModelEmptyError("No model is implemented")
            exp.experiment_workspace.inject_code(**{"model_xgb.py": exp.sub_workspace_list[0].code_dict["model.py"]})
        elif exp.sub_workspace_list[0].target_task.model_type == "RandomForest":
            if exp.sub_workspace_list[0].code_dict == {}:
                raise ModelEmptyError("No model is implemented")
            exp.experiment_workspace.inject_code(**{"model_rf.py": exp.sub_workspace_list[0].code_dict["model.py"]})
        elif exp.sub_workspace_list[0].target_task.model_type == "LightGBM":
            if exp.sub_workspace_list[0].code_dict == {}:
                raise ModelEmptyError("No model is implemented")
            exp.experiment_workspace.inject_code(**{"model_lgb.py": exp.sub_workspace_list[0].code_dict["model.py"]})
        elif exp.sub_workspace_list[0].target_task.model_type == "NN":
            if exp.sub_workspace_list[0].code_dict == {}:
                raise ModelEmptyError("No model is implemented")
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
    def init_develop(self, exp: KGFactorExperiment) -> KGFactorExperiment:
        """
        For the initial development, the experiment serves as a benchmark for feature engineering.
        """
        self.build_from_SOTA(exp)
        if RUNNER_SETTINGS.cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp

        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        exp.result = result
        sub_task = FactorTask(
            factor_name="original features", factor_description="here is the original features", factor_formulation=""
        )

        org_data_path = (
            Path(FACTOR_IMPLEMENT_SETTINGS.data_folder) / KAGGLE_IMPLEMENT_SETTING.competition / "X_valid.pkl"
        )
        with open(org_data_path, "rb") as f:
            org_data = pickle.load(f)
        feature_shape = org_data.shape[-1]
        exp.experiment_workspace.data_description.append((sub_task.get_task_information(), feature_shape))

        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp

    def develop(self, exp: KGFactorExperiment) -> KGFactorExperiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.init_develop(exp.based_experiments[-1])
        self.build_from_SOTA(exp)
        current_feature_file_count = len(list(exp.experiment_workspace.workspace_path.glob("feature/feature*.py")))
        implemented_factor_count = 0
        for sub_ws in exp.sub_workspace_list:
            if sub_ws.code_dict == {}:
                continue
            implemented_factor_count += 1
            target_feature_file_name = f"feature/feature_{current_feature_file_count:05d}.py"
            exp.experiment_workspace.inject_code(**{target_feature_file_name: sub_ws.code_dict["factor.py"]})
            feature_shape = sub_ws.execute()[1].shape[-1]
            exp.experiment_workspace.data_description.append((sub_ws.target_task.get_task_information(), feature_shape))
            current_feature_file_count += 1
        if implemented_factor_count == 0:
            raise FactorEmptyError("No factor is implemented")

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
