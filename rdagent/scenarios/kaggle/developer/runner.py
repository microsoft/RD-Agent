import json
import pickle
import shutil
from pathlib import Path

import pandas as pd

from rdagent.components.runner import CachedRunner
from rdagent.core.exception import CoderError, FactorEmptyError, ModelEmptyError
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.core.prompts import Prompts
from rdagent.core.utils import cache_with_pickle
from rdagent.oai.llm_utils import md5_hash
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import (
    KGFactorExperiment,
    KGModelExperiment,
)

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


class KGCachedRunner(CachedRunner[ASpecificExp]):
    def get_cache_key(self, exp: ASpecificExp) -> str:
        codes = []
        for f in sorted((exp.experiment_workspace.workspace_path / "feature").glob("*.py"), key=lambda x: x.name):
            codes.append(f.read_text())
        for f in sorted((exp.experiment_workspace.workspace_path / "model").glob("*.py"), key=lambda x: x.name):
            codes.append(f.read_text())
        codes = "\n".join(codes)
        cached_key_from_exp = CachedRunner.get_cache_key(self, exp)
        return md5_hash(codes + cached_key_from_exp)

    def assign_cached_result(self, exp: Experiment, cached_res: Experiment) -> Experiment:
        exp = CachedRunner.assign_cached_result(self, exp, cached_res)
        if cached_res.experiment_workspace.workspace_path.exists():
            for csv_file in cached_res.experiment_workspace.workspace_path.glob("*.csv"):
                shutil.copy(csv_file, exp.experiment_workspace.workspace_path)
            for py_file in (cached_res.experiment_workspace.workspace_path / "feature").glob("*.py"):
                shutil.copy(py_file, exp.experiment_workspace.workspace_path / "feature")
            for py_file in (cached_res.experiment_workspace.workspace_path / "model").glob("*.py"):
                shutil.copy(py_file, exp.experiment_workspace.workspace_path / "model")
        exp.experiment_workspace.data_description = cached_res.experiment_workspace.data_description
        return exp

    @cache_with_pickle(get_cache_key, CachedRunner.assign_cached_result)
    def init_develop(self, exp: KGFactorExperiment | KGModelExperiment) -> KGFactorExperiment | KGModelExperiment:
        """
        For the initial development, the experiment serves as a benchmark for feature engineering.
        """

        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        exp.result = result

        sub_result_score_path = Path(exp.experiment_workspace.workspace_path) / "sub_submission_score.csv"
        if sub_result_score_path.exists():
            sub_submission_df = pd.read_csv(sub_result_score_path)
            exp.sub_results = sub_submission_df.set_index("Model")["score"].to_dict()

        return exp


class KGModelRunner(KGCachedRunner[KGModelExperiment]):
    @cache_with_pickle(KGCachedRunner.get_cache_key, KGCachedRunner.assign_cached_result)
    def develop(self, exp: KGModelExperiment) -> KGModelExperiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.init_develop(exp.based_experiments[-1])

        sub_ws = exp.sub_workspace_list[0]
        if sub_ws is not None:
            # TODO: There's a possibility of generating a hybrid model (lightgbm + xgboost), which results in having two items in the model_type list.
            model_type = sub_ws.target_task.model_type

            if sub_ws.code_dict == {}:
                raise ModelEmptyError("No model is implemented.")
            else:
                model_file_name = f"model/model_{model_type.lower()}.py"
                exp.experiment_workspace.inject_code(**{model_file_name: sub_ws.code_dict["model.py"]})
        else:
            raise ModelEmptyError("No model is implemented.")
        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        if result is None:
            raise CoderError("No result is returned from the experiment workspace")

        exp.result = result
        sub_result_score_path = Path(exp.experiment_workspace.workspace_path) / "sub_submission_score.csv"
        if sub_result_score_path.exists():
            sub_submission_df = pd.read_csv(sub_result_score_path)
            exp.sub_results = sub_submission_df.set_index("Model")["score"].to_dict()

        return exp


class KGFactorRunner(KGCachedRunner[KGFactorExperiment]):
    @cache_with_pickle(KGCachedRunner.get_cache_key, KGCachedRunner.assign_cached_result)
    def develop(self, exp: KGFactorExperiment) -> KGFactorExperiment:
        current_feature_file_count = len(list(exp.experiment_workspace.workspace_path.glob("feature/feature*.py")))
        implemented_factor_count = 0
        for sub_ws in exp.sub_workspace_list:
            if sub_ws.code_dict == {}:
                continue
            execued_df = sub_ws.execute()[1]
            if execued_df is None:
                continue
            implemented_factor_count += 1
            target_feature_file_name = f"feature/feature_{current_feature_file_count:05d}.py"
            exp.experiment_workspace.inject_code(**{target_feature_file_name: sub_ws.code_dict["factor.py"]})
            feature_shape = execued_df.shape[-1]
            exp.experiment_workspace.data_description.append((sub_ws.target_task.get_task_information(), feature_shape))
            current_feature_file_count += 1
        if implemented_factor_count == 0:
            raise FactorEmptyError("No factor is implemented")

        # initial template result
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.init_develop(exp.based_experiments[-1])

        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        if result is None:
            raise CoderError("No result is returned from the experiment workspace")

        exp.result = result
        sub_result_score_path = Path(exp.experiment_workspace.workspace_path) / "sub_submission_score.csv"
        if sub_result_score_path.exists():
            sub_submission_df = pd.read_csv(sub_result_score_path)
            exp.sub_results = sub_submission_df.set_index("Model")["score"].to_dict()

        return exp
