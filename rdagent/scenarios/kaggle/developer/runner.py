import json
import pickle
import shutil
from pathlib import Path

from jinja2 import Environment, StrictUndefined

from rdagent.app.kaggle.conf import KAGGLE_IMPLEMENT_SETTING
from rdagent.components.coder.factor_coder.factor import FactorTask
from rdagent.components.coder.model_coder.model import ModelTask
from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.exception import CoderError, FactorEmptyError, ModelEmptyError
from rdagent.core.experiment import ASpecificExp
from rdagent.core.prompts import Prompts
from rdagent.oai.llm_utils import APIBackend, md5_hash
from rdagent.scenarios.kaggle.experiment.kaggle_experiment import (
    KGFactorExperiment,
    KGModelExperiment,
)

prompt_dict = Prompts(file_path=Path(__file__).parent.parent / "prompts.yaml")


class KGCachedRunner(CachedRunner[ASpecificExp]):
    def build_from_SOTA(self, exp: ASpecificExp) -> None:
        if len(exp.based_experiments) > 0:
            exp.experiment_workspace.inject_code(**exp.based_experiments[-1].experiment_workspace.code_dict)
            exp.experiment_workspace.data_description = exp.based_experiments[-1].experiment_workspace.data_description
            exp.experiment_workspace.model_description = exp.based_experiments[
                -1
            ].experiment_workspace.model_description.copy()

    def get_cache_key(self, exp: ASpecificExp) -> str:
        codes = []
        for f in sorted((exp.experiment_workspace.workspace_path / "feature").glob("*.py"), key=lambda x: x.name):
            codes.append(f.read_text())
        for f in sorted((exp.experiment_workspace.workspace_path / "model").glob("*.py"), key=lambda x: x.name):
            codes.append(f.read_text())
        codes = "\n".join(codes)
        return md5_hash(codes)

    def init_develop(self, exp: KGFactorExperiment | KGModelExperiment) -> KGFactorExperiment | KGModelExperiment:
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
            Path(KAGGLE_IMPLEMENT_SETTING.local_data_path) / KAGGLE_IMPLEMENT_SETTING.competition / "X_valid.pkl"
        )
        with open(org_data_path, "rb") as f:
            org_data = pickle.load(f)
        feature_shape = org_data.shape[-1]
        exp.experiment_workspace.data_description.append((sub_task.get_task_information(), feature_shape))

        sub_model_1_description = (
            self.extract_model_task_from_code(
                (exp.experiment_workspace.workspace_path / "model" / "model_randomforest.py").read_text()
            )
            + f"""code: { (exp.experiment_workspace.workspace_path / "model" / "model_randomforest.py").read_text()}"""
        )
        sub_model_2_description = (
            self.extract_model_task_from_code(
                (exp.experiment_workspace.workspace_path / "model" / "model_xgboost.py").read_text()
            )
            + f"""code: { (exp.experiment_workspace.workspace_path / "model" / "model_xgboost.py").read_text()}"""
        )

        exp.experiment_workspace.model_description["XGBoost"] = sub_model_1_description
        exp.experiment_workspace.model_description["RandomForest"] = sub_model_2_description

        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp


class KGModelRunner(KGCachedRunner[KGModelExperiment]):
    def develop(self, exp: KGModelExperiment) -> KGModelExperiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.init_develop(exp.based_experiments[-1])
        self.build_from_SOTA(exp)

        sub_ws = exp.sub_workspace_list[0]
        model_type = sub_ws.target_task.model_type

        if sub_ws.code_dict == {}:
            raise ModelEmptyError("No model is implemented.")
        else:
            model_file_name = f"model/model_{model_type.lower()}.py"
            exp.experiment_workspace.inject_code(**{model_file_name: sub_ws.code_dict["model.py"]})

            model_description = sub_ws.target_task.get_task_information()
            exp.experiment_workspace.model_description[model_type] = model_description

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
    def extract_model_task_from_code(self, code: str) -> str:
        sys_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["extract_model_task_from_code"]["system"])
            .render()
        )

        user_prompt = (
            Environment(undefined=StrictUndefined)
            .from_string(prompt_dict["extract_model_task_from_code"]["user"])
            .render(file_content=code)
        )

        model_task_description = APIBackend().build_messages_and_create_chat_completion(
            user_prompt=user_prompt,
            system_prompt=sys_prompt,
            json_mode=True,
        )

        try:
            response_json_analysis = json.loads(model_task_description)
            task_desc = f"""name: {response_json_analysis['name']}
        description: {response_json_analysis['description']}
        """
            task_desc += (
                f"formulation: {response_json_analysis['formulation']}\n"
                if response_json_analysis.get("formulation")
                else ""
            )
            task_desc += f"architecture: {response_json_analysis['architecture']}\n"
            task_desc += (
                f"variables: {json.dumps(response_json_analysis['variables'], indent=4)}\n"
                if response_json_analysis.get("variables")
                else ""
            )
            task_desc += f"hyperparameters: {json.dumps(response_json_analysis['hyperparameters'], indent=4)}\n"
            task_desc += f"model_type: {response_json_analysis['model_type']}\n"
        except json.JSONDecodeError:
            task_desc = "Failed to parse LLM's response as JSON"

        return task_desc

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

        if result is None:
            raise CoderError("No result is returned from the experiment workspace")

        exp.result = result

        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp
