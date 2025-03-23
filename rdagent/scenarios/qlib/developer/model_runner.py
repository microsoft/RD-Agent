from rdagent.components.runner import CachedRunner
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
import pandas as pd

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

    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: QlibModelExperiment) -> QlibModelExperiment:
        """
        # TODO: is this necessary?
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])
        """
        if exp.based_experiments:
            SOTA_factor = None
            if len(exp.based_experiments) > 1:
                SOTA_factor = self.process_factor_data(exp.based_experiments)
            
        combined_factors = SOTA_factor
        combined_factors = combined_factors.sort_index()
        combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
        new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
        combined_factors.columns = new_columns
        # TODO: calculate the factor numbers
        num_features = len(combined_factors.columns)

        target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

        # Save the combined factors to the workspace
        combined_factors.to_parquet(target_path, engine="pyarrow")

        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        if exp.sub_tasks[0].model_type == "TimeSeries":
            env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})
        elif exp.sub_tasks[0].model_type == "Tabular":
            env_to_use.update({"dataset_cls": "DatasetH"})

        result = exp.experiment_workspace.execute(qlib_config_name="conf_combined_with_model.yaml", run_env=env_to_use)
        # result = exp.experiment_workspace.execute(qlib_config_name="conf.yaml", run_env=env_to_use)
        result = None
        exp.result = result

        return exp
