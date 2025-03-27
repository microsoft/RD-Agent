import pickle
from pathlib import Path
from typing import List

from rdagent.components.runner import CachedRunner
from rdagent.core.exception import ModelEmptyError, FactorEmptyError
from rdagent.core.utils import cache_with_pickle

from rdagent.components.coder.CoSTEER.evaluators import CoSTEERMultiFeedback
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.utils import cache_with_pickle, multiprocessing_wrapper

from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
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
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])
        
        exist_sota_factor_exp = False
        if exp.based_experiments:
            SOTA_factor = None
            if len(exp.based_experiments) >= 1:
                SOTA_factor = self.process_factor_data(exp.based_experiments)
        
            if SOTA_factor is not None and not SOTA_factor.empty:
                exist_sota_factor_exp = True
                combined_factors = SOTA_factor
                combined_factors = combined_factors.sort_index()
                combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
                new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
                combined_factors.columns = new_columns
                # TODO: calculate the factor numbers
                num_features = len(combined_factors.columns) + 158

                target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

                # Save the combined factors to the workspace
                combined_factors.to_parquet(target_path, engine="pyarrow")

        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        if exist_sota_factor_exp:
            if exp.sub_tasks[0].model_type == "TimeSeries":
                env_to_use.update({"dataset_cls": "TSDatasetH", "num_features": 20, "step_len": 20, "num_timesteps": 20})
            elif exp.sub_tasks[0].model_type == "Tabular":
                env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})
            result = exp.experiment_workspace.execute(qlib_config_name="conf_model.yaml", run_env=env_to_use)
        else:
            if exp.sub_tasks[0].model_type == "TimeSeries":
                env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})  #TODO: step_len == input.size(-1)
            elif exp.sub_tasks[0].model_type == "Tabular":
                env_to_use.update({"dataset_cls": "DatasetH"})
            result = exp.experiment_workspace.execute(qlib_config_name="conf_model.yaml", run_env=env_to_use)
        exp.result = result

        return exp

    def process_factor_data(self, exp_or_list: List[QlibFactorExperiment] | QlibFactorExperiment) -> pd.DataFrame:
        """
        Process and combine factor data from experiment implementations.

        Args:
            exp (ASpecificExp): The experiment containing factor data.

        Returns:
            pd.DataFrame: Combined factor data without NaN values.
        """
        if isinstance(exp_or_list, QlibFactorExperiment):
            exp_or_list = [exp_or_list]
        factor_dfs = []

        # Collect all exp's dataframes
        for exp in exp_or_list:
            if isinstance(exp, QlibFactorExperiment):
                if len(exp.sub_tasks) > 0:
                    # if it has no sub_tasks, the experiment is results from template project.
                    # otherwise, it is developed with designed task. So it should have feedback.
                    assert isinstance(exp.prop_dev_feedback, CoSTEERMultiFeedback)
                    # Iterate over sub-implementations and execute them to get each factor data
                    message_and_df_list = multiprocessing_wrapper(
                        [
                            (implementation.execute, ("All",))
                            for implementation, fb in zip(exp.sub_workspace_list, exp.prop_dev_feedback)
                            if implementation and fb
                        ],  # only execute successfully feedback
                        n=RD_AGENT_SETTINGS.multi_proc_n,
                    )
                    for message, df in message_and_df_list:
                        # Check if factor generation was successful
                        if df is not None and "datetime" in df.index.names:
                            time_diff = df.index.get_level_values("datetime").to_series().diff().dropna().unique()
                            if pd.Timedelta(minutes=1) not in time_diff:
                                factor_dfs.append(df)

        # Combine all successful factor data
        if factor_dfs:
            return pd.concat(factor_dfs, axis=1)
        else:
            raise FactorEmptyError("No valid factor data found to merge.")
