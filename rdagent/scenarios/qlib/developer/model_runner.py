import pandas as pd

from rdagent.components.runner import CachedRunner
from rdagent.core.conf import RD_AGENT_SETTINGS
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.qlib.developer.utils import process_factor_data
from rdagent.scenarios.qlib.experiment.factor_experiment import QlibFactorExperiment
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment


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
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1] = self.develop(exp.based_experiments[-1])

        exist_sota_factor_exp = False
        if exp.based_experiments:
            SOTA_factor = None
            # Filter and retain only QlibFactorExperiment instances
            sota_factor_experiments_list = [
                base_exp for base_exp in exp.based_experiments if isinstance(base_exp, QlibFactorExperiment)
            ]
            if len(sota_factor_experiments_list) > 1:
                logger.info(f"SOTA factor processing ...")
                SOTA_factor = process_factor_data(sota_factor_experiments_list)

            if SOTA_factor is not None and not SOTA_factor.empty:
                exist_sota_factor_exp = True
                combined_factors = SOTA_factor
                combined_factors = combined_factors.sort_index()
                combined_factors = combined_factors.loc[:, ~combined_factors.columns.duplicated(keep="last")]
                new_columns = pd.MultiIndex.from_product([["feature"], combined_factors.columns])
                combined_factors.columns = new_columns
                num_features = str(RD_AGENT_SETTINGS.initial_fator_library_size + len(combined_factors.columns))

                target_path = exp.experiment_workspace.workspace_path / "combined_factors_df.parquet"

                # Save the combined factors to the workspace
                combined_factors.to_parquet(target_path, engine="pyarrow")

        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        training_hyperparameters = exp.sub_tasks[0].training_hyperparameters
        if training_hyperparameters:
            env_to_use.update(
                {
                    "n_epochs": str(training_hyperparameters.get("n_epochs", "100")),
                    "lr": str(training_hyperparameters.get("lr", "1e-3")),
                    "early_stop": str(training_hyperparameters.get("early_stop", 10)),
                    "batch_size": str(training_hyperparameters.get("batch_size", 256)),
                    "weight_decay": str(training_hyperparameters.get("weight_decay", 0.0001)),
                }
            )

        logger.info(f"start to run {exp.sub_tasks[0].name} model")
        if exp.sub_tasks[0].model_type == "TimeSeries":
            if exist_sota_factor_exp:
                env_to_use.update(
                    {"dataset_cls": "TSDatasetH", "num_features": num_features, "step_len": 20, "num_timesteps": 20}
                )
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_sota_factors_model.yaml", run_env=env_to_use
                )
            else:
                env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline_factors_model.yaml", run_env=env_to_use
                )
        elif exp.sub_tasks[0].model_type == "Tabular":
            if exist_sota_factor_exp:
                env_to_use.update({"dataset_cls": "DatasetH", "num_features": num_features})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_sota_factors_model.yaml", run_env=env_to_use
                )
            else:
                env_to_use.update({"dataset_cls": "DatasetH"})
                result, stdout = exp.experiment_workspace.execute(
                    qlib_config_name="conf_baseline_factors_model.yaml", run_env=env_to_use
                )

        exp.result = result
        exp.stdout = stdout

        if result is None:
            logger.error(f"Failed to run {exp.sub_tasks[0].name}, because {stdout}")
            raise ModelEmptyError(f"Failed to run {exp.sub_tasks[0].name} model, because {stdout}")

        return exp
