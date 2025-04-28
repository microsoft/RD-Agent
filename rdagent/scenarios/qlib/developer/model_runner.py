from rdagent.components.runner import CachedRunner
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.scenarios.qlib.experiment.model_experiment import QlibModelExperiment
from rdagent.log import rdagent_logger as logger

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
        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        training_hyperparameters = exp.sub_tasks[0].training_hyperparameters
        if training_hyperparameters:
            env_to_use.update({
                "n_epochs": str(training_hyperparameters.get("n_epochs", "1000")),
                "lr": str(training_hyperparameters.get("lr", "2e-4")),
                "early_stop": str(training_hyperparameters.get("early_stop", 20)),
                "batch_size": str(training_hyperparameters.get("batch_size", 400)),
                "weight_decay": str(training_hyperparameters.get("weight_decay", 0.0)),
            })
        
        if exp.sub_tasks[0].model_type == "TimeSeries":
            env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})
        elif exp.sub_tasks[0].model_type == "Tabular":
            env_to_use.update({"dataset_cls": "DatasetH"})
        logger.info(f"start to run {exp.sub_tasks[0].name} model")
        # In model loop, execpt the result, we also need to store the training loop
        result, stdout = exp.experiment_workspace.execute(qlib_config_name="conf.yaml", run_env=env_to_use)

        if result is None:
            logger.error(f"Failed to run {exp.sub_tasks[0].name} model")
            raise ModelEmptyError(f"Failed to run {exp.sub_tasks[0].name} model")

        exp.result = result
        exp.stdout = stdout

        return exp
