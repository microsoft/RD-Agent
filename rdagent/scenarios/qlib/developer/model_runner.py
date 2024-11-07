from rdagent.components.runner import CachedRunner
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import cache_with_pickle
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
        if exp.sub_workspace_list[0].code_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_code(**{"model.py": exp.sub_workspace_list[0].code_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        if exp.sub_tasks[0].model_type == "TimeSeries":
            env_to_use.update({"dataset_cls": "TSDatasetH", "step_len": 20, "num_timesteps": 20})
        elif exp.sub_tasks[0].model_type == "Tabular":
            env_to_use.update({"dataset_cls": "DatasetH"})

        result = exp.experiment_workspace.execute(qlib_config_name="conf.yaml", run_env=env_to_use)

        exp.result = result

        return exp
