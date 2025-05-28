from rdagent.components.runner import CachedRunner
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_mining.experiment.model_experiment import DMModelExperiment


class DMModelRunner(CachedRunner[DMModelExperiment]):
    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: DMModelExperiment) -> DMModelExperiment:
        if exp.sub_workspace_list[0].file_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_files(**{"model.py": exp.sub_workspace_list[0].file_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        result, stdout = exp.experiment_workspace.execute(run_env=env_to_use)

        if result is None:
            logger.error(f"Experiment failed to run, stdout: {stdout}")
            raise ModelEmptyError(f"Failed to run this experiment, because {stdout}")

        exp.result = result

        return exp
