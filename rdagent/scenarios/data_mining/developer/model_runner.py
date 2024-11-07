from rdagent.components.runner import CachedRunner
from rdagent.core.exception import ModelEmptyError
from rdagent.core.utils import cache_with_pickle
from rdagent.scenarios.data_mining.experiment.model_experiment import DMModelExperiment


class DMModelRunner(CachedRunner[DMModelExperiment]):
    @cache_with_pickle(CachedRunner.get_cache_key, CachedRunner.assign_cached_result)
    def develop(self, exp: DMModelExperiment) -> DMModelExperiment:
        if exp.sub_workspace_list[0].code_dict.get("model.py") is None:
            raise ModelEmptyError("model.py is empty")
        # to replace & inject code
        exp.experiment_workspace.inject_code(**{"model.py": exp.sub_workspace_list[0].code_dict["model.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        exp.result = result

        return exp
