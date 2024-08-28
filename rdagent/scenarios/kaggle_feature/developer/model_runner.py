from pathlib import Path

from rdagent.components.runner import CachedRunner
from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.exception import FactorEmptyError
from rdagent.scenarios.kaggle_feature.experiment.feature_experiment import FEFeatureExperiment


class KGFeatureRunner(CachedRunner[FEFeatureExperiment]):
    def develop(self, exp: FEFeatureExperiment) -> FEFeatureExperiment:
        if RUNNER_SETTINGS.cache_result:
            cache_hit, result = self.get_cache_result(exp)
            if cache_hit:
                exp.result = result
                return exp

        #TODO 多个feat文件
        if exp.sub_workspace_list[0].code_dict.get("feat.py") is None:
            raise FactorEmptyError("feat.py is empty")
        exp.experiment_workspace.inject_code(**{"feat.py": exp.sub_workspace_list[0].code_dict["feat.py"]})

        env_to_use = {"PYTHONPATH": "./"}

        result = exp.experiment_workspace.execute(run_env=env_to_use)

        exp.result = result
        if RUNNER_SETTINGS.cache_result:
            self.dump_cache_result(exp, result)

        return exp
