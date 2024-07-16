import pickle
from pathlib import Path
from typing import Tuple

from rdagent.components.runner.conf import RUNNER_SETTINGS
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.core.task_generator import TaskGenerator
from rdagent.oai.llm_utils import md5_hash


class CachedRunner(TaskGenerator[ASpecificExp]):
    def get_cache_key(self, exp: Experiment) -> str:
        all_tasks = []
        for based_exp in exp.based_experiments:
            all_tasks.extend(based_exp.sub_tasks)
        all_tasks.extend(exp.sub_tasks)
        task_info_list = [task.get_task_information() for task in all_tasks]
        task_info_str = "\n".join(task_info_list)
        return md5_hash(task_info_str)

    def get_cache_result(self, exp: Experiment) -> Tuple[bool, object]:
        task_info_key = self.get_cache_key(exp)
        Path(RUNNER_SETTINGS.runner_cache_path).mkdir(parents=True, exist_ok=True)
        cache_path = Path(RUNNER_SETTINGS.runner_cache_path) / f"{task_info_key}.pkl"
        if cache_path.exists():
            return True, pickle.load(open(cache_path, "rb"))
        else:
            return False, None

    def dump_cache_result(self, exp: Experiment, result: object):
        task_info_key = self.get_cache_key(exp)
        cache_path = Path(RUNNER_SETTINGS.runner_cache_path) / f"{task_info_key}.pkl"
        pickle.dump(result, open(cache_path, "wb"))
