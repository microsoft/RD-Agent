from rdagent.core.developer import Developer
from rdagent.core.experiment import ASpecificExp, Experiment
from rdagent.oai.llm_utils import md5_hash


class CachedRunner(Developer[ASpecificExp]):
    def get_cache_key(self, exp: Experiment) -> str:
        all_tasks = []
        for based_exp in exp.based_experiments:
            all_tasks.extend(based_exp.sub_tasks)
        all_tasks.extend(exp.sub_tasks)
        task_info_list = [task.get_task_information() for task in all_tasks]
        task_info_str = "\n".join(task_info_list)
        
        # Include base_features and base_feature_codes for QlibFactorExperiment
        if hasattr(exp, 'base_features') and exp.base_features:
            task_info_str += "\nBASE_FEATURES:" + str(sorted(exp.base_features.keys()))
        if hasattr(exp, 'base_feature_codes') and exp.base_feature_codes:
            task_info_str += "\nBASE_CODES:" + str(sorted(exp.base_feature_codes.keys()))
        
        return md5_hash(task_info_str)

    def assign_cached_result(self, exp: Experiment, cached_res: Experiment) -> Experiment:
        if exp.based_experiments and exp.based_experiments[-1].result is None:
            exp.based_experiments[-1].result = cached_res.based_experiments[-1].result
        exp.result = cached_res.result
        return exp
