from scripts.benchmark.config import TASK_VERSION
from rdagent.core.implementation import TaskImplementation
from typing import List
from rdagent.core.task import FactorImplementTask, TestCase
# TODO:Need to verify the type of input data，how to deal with the gt

# (haoxue) need to check the following code, it seems that there exists task.py 
class task(object):
    def __init__(self, task_name, task_description, task_formulation, task_formulation_description, variables: dict = {}, resource: str = None):
        self.task_name = task_name
        self.task_description = task_description
        self.task_formulation = task_formulation
        self.task_formulation_description = task_formulation_description
        self.variables = variables
        self.task_resources = resource
    def task_key_adaptor(self, fname, task):
        # FIXME: we should align the code and task to make the interface simpler
        res = {"factor_name": fname}
        for k, v in task.items():
            res[
                {
                    "formulation": "factor_formulation",
                    "description": "factor_description",
                    "variable": "variables",
                }.get(k, k)
            ] = v
            res.update(
                {
                    # "factor_description": "",
                    "factor_formulation_description": "",
                },
            )
        return res

    def task_set_adaptor(self, task_set):
        res = {}
        for k, f_d in task_set.items():
            new_f_d = [self.task_key_adaptor(f, d) for f, d in f_d.items()]
            res[k] = new_f_d
        return res

    def load_all_task_json_disk(self, path):
        with open(path) as f:
            data = json.load(f)
        return self.task_set_adaptor(data)


def get_test_task_json(version: TASK_VERSION, path):
    if version == "":
        res=task().load_all_task_json_disk(path=path)
    elif version == "random":
        pass
    elif version == "Naive":
        pass
    elif version == "CoT":
        pass
    elif version == "Past":
        pass
    else:
        raise ValueError(f"Unknown version: {version}")
    return res

def load_tasks(data: dict, with_fname=False):
    # TODO: we should put these into a staticmethod as BaseEval (maybe FactorImplementTask)
    # load tasks from json
    ft_l = []
    for fname, factor_list in data.items():
        for t in factor_list:
            ft = FactorImplementTask.from_dict(t)
            # key in factor
            if "variables" in t:
                ft.factor_formulation_description = str(t["variables"])
            if with_fname:
                ft_l.append((fname, ft))
            else:
                ft_l.append(ft)
    return ft_l

def load_eval_data(version: TASK_VERSION, path) -> List[TestCase]:
    # prepare the input data used for generation
    # The process should contain: 1. read the factor/model info, 2. Prepare and check the gt
    all_task_json = get_test_task_json(version, path)
    return load_tasks(all_task_json)
