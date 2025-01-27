"""
Beyond previous tests
- 
"""

import json
import re
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.exception import CoderError
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry
from rdagent.utils.env import DockerEnv, DSDockerConf

DIRNAME = Path(__file__).absolute().resolve().parent
ModelSingleFeedback = CoSTEERSingleFeedback


# Below are unit tests for testing the specification of the implemented model ------------------
class ModelGeneralCaseSpecEvaluator(CoSTEEREvaluator):
    """
    Motivation case:
    - Simplest case, we already split the data into train_data, valid_data, and test_data. We require the model to learn (optionally validate on valid data), and infer on test data.

    Test workflow:
    - Build train, valid, and test data to run it, and test the output (e.g., shape, etc.)
    """

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> ModelSingleFeedback:
        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return ModelSingleFeedback(
                execution="This task has failed too many times, skip implementation.",
                return_checking="This task has failed too many times, skip implementation.",
                code="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {
            f"{DS_RD_SETTING.local_data_path}/sample/{self.scen.competition}": "/kaggle/input"
        }
        de = DockerEnv(conf=ds_docker_conf)

        fname = "model_test.py"
        test_code = (
            (DIRNAME / "eval_tests" / "model_test.txt").read_text().replace("model01", target_task.name)
        )  # only check the model changed this time
        implementation.inject_files(**{fname: test_code})
        stdout = implementation.execute(env=de, entry=f"python {fname}")

        if stdout is None:
            raise CoderError(
                "The execution output contains too many progress bars and results in the LLM's token size exceeding the limit."
            )

        if "main.py" in implementation.file_dict:
            workflow_stdout = implementation.execute(env=de, entry="python main.py")
        else:
            workflow_stdout = None

        system_prompt = T(".prompts:model_eval.system").r(
            task_desc=target_task.get_task_information(),
            test_code=test_code,
            scenario=self.scen.get_scenario_all_desc(),
            spec=implementation.file_dict["spec/model.md"],
            workflow_stdout=workflow_stdout,
            workflow_code=implementation.all_codes,
        )
        user_prompt = T(".prompts:model_eval.user").r(
            stdout=stdout,
            code=implementation.file_dict[f"{target_task.name}.py"],
            workflow_stdout=workflow_stdout,
        )
        return build_cls_from_json_with_retry(ModelSingleFeedback, system_prompt=system_prompt, user_prompt=user_prompt)
