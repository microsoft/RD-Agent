"""
Beyond previous tests
- 
"""

import json
import re
from pathlib import Path

import pandas as pd

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
        fname = "main.py"
        if "Model code test passed successfully." in stdout and implementation.file_dict.get(fname):
            stdout = implementation.execute(env=de, entry=f"python {fname}")

            # Check score file
            score_fp = implementation.workspace_path / "scores.csv"
            if not score_fp.exists():
                stdout += "\nMetrics file (scores.csv) is not generated."
            else:
                score_df = pd.read_csv(score_fp, index_col=0)
                model_set_in_scores = set(score_df.index)
                model_set_in_folder = set(
                    f[:-3] for f in implementation.file_dict.keys() if re.match(r"^model_(?!test)\w+\.py$", f)
                )
                for model in model_set_in_folder:
                    if model not in model_set_in_scores:
                        stdout += (
                            f"\nModel {model} is not evaluated in the scores.csv. The scores.csv has {model_set_in_scores}."
                        )

        system_prompt = T(".prompts:model_eval.system").r(
            task_desc=target_task.get_task_information(),
            test_code=test_code,
            scenario=self.scen.get_scenario_all_desc(),
            spec=implementation.file_dict["spec/model.md"],
        )
        user_prompt = T(".prompts:model_eval.user").r(
            stdout=stdout,
            code=implementation.file_dict[f"{target_task.name}.py"],
        )
        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
        return ModelSingleFeedback(**json.loads(resp))
