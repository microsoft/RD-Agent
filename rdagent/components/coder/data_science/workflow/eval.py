import json
import re
from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
    CoSTEERSingleFeedbackDeprecated,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf

DIRNAME = Path(__file__).absolute().resolve().parent

WorkflowSingleFeedback = CoSTEERSingleFeedback
WorkflowMultiFeedback = CoSTEERMultiFeedback


class WorkflowGeneralCaseSpecEvaluator(CoSTEEREvaluator):
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
    ) -> CoSTEERSingleFeedbackDeprecated:
        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return WorkflowSingleFeedback(
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
        fname = "main.py"
        stdout = implementation.execute(env=de, entry=f"python {fname}")

        # Check score file
        score_fp = implementation.workspace_path / "scores.csv"
        if not score_fp.exists():
            stdout += "\nMetrics file (scores.csv) is not generated."
        else:
            score_df = pd.read_csv(score_fp, index_col=0)
            model_set_in_scores = set(score_df.index)
            model_set_in_folder = set(
                f[:-3] for f in implementation.file_dict.keys() if re.match(r"^model_.+\.py$", f) and "test" not in f
            )
            for model in model_set_in_folder:
                if model not in model_set_in_scores:
                    stdout += (
                        f"\nModel {model} is not evaluated in the scores.csv. The scores.csv has {model_set_in_scores}."
                    )

        # Check submission file
        submission_fp = implementation.workspace_path / "submission.csv"
        if not submission_fp.exists():
            stdout += "\nSubmission file (submission.csv) is not generated."

        system_prompt = T(".prompts:workflow_eval.system").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            spec=implementation.file_dict["spec/workflow.md"],
        )
        user_prompt = T(".prompts:workflow_eval.user").r(
            stdout=stdout,
            code=implementation.file_dict["main.py"],
        )
        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
        return WorkflowSingleFeedback(**json.loads(resp))
