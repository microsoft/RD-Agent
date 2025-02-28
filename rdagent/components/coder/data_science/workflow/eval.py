import json
import re
from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry
from rdagent.utils.env import DockerEnv, DSDockerConf, MLEBDockerConf

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
    ) -> CoSTEERSingleFeedback:
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

        # DockerEnv for Kaggle Competition
        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {
            f"{DS_RD_SETTING.local_data_path}/sample/{self.scen.competition}": "/kaggle/input"
        }
        de = DockerEnv(conf=ds_docker_conf)

        # # DockerEnv for MLEBench submission validation
        # mle_de_conf = MLEBDockerConf()
        # mle_de_conf.extra_volumes = {
        #     f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data",
        # }
        # mde = DockerEnv(conf=mle_de_conf)
        # mde.prepare()

        # Clean the scores.csv & submission.csv.
        implementation.execute(env=de, entry=f"rm submission.csv scores.csv")

        stdout = implementation.execute(env=de, entry=f"python main.py")
        stdout = re.sub(r"=== Start of EDA part ===(.*)=== End of EDA part ===", "", stdout)

        # Check score file
        score_fp = implementation.workspace_path / "scores.csv"
        score_ret_code = 0
        score_check_text = ""
        if not score_fp.exists():
            score_check_text = "[Error] Metrics file (scores.csv) is not generated!"
            score_ret_code = 1
        else:
            try:
                score_df = pd.read_csv(score_fp, index_col=0)
                model_set_in_scores = set(score_df.index)
                model_set_in_folder = set(
                    f[:-3] for f in implementation.file_dict.keys() if re.match(r"^model_(?!test)\w+\.py$", f)
                )
                for model in model_set_in_folder:
                    if model not in model_set_in_scores:
                        score_check_text += f"\nModel {model} is not evaluated in the scores.csv. The scores.csv has {model_set_in_scores}."
                        score_ret_code = 1
            except Exception as e:
                score_check_text += f"\nError in checking the scores.csv file: {e}\nscores.csv's content:\n-----\n{score_fp.read_text()}\n-----"
                score_ret_code = 1

        # Check submission file
        base_check_code = (DIRNAME / "eval_tests" / "submission_format_test.txt").read_text()
        implementation.inject_files(**{"test/submission_format_test.py": base_check_code})
        # stdout += "----Submission Check 1-----\n"
        submission_check_out, submission_ret_code = implementation.execute_ret_code(
            env=de, entry="python test/submission_format_test.py"
        )
        stdout += "\n" + submission_check_out

        # MLEBench Check
        # !!! Since we are running on a sampled dataset, mlebench check is not required.
        # mle_check_code = (
        #     (DIRNAME / "eval_tests" / "mle_submission_format_test.txt")
        #     .read_text()
        #     .replace("<competition_id>", self.scen.competition)
        # )
        # implementation.inject_files(**{"test/mle_submission_format_test.py": mle_check_code})
        # stdout += "----Submission Check 2-----\n"
        # stdout += implementation.execute(env=mde, entry=f"python test/mle_submission_format_test.py")

        system_prompt = T(".prompts:workflow_eval.system").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            spec=implementation.file_dict["spec/workflow.md"],
        )
        user_prompt = T(".prompts:workflow_eval.user").r(
            stdout=stdout.strip(),
            code=implementation.file_dict["main.py"],
        )
        wfb = build_cls_from_json_with_retry(
            WorkflowSingleFeedback, system_prompt=system_prompt, user_prompt=user_prompt
        )
        if score_ret_code != 0:
            wfb.final_decision = False
            wfb.execution += "\n" + score_check_text
        if submission_ret_code != 0:
            wfb.final_decision = False
            wfb.execution += "\nSubmission file check failed."
        return wfb
