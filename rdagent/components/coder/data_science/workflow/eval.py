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
from rdagent.components.coder.data_science.conf import get_clear_ws_cmd, get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

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

        env = get_ds_env(extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()})

        # # DockerEnv for MLEBench submission validation
        # mle_de_conf = MLEBDockerConf()
        # mle_de_conf.extra_volumes = {
        #     f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data",
        # }
        # mde = DockerEnv(conf=mle_de_conf)
        # mde.prepare()

        # Clean the scores.csv & submission.csv.
        implementation.execute(env=env, entry=get_clear_ws_cmd())

        stdout = implementation.execute(env=env, entry=f"python -m coverage run main.py")

        # remove EDA part
        stdout = remove_eda_part(stdout)

        # Check score file
        score_fp = implementation.workspace_path / "scores.csv"
        score_ret_code = 0
        score_check_text = ""
        if not score_fp.exists():
            score_check_text = "[Error] Metrics file (scores.csv) is not generated!"
            score_ret_code = 1
            implementation.execute(env=env, entry="python -m coverage json -o coverage.json")
            coverage_report_path = implementation.workspace_path / "coverage.json"
            if coverage_report_path.exists():
                used_files = set(json.loads(coverage_report_path.read_text())["files"].keys())
                coverage_report_path.unlink()
                logger.info(f"All used scripts: {used_files}")
                if len(used_files) == 1:
                    score_check_text += f"\n[Error] The only used script is {used_files}.\nPlease check if you have implemented entry point in 'main.py'."
        else:
            try:
                score_df = pd.read_csv(score_fp, index_col=0)
                model_set_in_scores = set(score_df.index)
                # We assume that model names in `score_df` are stored without the '.py' file extension.
                model_set_in_folder = set(
                    f[:-3] for f in implementation.file_dict.keys() if re.match(r"^model_(?!test)\w+\.py$", f)
                )

                # Check model names (index)
                if model_set_in_scores != model_set_in_folder.union({"ensemble"}):
                    score_check_text += f"\n[Error] The scores dataframe does not contain the correct model names as index.\ncorrect model names are: {model_set_in_folder.union({'ensemble'})}\nscore_df is:\n{score_df}"
                    score_ret_code = 1

                # Check metric name (columns)
                if score_df.columns.tolist() != [self.scen.metric_name]:
                    score_check_text += f"\n[Error] The scores dataframe does not contain the correct column names.\nCorrect columns is: ['{self.scen.metric_name}']\nBut got: {score_df.columns.tolist()}"
                    score_ret_code = 1

                # Check if scores contain NaN (values)
                if score_df.isnull().values.any():
                    nan_locations = score_df[score_df.isnull().any(axis=1)]
                    score_check_text += f"\n[Error] The scores dataframe contains NaN values at the following locations:\n{nan_locations}"
                    score_ret_code = 1

            except Exception as e:
                score_check_text += f"\n[Error] in checking the scores.csv file: {e}\nscores.csv's content:\n-----\n{score_fp.read_text()}\n-----"
                score_ret_code = 1

        # Check submission file
        base_check_code = T(".eval_tests.submission_format_test", ftype="txt").r()
        implementation.inject_files(**{"test/submission_format_test.py": base_check_code})
        # stdout += "----Submission Check 1-----\n"
        submission_result = implementation.run(env=env, entry="python test/submission_format_test.py")
        submission_check_out = submission_result.stdout
        submission_ret_code = submission_result.exit_code
        stdout += "\n" + submission_check_out

        system_prompt = T(".prompts:workflow_eval.system").r(
            # here we pass `None` to `eda_output` because we do not have nor need EDA output for workflow.
            scenario=self.scen.get_scenario_all_desc(eda_output=None),
            task_desc=target_task.get_task_information(),
            spec=(
                implementation.file_dict["spec/workflow.md"]
                if DS_RD_SETTING.spec_enabled
                else T("scenarios.data_science.share:component_spec.Workflow").r()
            ),
        )
        user_prompt = T(".prompts:workflow_eval.user").r(
            stdout=stdout.strip(),
            code=implementation.file_dict["main.py"],
        )
        wfb = build_cls_from_json_with_retry(
            WorkflowSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=WorkflowSingleFeedback.val_and_update_init_dict,
        )
        if score_ret_code != 0:
            wfb.final_decision = False
            wfb.return_checking += "\n" + score_check_text
        if submission_ret_code != 0:
            wfb.final_decision = False
            wfb.return_checking += "\nSubmission file check failed."
        return wfb
