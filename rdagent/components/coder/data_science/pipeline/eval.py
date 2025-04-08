# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
import json
import re
from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent

PipelineSingleFeedback = CoSTEERSingleFeedback
PipelineMultiFeedback = CoSTEERMultiFeedback


class PipelineCoSTEEREvaluator(CoSTEEREvaluator):

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: CoSTEERQueriedKnowledgeV2 = None,
        **kwargs,
    ) -> PipelineSingleFeedback:

        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return PipelineSingleFeedback(
                execution="This task has failed too many times, skip implementation.",
                return_checking="This task has failed too many times, skip implementation.",
                code="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        env = get_ds_env()
        env.conf.extra_volumes = {f"{DS_RD_SETTING.local_data_path}/sample/{self.scen.competition}": "/kaggle/input"}

        # Clean the scores.csv & submission.csv.
        implementation.execute(env=env, entry=f"rm submission.csv scores.csv")
        stdout, execute_ret_code = implementation.execute_ret_code(env=env, entry=f"python main.py")
        stdout = re.sub(r"=== Start of EDA part ===(.*)=== End of EDA part ===", "", stdout)

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

                # Check model names (index)
                if not score_df.index.is_unique:
                    score_check_text += "\n[Error] The score dataframe contains duplicate model names."
                    score_ret_code = 1
                if "ensemble" not in model_set_in_scores:
                    score_check_text += "\n[Error] The score dataframe doesn't contain the ensemble model."
                    score_ret_code = 1
                if score_ret_code != 0:
                    score_check_text += f"The score_df is:\n{score_df}"

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
        base_check_code = (DIRNAME / "eval_tests" / "submission_format_test.txt").read_text()
        implementation.inject_files(**{"test/submission_format_test.py": base_check_code})
        # stdout += "----Submission Check 1-----\n"
        submission_check_out, submission_ret_code = implementation.execute_ret_code(
            env=env, entry="python test/submission_format_test.py"
        )
        if DS_RD_SETTING.rule_base_eval:
            if execute_ret_code == 0 and score_ret_code == 0 and submission_ret_code == 0:
                return PipelineSingleFeedback(
                    execution=stdout,
                    return_checking=score_check_text + "\n" + submission_check_out,
                    code="Code evaluation is not available.",
                    final_decision=True,
                )
            else:
                return PipelineSingleFeedback(
                    execution=stdout,
                    return_checking=score_check_text + "\n" + submission_check_out,
                    code="Code evaluation is not available.",
                    final_decision=False,
                )
        stdout += "\n" + submission_check_out

        system_prompt = T(".prompts:pipeline_eval.system").r(
            scenario=self.scen.get_scenario_all_desc(),
            task_desc=target_task.get_task_information(),
            spec=T("scenarios.data_science.share:component_spec.Pipeline").r(),
        )
        user_prompt = T(".prompts:pipeline_eval.user").r(
            stdout=stdout.strip(),
            code=implementation.file_dict["main.py"],
        )
        wfb = build_cls_from_json_with_retry(
            PipelineSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=PipelineSingleFeedback.val_and_update_init_dict,
        )
        if score_ret_code != 0:
            wfb.final_decision = False
            wfb.return_checking += "\n" + score_check_text
        if submission_ret_code != 0:
            wfb.final_decision = False
            wfb.return_checking += "\nSubmission file check failed."
        return wfb
