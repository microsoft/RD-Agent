# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
import json
import re
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.data_science.conf import get_clear_ws_cmd, get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.oai.llm_utils import APIBackend
from rdagent.scenarios.data_science.test_eval import get_test_eval
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent

PipelineSingleFeedback = CoSTEERSingleFeedback
PipelineMultiFeedback = CoSTEERMultiFeedback


class CodingFeedback(BaseModel):
    execution: str = Field(
        description="Describe whether the code executed successfully. Detail any errors (with full tracebacks) or critical warnings from stdout. State if execution was clean."
    )
    return_checking: str = Field(
        description="Confirm generation of the submission file. Describe verification of its format (index, columns, content plausibility) against requirements. Note any formatting issues."
    )
    competition_alignment: str = Field(
        description="Analyze whether the code might lead to a discrepancy between local validation performance and the competition's test leaderboard performance."
    )
    code: str = Field(
        description="Begin with '[Code analysis]' or '[Evaluation error]' as per Step 3. Provide the detailed analysis of code quality, instruction adherence, and alignment with competition requirements."
    )
    final_decision: bool = Field(
        description="Indicate whether the code is correct or not based on all the analysis and checks above."
    )


class CodingFeedbackNoSubmission(BaseModel):
    execution: str = Field(
        description="Describe whether the code executed successfully. Detail any errors (with full tracebacks) or critical warnings from stdout. State if execution was clean."
    )
    final_decision: bool = Field(
        description="Indicate whether the code is correct or not based on all the analysis and checks above."
    )


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

        env = get_ds_env(extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()})

        # Clean the scores.csv & submission.csv.
        implementation.execute(env=env, entry=get_clear_ws_cmd())
        stdout, execute_ret_code = implementation.execute_ret_code(env=env, entry=f"python -m coverage run main.py")
        stdout = remove_eda_part(stdout)
        stdout += f"The code executed {'successfully' if execute_ret_code == 0 else 'failed'}."

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

        test_eval = get_test_eval()
        if not test_eval.is_sub_enabled(self.scen.competition):
            submission_ret_code = 0
        else:
            # Check submission file
            base_check_code = T(".eval_tests.submission_format_test", ftype="txt").r()
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

        eda_output = implementation.file_dict.get("EDA.md", None)

        eda_output = implementation.file_dict.get("EDA.md", None)

        if not isinstance(implementation, FBWorkspace):
            eda_output = None
        else:
            eda_output = implementation.file_dict.get("EDA.md", None)

        system_prompt = T(".prompts:pipeline_eval.system").r(
            scenario=self.scen.get_scenario_all_desc(eda_output=eda_output),
            task_desc=target_task.get_task_information(),
            is_sub_enabled=test_eval.is_sub_enabled(self.scen.competition),
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


class PipelineCoSTEEREvaluatorV3(PipelineCoSTEEREvaluator):

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

        env = get_ds_env(extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()})

        # Clean the scores.csv & submission.csv.
        implementation.execute(env=env, entry=get_clear_ws_cmd())
        stdout, execute_ret_code = implementation.execute_ret_code(env=env, entry=f"python -m coverage run main.py")
        stdout = remove_eda_part(stdout)
        stdout += f"The code execution was {'successful' if execute_ret_code == 0 else 'a failure'}."

        score_fp = implementation.workspace_path / "scores.csv"
        score_ret_code = 0
        score_check_text = ""
        if not score_fp.exists():
            score_check_text = "[Scores File Checker] Metrics file (scores.csv) is not generated!"
            score_ret_code = 1
        else:
            try:
                score_df = pd.read_csv(score_fp, index_col=0)
                model_set_in_scores = set(score_df.index)

                # Check model names (index)
                if not score_df.index.is_unique:
                    score_check_text += "\n[Scores File Checker] The score dataframe contains duplicate model names."
                    score_ret_code = 1
                if "ensemble" not in model_set_in_scores:
                    score_check_text += "\n[Scores File Checker] The score dataframe doesn't contain the ensemble model."
                    score_ret_code = 1
                if score_ret_code != 0:
                    score_check_text += f"\n[Scores File Checker] score_df contains:\n```\n{score_df}\n```"

                # Check metric name (columns)
                if score_df.columns.tolist() != [self.scen.metric_name]:
                    score_check_text += f"\n[Scores File Checker] The scores dataframe does not contain the correct column names.\nCorrect columns is: ['{self.scen.metric_name}']\nBut got: {score_df.columns.tolist()}"
                    score_ret_code = 1

                # Check if scores contain NaN (values)
                if score_df.isnull().values.any():
                    nan_locations = score_df[score_df.isnull().any(axis=1)]
                    score_check_text += f"\n[Scores File Checker] The scores dataframe contains NaN values at the following locations:\n```\n{nan_locations}\n```"
                    score_ret_code = 1

            except Exception as e:
                score_check_text += f"\n[Scores File Checker] The checker crashes when checking the scores.csv file: {e}\nscores.csv's content:\n```\n{score_fp.read_text()}\n```"
                score_ret_code = 1

        test_eval = get_test_eval()
        if not test_eval.is_sub_enabled(self.scen.competition):
            submission_ret_code = 0
        else:
            # Check submission file
            base_check_code = T(".eval_tests.submission_format_test", ftype="txt").r()
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

        # NOTE(yuge): don't know why, keep as is.

        eda_output = implementation.file_dict.get("EDA.md", None)

        eda_output = implementation.file_dict.get("EDA.md", None)

        if not isinstance(implementation, FBWorkspace):
            eda_output = None
        else:
            eda_output = implementation.file_dict.get("EDA.md", None)

        is_sub_enabled = test_eval.is_sub_enabled(self.scen.competition)
        system_prompt = T(".prompts_v3:pipeline_eval.system").r(
            is_sub_enabled=is_sub_enabled,
        )
        user_prompt = T(".prompts_v3:pipeline_eval.user").r(
            scenario=self.scen.get_scenario_all_desc(eda_output=eda_output),
            task=target_task,
            stdout=stdout.strip(),
            code=implementation.file_dict["main.py"],
        )

        if is_sub_enabled:
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt,
                response_format=CodingFeedback, **kwargs
            )
            resp = CodingFeedback(**json.loads(resp))
            logger.info(f"Pipeline single feedback: {resp}")
            wfb = PipelineSingleFeedback(
                execution=resp.execution,
                return_checking=resp.return_checking,
                code=resp.code,
                final_decision=resp.final_decision,
            )
        else:
            resp = APIBackend().build_messages_and_create_chat_completion(
                user_prompt=user_prompt, system_prompt=system_prompt,
                response_format=CodingFeedbackNoSubmission, **kwargs
            )
            resp = CodingFeedbackNoSubmission(**json.loads(resp))
            logger.info(f"Pipeline single feedback: {resp}")
            wfb = PipelineSingleFeedback(
                execution=resp.execution,
                return_checking="Return checking is not available.",
                code="Code evaluation is not available.",
                final_decision=resp.final_decision,
            )

        wfb_post_updated = False
        if submission_ret_code != 0:
            wfb_post_updated = wfb.final_decision
            wfb.final_decision = False
            wfb.return_checking = "[Submission Format Test] Submission format check did not pass.\n\n---------------\n\n" + (wfb.return_checking or "")
        if score_ret_code != 0:
            wfb_post_updated = wfb_post_updated or wfb.final_decision
            wfb.final_decision = False
            wfb.return_checking = score_check_text + "[Scores File Checker] scores.csv is problematic.\n\n---------------\n\n" + (wfb.return_checking or "")
        if wfb_post_updated:
            logger.warning(f"Pipeline single feedback is set to false by rule, not by LLM:\n{wfb.return_checking}")
        return wfb
