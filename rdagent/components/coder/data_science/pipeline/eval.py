# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
import json
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.agent.context7 import Agent as DocAgent
from rdagent.components.coder.CoSTEER import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.data_science.conf import get_clear_ws_cmd, get_ds_env
from rdagent.components.coder.data_science.share.notebook import NotebookConverter
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.scenarios.data_science.test_eval import get_test_eval
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent


@dataclass
class DSCoderFeedback(CoSTEERSingleFeedback):
    """
    Feedback for Data Science CoSTEER evaluation.
    This feedback is used to evaluate the code and execution of the Data Science CoSTEER task.
    """

    requires_documentation_search: bool | None = None  # Keep None means the feature is disabled
    error_message: str | None = None

    @staticmethod
    def val_and_update_init_dict(data: dict) -> dict:
        # First call parent class validation method to handle base fields
        data = CoSTEERSingleFeedback.val_and_update_init_dict(data)

        # Validate new fields
        if "requires_documentation_search" in data:
            if isinstance(data["requires_documentation_search"], str):
                if data["requires_documentation_search"] == "false" or data["requires_documentation_search"] == "False":
                    data["requires_documentation_search"] = False
                elif data["requires_documentation_search"] == "true" or data["requires_documentation_search"] == "True":
                    data["requires_documentation_search"] = True
                else:
                    raise ValueError(
                        f"'requires_documentation_search' string value must be 'true', 'True', 'false', or 'False', not '{data['requires_documentation_search']}'"
                    )
            elif data["requires_documentation_search"] is not None and not isinstance(
                data["requires_documentation_search"], bool
            ):
                raise ValueError(
                    f"'requires_documentation_search' must be a boolean, string, or None, not {type(data['requires_documentation_search'])}"
                )

        if "error_message" in data:
            if data["error_message"] is not None and not isinstance(data["error_message"], str):
                raise ValueError(f"'error_message' must be a string or None, not {type(data['error_message'])}")

        return data

    def __str__(self) -> str:
        base_str = super().__str__()

        if self.requires_documentation_search is not None:
            base_str += f"-------------------Documentation Search Required------------------\n{self.requires_documentation_search}\n"

        if self.error_message is not None:
            # Check if error_message contains Context7 documentation results
            if "### API Documentation Reference:" in self.error_message:
                base_str += f"-------------------Error Analysis & Documentation Search Results ------------------\n{self.error_message}\n"
            else:
                base_str += f"-------------------Error Message------------------\n{self.error_message}\n"

        return base_str

    @classmethod
    def merge(cls, feedback_li: list[CoSTEERSingleFeedback]) -> "DSCoderFeedback":
        # Call parent class merge method to handle base fields
        merged_fb = super().merge(feedback_li)

        # Convert to DSCoderFeedback type if needed
        if not isinstance(merged_fb, DSCoderFeedback):
            merged_fb = DSCoderFeedback(
                execution=merged_fb.execution,
                return_checking=merged_fb.return_checking,
                code=merged_fb.code,
                final_decision=merged_fb.final_decision,
            )

        # Merge error_message fields
        error_messages = [
            fb.error_message for fb in feedback_li if isinstance(fb, DSCoderFeedback) and fb.error_message is not None
        ]
        if error_messages:
            merged_fb.error_message = "\n\n".join(error_messages)

        # Merge requires_documentation_search fields (True if any is True)
        requires_search = [
            fb.requires_documentation_search
            for fb in feedback_li
            if isinstance(fb, DSCoderFeedback) and fb.requires_documentation_search is not None
        ]
        if requires_search:
            merged_fb.requires_documentation_search = any(requires_search)

        return merged_fb


PipelineSingleFeedback = DSCoderFeedback  # Only for compatible
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
                error_message="This task has failed too many times, skip implementation.",
                requires_documentation_search=None,
                final_decision=False,
            )

        env = get_ds_env(
            extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()},
            running_timeout_period=self.scen.real_debug_timeout(),
        )

        stdout = ""
        implementation.execute(env=env, entry=get_clear_ws_cmd())
        if DS_RD_SETTING.sample_data_by_LLM:
            # Because coder runs on full data, we need to run debug mode in advance to save time
            result = implementation.run(
                env=env, entry=f"strace -e trace=file -f -o trace.log python -m coverage run main.py --debug"
            )
        else:
            result = implementation.run(
                env=env, entry=f"strace -e trace=file -f -o trace.log python -m coverage run main.py"
            )
        result_stdout = result.get_truncated_stdout()

        nb_conversion_ret_code = 0
        nb_conversion_check_text = ""
        if DS_RD_SETTING.enable_notebook_conversion:
            notebook_converter = NotebookConverter()
            code = implementation.file_dict["main.py"]
            error_msg = notebook_converter.validate_code_format(code)
            if error_msg is not None:
                nb_conversion_check_text = error_msg
                nb_conversion_ret_code = 1
            else:
                notebook_converter.convert(
                    task=target_task,
                    code=code,
                    stdout=result_stdout,
                    outfile=implementation.workspace_path / "main.ipynb",
                    use_debug_flag=DS_RD_SETTING.sample_data_by_LLM,
                )

        sample_submission_check = True
        test_eval = get_test_eval()
        if (sample_submission_file_name := test_eval.get_sample_submission_name(self.scen.competition)) is not None:
            # check whether code ever opens the sample submission file
            if (implementation.workspace_path / "trace.log").exists():
                opened_trace_lines = [
                    line
                    for line in (implementation.workspace_path / "trace.log").read_text().splitlines()
                    if "openat" in line and sample_submission_file_name in line
                ]
                if len(opened_trace_lines) > 0:
                    stdout += f"Code opened the sample submission file '{sample_submission_file_name}' during execution.\n Reject the implementation!\n"
                    sample_submission_check = False

        result_stdout = remove_eda_part(result_stdout)
        if result.exit_code != 0:
            stdout += f"Code failed to run. Please check the stdout:\n Following the stdout of the debug mode run:\n{result_stdout.strip()}\n"
        else:
            stdout += f"Code ran successfully.\n Following the stdout of the debug mode run:\n{result_stdout.strip()}\n"
        if DS_RD_SETTING.sample_data_by_LLM:
            debug_time, full_estimated_time = None, None
            if match := re.search(r"debug_time:\s*(\d+(?:.\d+)?)", result_stdout, re.DOTALL):
                debug_time = float(match.group(1))
            if match := re.search(r"estimated_time:\s*(\d+(?:.\d+)?)", result_stdout, re.DOTALL):
                full_estimated_time = float(match.group(1))
            if debug_time is not None and full_estimated_time is not None:
                stdout += f"Debug mode ran in {debug_time:.2f} seconds, estimated full run time is {full_estimated_time:.2f} seconds. The estimated time is {full_estimated_time / env.conf.running_timeout_period * 100:.2f}% the debug time."
            else:
                stdout += "Debug mode did not provide debug_time or estimated_time, it's a buggy implementation.\n"

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
                    score_check_text += "\n[Error] The file 'scores.csv' contains duplicate model names."
                    score_ret_code = 1
                if "ensemble" not in model_set_in_scores:
                    score_check_text += "\n[Error] The file 'scores.csv' doesn't contain the ensemble model."
                    score_ret_code = 1
                if score_ret_code != 0:
                    score_check_text += f"The dataframe in file 'scores.csv' is:\n{score_df}"

                # Check metric name (columns) - case insensitive
                if [col.lower() for col in score_df.columns.tolist()] != [self.scen.metric_name.lower()]:
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
        if DS_RD_SETTING.sample_data_by_LLM and test_eval.enabled(self.scen.competition):
            submission_check_out, submission_ret_code = test_eval.valid(self.scen.competition, implementation)
            stdout += f"\n### Submission check:\n{submission_check_out}\nIf Submission check returns a 'Submission is valid' or similar message, despite some warning messages, you should still consider the submission as valid and give a positive final decision. "
        elif not test_eval.is_sub_enabled(self.scen.competition):
            submission_ret_code = 0
        else:
            # Check submission file
            base_check_code = T(".eval_tests.submission_format_test", ftype="txt").r()
            implementation.inject_files(**{"test/submission_format_test.py": base_check_code})
            # stdout += "----Submission Check 1-----\n"
            submission_result = implementation.run(env=env, entry="python test/submission_format_test.py")
            submission_check_out = submission_result.get_truncated_stdout()
            submission_ret_code = submission_result.exit_code
            stdout += "\n" + submission_check_out

        if not isinstance(implementation, FBWorkspace):
            eda_output = None
        else:
            eda_output = implementation.file_dict.get("EDA.md", None)

        # extract enable_mcp_documentation_search from data science configuration
        enable_mcp_documentation_search = DS_RD_SETTING.enable_mcp_documentation_search

        queried_similar_successful_knowledge = (
            queried_knowledge.task_to_similar_task_successful_knowledge[target_task.get_task_information()]
            if queried_knowledge is not None
            else []
        )

        system_prompt = T(".prompts:pipeline_eval.system").r(
            is_sub_enabled=test_eval.is_sub_enabled(self.scen.competition),
            debug_mode=DS_RD_SETTING.sample_data_by_LLM,
            enable_mcp_documentation_search=enable_mcp_documentation_search,
            mle_check=DS_RD_SETTING.sample_data_by_LLM,
            queried_similar_successful_knowledge=queried_similar_successful_knowledge,
        )
        user_prompt = T(".prompts:pipeline_eval.user").r(
            scenario=self.scen.get_scenario_all_desc(eda_output=eda_output),
            task_desc=target_task.get_task_information(),
            stdout=stdout.strip(),
            spec=T("scenarios.data_science.share:component_spec.Pipeline").r(
                metric_name=self.scen.metric_name,
                enable_notebook_conversion=DS_RD_SETTING.enable_notebook_conversion,
            ),
            code=implementation.file_dict["main.py"],
        )
        wfb = build_cls_from_json_with_retry(
            PipelineSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=PipelineSingleFeedback.val_and_update_init_dict,
        )

        # judge whether we should perform documentation search
        do_documentation_search = enable_mcp_documentation_search and wfb.requires_documentation_search

        if do_documentation_search:
            # Use MCPAgent for clean, user-friendly interface
            try:
                # Create agent targeting Context7 service - model config comes from mcp_config.json
                doc_agent = DocAgent()

                # Synchronous query - perfect for evaluation context
                if wfb.error_message:  # Type safety check
                    context7_result = doc_agent.query(query=wfb.error_message)

                    if context7_result:
                        logger.info("Context7: Documentation search completed successfully")
                        wfb.error_message += f"\n\n### API Documentation Reference:\nThe following API documentation was retrieved based on the error. This provides factual information about API changes or parameter specifications only:\n\n{context7_result}"
                    else:
                        logger.warning("Context7: Documentation search failed or no results found")
                else:
                    logger.warning("Context7: No error message to search for")

            # TODO: confirm what exception will be raised when timeout
            # except concurrent.futures.TimeoutError:
            #     logger.error("Context7: Query timed out after 180 seconds")
            except Exception as e:
                error_msg = str(e) if str(e) else type(e).__name__
                logger.error(f"Context7: Query failed - {error_msg}")

        if score_ret_code != 0 and wfb.final_decision is True:
            wfb.final_decision = False
            wfb.return_checking += "\n" + score_check_text
        if submission_ret_code != 0 and wfb.final_decision is True:
            wfb.final_decision = False
            wfb.return_checking += "\nSubmission file check failed."
        if sample_submission_check is False and wfb.final_decision is True:
            wfb.final_decision = False
            wfb.return_checking += (
                "\nSample submission file check failed. Code should not open the sample submission file."
            )
        if nb_conversion_ret_code != 0 and wfb.final_decision is True:
            wfb.final_decision = False
            wfb.return_checking += "\n" + nb_conversion_check_text
        return wfb
