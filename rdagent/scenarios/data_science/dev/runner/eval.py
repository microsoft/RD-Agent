import json
import re
from pathlib import Path

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.conf import get_clear_ws_cmd, get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.log import rdagent_logger as logger
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry
from rdagent.utils.fmt import shrink_text

DIRNAME = Path(__file__).absolute().resolve().parent

DSCoSTEEREvalFeedback = CoSTEERSingleFeedback


class DSCoSTEERCoSTEEREvaluator(CoSTEEREvaluator):

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> DSCoSTEEREvalFeedback:

        env = get_ds_env(
            extra_volumes={
                f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}": T(
                    "scenarios.data_science.share:scen.input_path"
                ).r()
            },
            running_timeout_period=DS_RD_SETTING.full_timeout,
        )

        stdout = implementation.execute(
            env=env, entry=get_clear_ws_cmd()
        )  # Remove previous submission and scores files generated by worklfow.

        # execute workflow
        stdout, execute_ret_code = implementation.execute_ret_code(env=env, entry="python -m coverage run main.py")
        match = re.search(r"(.*?)=== Start of EDA part ===(.*)=== End of EDA part ===", stdout, re.DOTALL)
        eda_output = match.groups()[1] if match else None
        if eda_output is None:
            eda_output = "No EDA output."
        implementation.inject_files(**{"EDA.md": eda_output})
        stdout = remove_eda_part(stdout)

        # Check score file
        score_fp = implementation.workspace_path / "scores.csv"
        score_ret_code = 0
        score_check_text = ""
        if not score_fp.exists():
            logger.warning("Metrics file (scores.csv) is not generated!")
            score_check_text = "[Error] Metrics file (scores.csv) is not generated!"
            score_ret_code = 1
        else:
            try:
                score_df = pd.read_csv(score_fp, index_col=0)
                model_set_in_scores = set(score_df.index)
                model_set_in_folder = set(
                    f[:-3] for f in implementation.file_dict.keys() if re.match(r"^model_(?!test)\w+\.py$", f)
                )

                # Check model names (index)
                # in Pipeline task, we only check ensemble in scores.csv
                if DS_RD_SETTING.coder_on_whole_pipeline:
                    if not score_df.index.is_unique:
                        score_check_text += "\n[Error] The score dataframe contains duplicate model names."
                        score_ret_code = 1
                    if "ensemble" not in model_set_in_scores:
                        score_check_text += "\n[Error] The score dataframe doesn't contain the ensemble model."
                        score_ret_code = 1
                    if score_ret_code != 0:
                        score_check_text += f"The score_df is:\n{score_df}"
                else:
                    if model_set_in_scores != model_set_in_folder.union({"ensemble"}):
                        score_check_text += f"\n[Error] The scores dataframe does not contain the correct model names as index.\ncorrect model names are: {model_set_in_folder.union({'ensemble'})}\nscore_df is:\n{score_df}"
                        score_ret_code = 1

                # Check metric name (columns)
                if score_df.columns.tolist() != [self.scen.metric_name]:
                    score_check_text += f"\n[Error] The scores dataframe does not contain the correct column names.\nCorrect columns is: ['{self.scen.metric_name}']\nBut got: {score_df.columns.tolist()}"
                    score_ret_code = 1

            except Exception as e:
                logger.error(f"Error in checking the scores.csv file: {e}")
                score_check_text += f"\n[Error] in checking the scores.csv file: {e}\nscores.csv's content:\n-----\n{score_fp.read_text()}\n-----"
                score_ret_code = 1

        # DockerEnv for MLEBench submission validation
        submission_check_out = ""

        if DS_RD_SETTING.if_using_mle_data:
            mde = get_ds_env(
                conf_type="mlebench",
                extra_volumes={
                    f"{DS_RD_SETTING.local_data_path}/zip_files": "/mle/data",
                },
            )
            mde.prepare()
            # MLEBench Check
            mle_check_code = (
                (Path(__file__).absolute().resolve().parent / "eval_tests" / "mle_submission_format_test.txt")
                .read_text()
                .replace("<competition_id>", self.scen.competition)
            )
            implementation.inject_files(**{"test/mle_submission_format_test.py": mle_check_code})
            submission_check_out, submission_ret_code = implementation.execute_ret_code(
                env=mde, entry="python test/mle_submission_format_test.py"
            )
            stdout += f"\nMLEBench submission check:\n{submission_check_out}\nIf MLEBench submission check returns a 'Submission is valid' or similar message, despite some warning messages, you should still consider the submission as valid and give a positive final decision. "
            implementation.inject_files(**{"test/mle_submission_format_test.output": submission_check_out})

        if DS_RD_SETTING.rule_base_eval:
            if DS_RD_SETTING.if_using_mle_data:
                score_check_text = score_check_text + "\n" + submission_check_out
            if (
                execute_ret_code == 0
                and score_ret_code == 0
                and (not DS_RD_SETTING.if_using_mle_data or submission_ret_code == 0)
            ):
                return DSCoSTEEREvalFeedback(
                    execution=stdout,
                    return_checking=score_check_text,
                    code="Code evaluation is not available.",
                    final_decision=True,
                )
            else:
                return DSCoSTEEREvalFeedback(
                    execution=stdout,
                    return_checking=score_check_text,
                    code="Code evaluation is not available.",
                    final_decision=False,
                )

        system_prompt = T(".prompts:DSCoSTEER_eval.system").r(
            scenario=self.scen.get_scenario_all_desc(eda_output=implementation.file_dict.get("EDA.md", None)),
            task_desc=target_task.get_task_information(),
        )
        user_prompt = T(".prompts:DSCoSTEER_eval.user").r(
            code=implementation.all_codes,
            stdout=shrink_text(stdout),
        )

        feedback = build_cls_from_json_with_retry(
            DSCoSTEEREvalFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=DSCoSTEEREvalFeedback.val_and_update_init_dict,
        )

        if feedback and not DS_RD_SETTING.coder_on_whole_pipeline:
            # remove unused files
            implementation.execute(env=env, entry="python -m coverage json -o coverage.json")
            coverage_report_path = implementation.workspace_path / "coverage.json"
            if coverage_report_path.exists():
                used_files = set(json.loads(coverage_report_path.read_text())["files"].keys())
                coverage_report_path.unlink()
                logger.info(f"All used scripts: {used_files}")

                use_one_model = False
                for f in used_files:
                    if f.startswith("model_") and "test" not in f:
                        use_one_model = True
                        break

                if not use_one_model:
                    feedback.final_decision = False
                    logger.warning("No model script is used in `main.py`.")
                    feedback.code += "\n[Error] No model script is used in `main.py`."

                all_python_files = set(Path(implementation.workspace_path).rglob("*.py"))
                must_have_files = ["load_data.py", "feature.py", "ensemble.py"]

                unused_files = [
                    py_file.name
                    for py_file in all_python_files
                    if not (py_file.name in used_files or py_file.name.endswith("test.py"))
                ]
                if unused_files:
                    logger.warning(f"Unused scripts: {unused_files}")
                    error_files = set(unused_files).intersection(set(must_have_files))
                    if error_files:
                        feedback.final_decision = False
                        logger.warning(f"{error_files} must be used in `main.py`.")
                        feedback.code += f"\n[Error] {error_files} must be used in `main.py`."
                    elif use_one_model:
                        logger.info("Remove unused scripts.")
                        implementation.inject_files(**{file: implementation.DEL_KEY for file in unused_files})

        if score_ret_code != 0:
            feedback.final_decision = False
            feedback.return_checking += "\n" + score_check_text
        if DS_RD_SETTING.if_using_mle_data and submission_ret_code != 0:
            feedback.final_decision = False
            feedback.return_checking += "\nSubmission file check failed."
        return feedback
