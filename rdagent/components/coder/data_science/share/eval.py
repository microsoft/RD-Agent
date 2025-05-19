from pathlib import Path
from typing import Literal

import pandas as pd

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.conf import get_clear_ws_cmd, get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent

PipelineSingleFeedback = CoSTEERSingleFeedback
PipelineMultiFeedback = CoSTEERMultiFeedback

NO_SUB = "<No submission.csv file found.>"
NO_SCORE = "<No scores.csv file found.>"


class ModelDumpEvaluator(CoSTEEREvaluator):
    """This evaluator assumes that it runs after the model"""

    def __init__(self, scen: Scenario, data_type: Literal["sample", "full"]):
        super().__init__(scen)
        self.data_type = data_type

    def evaluate(
        self, target_task: Task, implementation: FBWorkspace, gt_implementation: FBWorkspace, *kargs, **kwargs
    ) -> CoSTEERSingleFeedback:

        model_folder = implementation.workspace_path / "models"
        # 1) Check if the model_folder is not empty
        if not model_folder.exists() or not any(model_folder.iterdir()):
            err_msg = "Model folder (`models` sub folder) is empty or does not exist. The model is not dumped."
            return CoSTEERSingleFeedback(
                execution=err_msg,
                return_checking=err_msg,
                code=err_msg,
                final_decision=False,
            )

        data_source_path = (
            f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}"
            if self.data_type == "full"
            else self.scen.debug_path
        )
        env = get_ds_env(
            extra_volumes={data_source_path: T("scenarios.data_science.share:scen.input_path").r()},
            running_timeout_period=(
                DS_RD_SETTING.full_timeout if self.data_type == "full" else DS_RD_SETTING.debug_timeout
            ),
        )

        # 2) check the result and stdout after reruning the model.

        # Read the content of files submission.csv and scores.csv before execution
        submission_content_before = (
            (implementation.workspace_path / "submission.csv").read_text()
            if (implementation.workspace_path / "submission.csv").exists()
            else NO_SUB
        )
        scores_content_before = (
            (implementation.workspace_path / "scores.csv").read_text()
            if (implementation.workspace_path / "scores.csv").exists()
            else NO_SCORE
        )

        # Remove the files submission.csv and scores.csv
        implementation.execute(env=env, entry=get_clear_ws_cmd(stage="before_inference"))

        # Execute the main script
        stdout = remove_eda_part(implementation.execute(env=env, entry="python main.py"))

        # walk model_folder and list the files
        model_folder_files = [
            str(file.relative_to(implementation.workspace_path)) for file in model_folder.iterdir() if file.is_file()
        ]

        # this will assert the generation of necessary files
        for f in ["submission.csv", "scores.csv"]:
            if not (implementation.workspace_path / f).exists():
                err_msg = f"{f} does not exist. The model is not dumped. Make sure that the required files, like submission.csv and scores.csv, are created even if you bypass the model training step by loading the saved model file directly."
                return CoSTEERSingleFeedback(
                    execution=err_msg,
                    return_checking=err_msg,
                    code=err_msg,
                    final_decision=False,
                )

        # Check if scores contain NaN (values)
        score_df = pd.read_csv((implementation.workspace_path / "scores.csv"), index_col=0)
        if score_df.isnull().values.any():
            nan_locations = score_df[score_df.isnull().any(axis=1)]
            err_msg = f"\n[Error] The scores dataframe contains NaN values at the following locations:\n{nan_locations}"
            return CoSTEERSingleFeedback(
                execution=err_msg,
                return_checking=err_msg,
                code=err_msg,
                final_decision=False,
            )

        submission_content_after = (
            (implementation.workspace_path / "submission.csv").read_text()
            if (implementation.workspace_path / "submission.csv").exists()
            else NO_SUB
        )
        scores_content_after = (
            (implementation.workspace_path / "scores.csv").read_text()
            if (implementation.workspace_path / "scores.csv").exists()
            else NO_SCORE
        )

        system_prompt = T(".prompts:dump_model_eval.system").r()
        user_prompt = T(".prompts:dump_model_eval.user").r(
            stdout=stdout.strip(),
            code=implementation.all_codes,
            model_folder_files=model_folder_files,
            scores_content_before=scores_content_before,
            scores_content_after=scores_content_after,
        )

        csfb = build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        if DS_RD_SETTING.model_dump_check_level == "high":
            # Read the content of files submission.csv and scores.csv after execution
            # Check if the content has changed
            # excactly same checking. But it will take more user's time
            if scores_content_before != scores_content_after:
                return_msg = "\n[Error] The content of scores.csv has changed. Please check the code to ensure that the model is dumped correctly, and rerun the code to use the model directly without retraining it."
                return_msg += f"\nBefore:\n{scores_content_before}\nAfter:\n{scores_content_after}"
                if submission_content_before != submission_content_after:
                    # If the scores file changes, display the two contents and append it into the return_checking
                    return_msg = "[Error] The content of submission.csv has changed. Please check the code to ensure that the model is dumped correctly, and rerun the code to use the model directly without retraining it."
                csfb.return_checking = (csfb.return_checking or "") + return_msg
        return csfb
