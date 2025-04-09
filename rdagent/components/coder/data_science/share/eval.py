from pathlib import Path

import pandas as pd
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER import CoSTEERMultiFeedback
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent

PipelineSingleFeedback = CoSTEERSingleFeedback
PipelineMultiFeedback = CoSTEERMultiFeedback

class ModelDumpEvaluator(CoSTEEREvaluator):
    """This evaluator assumes that it runs after the model"""
    def evaluate(self, target_task: Task, implementation: FBWorkspace, gt_implementation: FBWorkspace, **kwargs) -> CoSTEERSingleFeedback:

        model_folder = implementation.workspace_path / "models"
        # 1) Check if the model_folder is not empty
        if not model_folder.exists() or not any(model_folder.iterdir()):
            err_msg = "Model folder is empty or does not exist. The model is not dumped. Please add code to dump the trained models in `models` sub folder"
            return CoSTEERSingleFeedback(
                execution=err_msg,
                return_checking=err_msg,
                code=err_msg,
                final_decision=False,
            )
        env = get_ds_env()

        # 2) check the result and stdout after reruning the model.
        # Read the content of files submission.csv and scores.csv before execution
        submission_content_before = (implementation.workspace_path / "submission.csv").read_text()
        scores_content_before = (implementation.workspace_path / "scores.csv").read_text()

        # Remove the files submission.csv and scores.csv
        implementation.execute(env=env, entry="rm submission.csv scores.csv")

        # Execute the main script
        stdout = implementation.execute(env=env, entry="python main.py")

        # Read the content of files submission.csv and scores.csv after execution
        submission_content_after = (implementation.workspace_path / "submission.csv").read_text()
        scores_content_after = (implementation.workspace_path / "scores.csv").read_text()


        system_prompt = T(".prompts:dump_model_eval.system").r()
        user_prompt = T(".prompts:dump_model_eval.user").r(
            stdout=stdout.strip(),
            code=implementation.all_codes
        )

        csfb = build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        # Check if the content has changed
        if submission_content_before != submission_content_after:
            return_msg = "[Error] The content of submission.csv has changed. Please check the code to ensure that the model is dumped correctly, and rerun the code to use the model directly without retraining it."
            if scores_content_before != scores_content_after:
                # If the scores file changes, display the two contents and append it into the return_checking
                return_msg = "\n[Error] The content of scores.csv has changed. Please check the code to ensure that the model is dumped correctly, and rerun the code to use the model directly without retraining it."
                return_msg += f"\nBefore:\n{scores_content_before}\nAfter:\n{scores_content_after}"
            csfb.return_checking = (csfb.return_checking or "") +  return_msg
        return csfb

