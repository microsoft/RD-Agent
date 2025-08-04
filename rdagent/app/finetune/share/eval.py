from pathlib import Path

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.core.scenario import Scenario
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry


class PrevModelLoadEvaluator(CoSTEEREvaluator):
    """This evaluator checks whether the code actually loads a model from `prev_model`."""

    def __init__(self, scen: Scenario):
        super().__init__(scen)

    def evaluate(
        self, target_task: Task, implementation: FBWorkspace, gt_implementation: FBWorkspace, *args, **kwargs
    ) -> CoSTEERSingleFeedback:
        data_source_path = T("scenarios.data_science.share:scen.input_path").r()
        prev_model_dir = Path(data_source_path) / "prev_model"

        # 1) Inspect the code itself for references to prev_model loading
        code_str = implementation.file_dict["main.py"]
        code_contain_prev = "prev_model" in code_str
        print(f"Code references prev_model: {code_contain_prev}")
        if not code_contain_prev:
            err = (
                "No evidence found that your code loads a model from `prev_model`. "
                "Please check that you are calling the correct load function "
                f"and pointing it to the `{prev_model_dir}` directory."
            )
            return CoSTEERSingleFeedback(
                execution=err,
                return_checking=err,
                code=err,
                final_decision=False,
            )

        system_prompt = T(".prompts:prev_model_eval.system").r()
        user_prompt = T(".prompts:prev_model_eval.user").r(
            code=implementation.all_codes,
        )

        csfb = build_cls_from_json_with_retry(
            CoSTEERSingleFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        return csfb
