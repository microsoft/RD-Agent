import json
import re
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry
from rdagent.utils.fmt import shrink_text

DIRNAME = Path(__file__).absolute().resolve().parent

FeatureEvalFeedback = CoSTEERSingleFeedback


class FeatureCoSTEEREvaluator(CoSTEEREvaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> FeatureEvalFeedback:
        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return FeatureEvalFeedback(
                execution="This task has failed too many times, skip implementation.",
                return_checking="This task has failed too many times, skip implementation.",
                code="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        env = get_ds_env(
            extra_volumes={
                f"{DS_RD_SETTING.local_data_path}/sample/{self.scen.competition}": T(
                    "scenarios.data_science.share:scen.input_path"
                ).r()
            }
        )

        # TODO: do we need to clean the generated temporary content?
        fname = "test/feature_test.py"
        test_code = (DIRNAME / "eval_tests" / "feature_test.txt").read_text()
        implementation.inject_files(**{fname: test_code})

        stdout, ret_code = implementation.execute_ret_code(env=env, entry=f"python {fname}")

        if "main.py" in implementation.file_dict and ret_code == 0:
            workflow_stdout = implementation.execute(env=env, entry="python main.py")
            workflow_stdout = re.sub(r"=== Start of EDA part ===(.*)=== End of EDA part ===", "", workflow_stdout)
        else:
            workflow_stdout = None

        system_prompt = T(".prompts:feature_eval.system").r(
            task_desc=target_task.get_task_information(),
            test_code=test_code,
            code=implementation.file_dict["feature.py"],
            workflow_stdout=workflow_stdout,
            workflow_code=implementation.all_codes,
        )
        user_prompt = T(".prompts:feature_eval.user").r(
            stdout=shrink_text(stdout),
            workflow_stdout=workflow_stdout,
        )

        fb = build_cls_from_json_with_retry(
            FeatureEvalFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=FeatureEvalFeedback.val_and_update_init_dict,
        )
        fb.final_decision = fb.final_decision and ret_code == 0

        return fb
