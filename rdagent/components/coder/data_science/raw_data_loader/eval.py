# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
import json
import re
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.components.coder.CoSTEER.knowledge_management import (
    CoSTEERQueriedKnowledgeV2,
)
from rdagent.components.coder.data_science.conf import get_ds_env
from rdagent.components.coder.data_science.utils import remove_eda_part
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry

DIRNAME = Path(__file__).absolute().resolve().parent

DataLoaderEvalFeedback = CoSTEERSingleFeedback


class DataLoaderCoSTEEREvaluator(CoSTEEREvaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: CoSTEERQueriedKnowledgeV2 = None,
        **kwargs,
    ) -> DataLoaderEvalFeedback:
        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return DataLoaderEvalFeedback(
                execution="This task has failed too many times, skip implementation.",
                return_checking="This task has failed too many times, skip implementation.",
                code="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        env = get_ds_env(extra_volumes={self.scen.debug_path: T("scenarios.data_science.share:scen.input_path").r()})

        # TODO: do we need to clean the generated temporary content?
        fname = "test/data_loader_test.py"
        test_code = (DIRNAME / "eval_tests" / "data_loader_test.txt").read_text()
        implementation.inject_files(**{fname: test_code})
        result = implementation.run(env=env, entry=f"python {fname}")
        stdout = result.stdout
        ret_code = result.exit_code
        match = re.search(r"(.*?)=== Start of EDA part ===(.*)=== End of EDA part ===(.*)", stdout, re.DOTALL)
        stdout_part_1, eda_output, stdout_part_2 = match.groups() if match else (stdout, None, "")
        stdout = stdout_part_1 + stdout_part_2
        if eda_output is not None and len(eda_output.split(" ")) > 10000:
            eda_output += "Length of EDA output is too long, truncated. Please reject this implementation and motivate it to reduce the length of EDA output."

        if "main.py" in implementation.file_dict and ret_code == 0:
            workflow_stdout = implementation.execute(env=env, entry="python main.py")
            workflow_stdout = remove_eda_part(workflow_stdout)
        else:
            workflow_stdout = None

        system_prompt = T(".prompts:data_loader_eval.system").r(
            task_desc=target_task.get_task_information(),
            test_code=test_code,
            code=implementation.file_dict["load_data.py"],
            workflow_stdout=workflow_stdout,
            workflow_code=implementation.all_codes,
        )
        user_prompt = T(".prompts:data_loader_eval.user").r(
            stdout=stdout,
            eda_output=eda_output,
            workflow_stdout=workflow_stdout,
        )

        fb = build_cls_from_json_with_retry(
            DataLoaderEvalFeedback,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            init_kwargs_update_func=DataLoaderEvalFeedback.val_and_update_init_dict,
        )
        fb.final_decision = fb.final_decision and ret_code == 0

        return fb
