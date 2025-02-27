import json
import re
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.agent.workflow import build_cls_from_json_with_retry
from rdagent.utils.env import DockerEnv, DSDockerConf
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

        ds_docker_conf = DSDockerConf()
        # TODO: we should /= 20 for the timeout period on debug component
        ds_docker_conf.extra_volumes = {
            f"{DS_RD_SETTING.local_data_path}/sample/{self.scen.competition}": "/kaggle/input"
        }
        de = DockerEnv(conf=ds_docker_conf)

        # TODO: do we need to clean the generated temporary content?
        fname = "test/feature_test.py"
        test_code = (DIRNAME / "eval_tests" / "feature_test.txt").read_text()
        implementation.inject_files(**{fname: test_code})

        stdout = implementation.execute(env=de, entry=f"python {fname}")

        if "main.py" in implementation.file_dict:
            workflow_stdout = implementation.execute(env=de, entry="python main.py")
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

        return build_cls_from_json_with_retry(FeatureEvalFeedback, system_prompt=system_prompt, user_prompt=user_prompt)
