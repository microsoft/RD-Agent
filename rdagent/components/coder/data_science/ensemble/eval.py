import json
from pathlib import Path

from jinja2 import Environment, StrictUndefined

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

DIRNAME = Path(__file__).absolute().resolve().parent

EnsembleEvalFeedback = CoSTEERSingleFeedback


class EnsembleCoSTEEREvaluator(CoSTEEREvaluator):
    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> EnsembleEvalFeedback:

        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return EnsembleEvalFeedback(
                execution="This task has failed too many times, skip implementation.",
                code="This task has failed too many times, skip implementation.",
                return_checking="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {
            f"{DS_RD_SETTING.local_data_path}/sample/{self.scen.competition}": "/kaggle/input"
        }
        de = DockerEnv(conf=ds_docker_conf)

        fname = "test/ensemble_test.txt"
        test_code = (DIRNAME / "eval_tests" / "ensemble_test.txt").read_text()
        test_code = (
            Environment(undefined=StrictUndefined)
            .from_string(test_code)
            .render(
                model_names=[
                    fn[:-3] for fn in implementation.file_dict.keys() if fn.startswith("model_") and "test" not in fn
                ]
            )
        )

        implementation.inject_files(**{fname: test_code})
        stdout, ret_code = implementation.execute_ret_code(env=de, entry=f"python {fname}")

        stdout += f"\nNOTE: the above scripts run with return code {ret_code}"

        if "main.py" in implementation.file_dict:
            workflow_stdout = implementation.execute(env=de, entry="python main.py")
        else:
            workflow_stdout = None

        system_prompt = T(".prompts:ensemble_eval.system").r(
            task_desc=target_task_information,
            test_code=test_code,
            code=implementation.file_dict["ensemble.py"],
            workflow_stdout=workflow_stdout,
            workflow_code=implementation.all_codes,
        )
        user_prompt = T(".prompts:ensemble_eval.user").r(
            stdout=stdout,
            workflow_stdout=workflow_stdout,
        )
        efb = build_cls_from_json_with_retry(EnsembleEvalFeedback, system_prompt=system_prompt, user_prompt=user_prompt)
        efb.final_decision = efb.final_decision and ret_code == 0
        return efb
