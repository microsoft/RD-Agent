import json
from rdagent.core.experiment import FBWorkspace, Task
from pathlib import Path
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
    CoSTEERSingleFeedbackDeprecated,
)
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf
from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.oai.llm_utils import APIBackend

DIRNAME = Path(__file__).absolute().resolve().parent

WorkflowSingleFeedback = CoSTEERSingleFeedback
WorkflowMultiFeedback = CoSTEERMultiFeedback

class WorkflowGeneralCaseSpecEvaluator(CoSTEEREvaluator):
    """
    Motivation case:
    - Simplest case, we already split the data into train_data, valid_data, and test_data. We require the model to learn (optionally validate on valid data), and infer on test data.

    Test workflow:
    - Build train, valid, and test data to run it, and test the output (e.g., shape, etc.)
    """
    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedbackDeprecated:
        target_task_information = target_task.get_task_information()
        if (
            queried_knowledge is not None
            and target_task_information in queried_knowledge.success_task_to_knowledge_dict
        ):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return WorkflowSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                shape_feedback="This task has failed too many times, skip implementation.",
                value_feedback="This task has failed too many times, skip implementation.",
                code_feedback="This task has failed too many times, skip implementation.",
                final_feedback="This task has failed too many times, skip implementation.",
                final_decision=False,
            )
        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}": "/kaggle/input"}
        de = DockerEnv(conf=ds_docker_conf)
        fname = "main.py"
        stdout = implementation.execute(env=de, entry=f"python {fname}")
        system_prompt = T(".prompts:workflow_eval.system").r(
            scenario="No scenario information yet.",
            spec=implementation.code_dict["spec/workflow.md"]
        )
        user_prompt = T(".prompts:workflow_eval.user").r(
            stdout=stdout,
            code=implementation.code_dict["main.py"],
        )
        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
        return WorkflowSingleFeedback(**json.loads(resp))
        