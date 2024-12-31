import json
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
    CoSTEERSingleFeedbackDeprecated,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf
from rdagent.core.exception import CoderError

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
                execution="This task has failed too many times, skip implementation.",
                return_checking="This task has failed too many times, skip implementation.",
                code="This task has failed too many times, skip implementation.",
                final_decision=False,
            )
        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}": "/kaggle/input"}
        de = DockerEnv(conf=ds_docker_conf)
        fname = "main.py"
        stdout = implementation.execute(env=de, entry=f"python {fname}")
        
        # Check if the submission file and score file are generated
        submission_fp = implementation.experiment_workspace.workspace_path / "submission.csv"
        score_fp = implementation.experiment_workspace.workspace_path / "scores.csv"
        if not submission_fp.exists():
            raise CoderError("Submission file (submission.csv) is not generated.")
        if not score_fp.exists():
            raise CoderError("Metrics file (scores.csv) is not generated.")

        if stdout is None:
            stdout = "The execution exceeded the time limit, and no stdout information has been generated yet."
        system_prompt = T(".prompts:workflow_eval.system").r(
            scenario="No scenario information yet.", spec=implementation.file_dict["spec/workflow.md"]
        )
        user_prompt = T(".prompts:workflow_eval.user").r(
            stdout=stdout,
            code=implementation.file_dict["main.py"],
        )
        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
        return WorkflowSingleFeedback(**json.loads(resp))
