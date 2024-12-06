# tess successfully running.
# (GPT) if it aligns with the spec & rationality of the spec.
import json
from abc import abstractclassmethod
from dataclasses import dataclass
from os import system
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
)
from rdagent.core.evaluation import Feedback
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task, Workspace
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DSDockerConf, DockerEnv
from pathlib import Path

DIRNAME = Path(__file__).absolute().resolve().parent

# TODO:
# 1. It seems logically sound, but we currently lack a scenario to apply it.
# 2. If it proves to be useful, relocate it to a more general location.
#
# class FBWorkspaceExeFeedback(Feedback):
#     """
#     It pairs with FBWorkspace in the abstract level.
#     """
#     # ws: FBWorkspace   # potential
#     stdout: str


@dataclass
class DataLoaderEvalFeedback(Feedback):
    execution: str
    checking: str  # inlucding every check in the testing (constraints about the generated value)
    code: str
    final_decision: bool


class DataLoaderCoSTEEREvaluator(CoSTEEREvaluator):

    def evaluate(
        self,
        target_task: Task,
        implementation: FBWorkspace,
        gt_implementation: FBWorkspace,
        queried_knowledge: QueriedKnowledge = None,
        **kwargs,
    ) -> CoSTEERSingleFeedback:

        target_task_information = target_task.get_task_information()
        if (queried_knowledge is not None and
                target_task_information in queried_knowledge.success_task_to_knowledge_dict):
            return queried_knowledge.success_task_to_knowledge_dict[target_task_information].feedback
        elif queried_knowledge is not None and target_task_information in queried_knowledge.failed_task_info_set:
            return CoSTEERSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                shape_feedback="This task has failed too many times, skip implementation.",
                value_feedback="This task has failed too many times, skip implementation.",
                code_feedback="This task has failed too many times, skip implementation.",
                final_feedback="This task has failed too many times, skip implementation.",
                final_decision=False,
            )

        de = DockerEnv(conf=DSDockerConf())

        # TODO: do we need to clean the generated tempory content?
        fname = "data_loader_test.py"
        with (DIRNAME / "eval_tests" / "data_loader_test.py").open("r") as f:
            test_code = f.read()
            implementation.inject_code(**{fname: test_code})
        stdout = implementation.execute(env=de, entry=f"python {fname}")

        system_prompt = T(".prompts:data_loader_eval.system").r(test_code=test_code)
        user_prompt = T(".prompts:data_loader_eval.user").r(stdout=stdout)

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
        return DataLoaderEvalFeedback(**json.loads(resp))
