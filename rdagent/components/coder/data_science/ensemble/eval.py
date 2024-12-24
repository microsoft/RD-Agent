import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERSingleFeedback,
)
from rdagent.core.evaluation import Feedback
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
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

        de = DockerEnv(conf=DSDockerConf())

        fname = "ensemble_test.py"
        with (DIRNAME / "eval_tests" / "ensemble_test.py").open("r") as f:
            test_code = f.read()
            implementation.inject_files(**{fname: test_code})
        stdout = implementation.execute(env=de, entry=f"python {fname}")

        system_prompt = T(".prompts:ensemble_eval.system").r(
            test_code=test_code, code=implementation.file_dict["ensemble.py"]
        )
        user_prompt = T(".prompts:ensemble_eval.user").r(stdout=stdout)

        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
        return EnsembleEvalFeedback(**json.loads(resp))
