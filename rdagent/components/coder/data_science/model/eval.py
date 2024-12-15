"""
Beyond previous tests
- 
"""

import json
from pathlib import Path

from rdagent.app.data_science.conf import DS_RD_SETTING
from rdagent.components.coder.CoSTEER.evaluators import (
    CoSTEEREvaluator,
    CoSTEERMultiFeedback,
    CoSTEERSingleFeedback,
    CoSTEERSingleFeedbackDeprecated,
)
from rdagent.components.coder.data_science.model.eva_utils import (
    ModelCodeEvaluator,
    ModelFinalEvaluator,
    expected_shape_evaluate,
)
from rdagent.core.evolving_framework import QueriedKnowledge
from rdagent.core.experiment import FBWorkspace, Task, Workspace
from rdagent.oai.llm_utils import APIBackend
from rdagent.utils.agent.tpl import T
from rdagent.utils.env import DockerEnv, DSDockerConf

DIRNAME = Path(__file__).absolute().resolve().parent


ModelSingleFeedback = CoSTEERSingleFeedback
ModelMultiFeedback = CoSTEERMultiFeedback


# Below are unit tests for testing the specification of the implemented model ------------------
#
class ModelGeneralCaseSpecEvaluator(CoSTEEREvaluator):
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
            return ModelSingleFeedback(
                execution_feedback="This task has failed too many times, skip implementation.",
                shape_feedback="This task has failed too many times, skip implementation.",
                value_feedback="This task has failed too many times, skip implementation.",
                code_feedback="This task has failed too many times, skip implementation.",
                final_feedback="This task has failed too many times, skip implementation.",
                final_decision=False,
            )
        # assert isinstance(target_task, ModelTask)

        batch_size = 8
        assert isinstance(implementation, FBWorkspace)
        """model_execution_feedback, pred_list= implementation.execute(
            batch_size=batch_size,
        )"""
        ds_docker_conf = DSDockerConf()
        ds_docker_conf.extra_volumes = {f"{DS_RD_SETTING.local_data_path}/{self.scen.competition}": "/kaggle/input"}
        de = DockerEnv(conf=ds_docker_conf)
        fname = "model_execute.py"
        with (DIRNAME / "eval_tests" / "model_execute.py").open("r") as f:
            test_code = f.read()
            implementation.inject_code(**{fname: test_code})
        stdout = implementation.execute(env=de, entry=f"python {fname}")
        system_prompt = T(".prompts:model_eval.system").r(
            test_code=test_code,
            scenario="No scenario information yet.",
            spec=target_task.spec,
        )
        user_prompt = T(".prompts:model_eval.user").r(
            stdout=stdout,
            code=implementation.code_dict["model01.py"],
        )
        resp = APIBackend().build_messages_and_create_chat_completion(user_prompt, system_prompt, json_mode=True)
        return ModelSingleFeedback(**json.loads(resp))

    """feedback"""


class XXX2SpecEval:
    """
    Based on XXX1SpecEval, but considering the following case:

    Motivation case:
    - Sometimes we don't need validation (e.g., simple models not prone to overfitting, or data is too scarce to split).

    Test workflow:
    - Build train and test data to run it, and test the output (e.g., shape, etc.)
    - valid_data == None
    """


class XXX3SpecEval:
    """
    Motivation case:
    - We need to tune hyperparameters.

    Test workflow:
    - Input:
        - Build train and valid data
        - test == None
        - Hyperparameters are not blank
    - Output:
        - The early stop hyperparameters must be returned
    """


class XXX4SpecEval:
    """
    Motivation case:
    - After obtaining good hyperparameters, we retrain the model.

    Test workflow:
    - Test1: Since we have already tested it in XXX2SpecEval, we'll focus on another aspect.
        - Input:
            - Build train and test data
            - valid == None
            - Previous good hyperparameters (a parameter representing early stop)
    - Test2: Ensure the hyperparameters are 1) being used, and 2) the model remains stable.
        - Different hyperparameters will yield different results
        - Same hyperparameters will yield the same results
    """
